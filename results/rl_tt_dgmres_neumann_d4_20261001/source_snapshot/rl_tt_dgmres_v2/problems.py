"""Broader index-one graph-Laplacian problems without dense full tensors.

Every local graph is connected with positive edge weights.  Its Laplacian is
symmetric positive semidefinite with precisely the constant null vector, so
a positive Kronecker sum has index one.  The sum is divided by the sum of
its weighted local largest eigenvalues: its spectral norm is one.  Only
small n-by-n matrices are diagonalized, including for high-dimensional OOD
cases.  References remain evaluation data and never enter ``online_view``.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import json
import math
import random
from typing import Any, Mapping

import torch

from adaptive_tt_dgmres.backend.paper_tt_dgmres import (
    neumann_1d_laplacian, tt_norm_stable, variable_kron_sum_tt_operator,
)
from adaptive_tt_dgmres.backend.paper_tt_gmres import tt_dot, tt_rank1, tt_scale
from rl_tt_dgmres_v1.problems import (
    Problem as _V1Problem,
    ProblemSpec as _V1ProblemSpec,
    _manufactured_reference,
    audit_solution,
)


FAMILIES = (
    "symmetric_neumann_path",
    "symmetric_weighted_neumann_path",
    "symmetric_weighted_cycle",
)


@dataclass(frozen=True)
class ProblemSpec(_V1ProblemSpec):
    # Each axis has n-1 path edges or n cycle edges.  For a cycle the final
    # edge joins n-1 to 0.  Missing weights mean all edges have weight one.
    edge_weights: tuple[tuple[float, ...], ...] = ()
    curriculum_tier: str = "standard"


@dataclass
class Problem(_V1Problem):
    local_edge_weights: tuple[tuple[float, ...], ...] = ()
    local_lambda_min_positive: tuple[float, ...] = ()
    local_lambda_max: tuple[float, ...] = ()

    def construction_metadata(self) -> dict[str, Any]:
        result = super().construction_metadata()
        result.update({
            "edge_weights": [list(values) for values in self.local_edge_weights],
            "local_lambda_min_positive": list(self.local_lambda_min_positive),
            "local_lambda_max": list(self.local_lambda_max),
            "normalization": "sum(axis_weight[j] * L[j]) / sum(axis_weight[j] * lambda_max(L[j]))",
            "normalized_axis_weight_meaning": "spectral share of each normalized summand",
            "curriculum_tier": getattr(self.spec, "curriculum_tier", "standard"),
        })
        return result


def spec_from_dict(values: Mapping[str, Any]) -> ProblemSpec:
    """Restore a JSON-serialized v1 or v2 specification, rejecting typos."""
    payload = dict(values)
    known = {field.name for field in fields(ProblemSpec)}
    extra = payload.keys() - known
    if extra:
        raise ValueError(f"Unknown problem specification fields: {sorted(extra)}")
    if "axis_weights" in payload:
        payload["axis_weights"] = tuple(payload["axis_weights"])
    if "edge_weights" in payload:
        payload["edge_weights"] = tuple(tuple(axis) for axis in payload["edge_weights"])
    return ProblemSpec(**payload)


def _validated_weights(spec: _V1ProblemSpec) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    if not isinstance(spec.n, int) or not isinstance(spec.dims, int) or spec.n < 2 or spec.dims < 1:
        raise ValueError("Require integer n >= 2 and dims >= 1.")
    if spec.family not in FAMILIES:
        raise ValueError(f"Unknown graph-Laplacian family: {spec.family}")
    if spec.family == "symmetric_weighted_cycle" and spec.n < 3:
        raise ValueError("A simple cycle requires n >= 3.")
    if not math.isfinite(spec.anisotropy) or spec.anisotropy <= 0:
        raise ValueError("anisotropy must be finite and positive.")
    if not isinstance(spec.rhs_rank, int) or spec.rhs_rank < 1:
        raise ValueError("rhs_rank must be a positive integer.")
    if not isinstance(spec.rhs_frequency, int) or not 1 <= spec.rhs_frequency < spec.n:
        raise ValueError("rhs_frequency must be an integer in [1, n-1].")
    if not math.isfinite(spec.contamination) or spec.contamination < 0:
        raise ValueError("contamination must be finite and nonnegative.")
    weights = tuple(float(value) for value in spec.axis_weights) if spec.axis_weights else tuple(
        spec.anisotropy ** (axis / max(spec.dims - 1, 1)) for axis in range(spec.dims)
    )
    if len(weights) != spec.dims or any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("axis_weights must contain one finite positive weight per dimension.")
    if not math.isfinite(math.fsum(weights)):
        raise ValueError("Sum of axis weights must be finite.")
    edge_count = spec.n if spec.family == "symmetric_weighted_cycle" else spec.n - 1
    supplied = getattr(spec, "edge_weights", ())
    edges = tuple(tuple(float(value) for value in axis) for axis in supplied) if supplied else tuple(
        (1.0,) * edge_count for _ in range(spec.dims)
    )
    if len(edges) != spec.dims or any(len(axis) != edge_count for axis in edges):
        raise ValueError("edge_weights must have one correctly sized edge sequence per axis.")
    if any(not math.isfinite(value) or value <= 0 for axis in edges for value in axis):
        raise ValueError("Every edge weight must be finite and strictly positive.")
    if spec.family == "symmetric_neumann_path" and any(value != 1.0 for axis in edges for value in axis):
        raise ValueError("Nonuniform edge weights require symmetric_weighted_neumann_path.")
    return weights, edges


def _local_data(spec: _V1ProblemSpec) -> tuple[Any, ...]:
    weights, edges = _validated_weights(spec)
    local_matrices = []
    minimums, maximums = [], []
    for axis_edges in edges:
        if spec.family == "symmetric_neumann_path":
            local = neumann_1d_laplacian(spec.n, dtype=torch.float64)
            minimum = 4.0 * (spec.n - 1) ** 2 * math.sin(math.pi / (2.0 * spec.n)) ** 2
            maximum = 4.0 * (spec.n - 1) ** 2 * math.cos(math.pi / (2.0 * spec.n)) ** 2
        else:
            local = torch.zeros((spec.n, spec.n), dtype=torch.float64)
            for edge, value in enumerate(axis_edges):
                left, right = edge, (edge + 1) % spec.n
                local[left, left] += value
                local[right, right] += value
                local[left, right] -= value
                local[right, left] -= value
            local *= (spec.n - 1) ** 2
            spectrum = torch.linalg.eigvalsh(local)
            minimum, maximum = float(spectrum[1]), float(spectrum[-1])
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum <= 0 or maximum <= 0:
            raise ValueError("Local graph spectrum is nonfinite or numerically disconnected.")
        local_matrices.append(local)
        minimums.append(minimum)
        maximums.append(maximum)
    raw_norm = math.fsum(weight * maximum for weight, maximum in zip(weights, maximums))
    if not math.isfinite(raw_norm) or raw_norm <= 0:
        raise ValueError("Unnormalized operator norm must be finite and positive.")
    normalized = [local * (weight / raw_norm) for local, weight in zip(local_matrices, weights)]
    return weights, edges, tuple(minimums), tuple(maximums), raw_norm, normalized


def _group_from_matrices(spec: _V1ProblemSpec, matrices: list[torch.Tensor]) -> str:
    # Identity uses the matrices in matvec, not family labels, split labels,
    # seeds or RHS parameters.  Thus the weighted family with all-one edges
    # groups with the original path family, and common scaling cancels.
    payload = {
        "n": spec.n,
        "dims": spec.dims,
        "normalized_local_matrices": [
            [format(float(value), ".13g") if value != 0 else "0" for value in matrix.reshape(-1)]
            for matrix in matrices
        ],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return "graph_laplacian_" + digest[:24]


def operator_group_key(spec: _V1ProblemSpec) -> str:
    """Canonical operator identity, with roundoff-equivalent entries grouped."""
    return _group_from_matrices(spec, _local_data(spec)[-1])


def build_problem(spec: _V1ProblemSpec) -> Problem:
    """Construct a connected-graph index-one problem using TT arithmetic."""
    weights, edges, minimums, maximums, raw_norm, local_matrices = _local_data(spec)
    operator = variable_kron_sum_tt_operator(
        local_matrices, name=spec.name, exact_norm=1.0, dtype=torch.float64,
    )
    q0 = tt_rank1([
        torch.full((spec.n,), 1.0 / math.sqrt(spec.n), dtype=torch.float64)
        for _ in range(spec.dims)
    ])

    def projector(vector: Any) -> Any:
        coefficient = tt_dot(q0, vector)
        return vector - tt_scale(q0, coefficient) if coefficient != 0.0 else vector

    # A DCT-filtered low-rank tensor is valid for all graphs because its
    # constant component vanishes.  Frequency is a spatial smoothness knob,
    # not a claim that DCT vectors diagonalize a weighted graph or cycle.
    reference = _manufactured_reference(spec)
    ax = operator.matvec(reference)
    ax_norm = tt_norm_stable(ax)
    rhs = ax + tt_scale(q0, spec.contamination * ax_norm) if spec.contamination else ax
    fingerprint = hashlib.sha256()
    fingerprint.update(f"{spec.n}:{spec.dims}:float64".encode("ascii"))
    for matrix in local_matrices:
        fingerprint.update(matrix.contiguous().numpy().tobytes())
    return Problem(
        spec=spec, operator=operator, rhs=rhs, reference_solution=reference,
        projector=projector, q0=q0, operator_norm=1.0,
        rhs_norm=tt_norm_stable(rhs), ab_norm=tt_norm_stable(operator.matvec(rhs)),
        lambda_min_positive=min(weight * minimum for weight, minimum in zip(weights, minimums)) / raw_norm,
        axis_weights=weights,
        normalized_axis_weights=tuple(weight * maximum / raw_norm for weight, maximum in zip(weights, maximums)),
        unnormalized_operator_norm=raw_norm,
        operator_group_key=_group_from_matrices(spec, local_matrices),
        operator_fingerprint=fingerprint.hexdigest(),
        local_edge_weights=edges,
        local_lambda_min_positive=minimums,
        local_lambda_max=maximums,
    )


def _balanced_values(rng: random.Random, choices: tuple[Any, ...], count: int) -> list[Any]:
    result: list[Any] = []
    while len(result) < count:
        block = list(choices)
        rng.shuffle(block)
        result.extend(block)
    return result[:count]


def make_splits(
    seed: int = 20260929,
    train_count: int = 48,
    validation_count: int = 12,
    test_count: int = 12,
    ood_count: int = 6,
) -> dict[str, list[ProblemSpec]]:
    """Generate balanced coverage and disjoint actual operator groups.

    Every nine ID cases cover n=4,...,12; every five cover d=2,...,6 and
    requested reference TT rank=1,...,5.  Every three cover all families.
    OOD uses d=7,8 with n=4,5,6 and rank <= 3 to bound audit cost.  The tier
    is fixed from construction size alone; it is never based on test results.
    Counts can be reduced for smoke runs without changing existing prefixes.
    """
    counts = {"train": train_count, "validation": validation_count, "test": test_count, "ood": ood_count}
    if any(not isinstance(count, int) or count < 0 for count in counts.values()):
        raise ValueError("Split counts must be nonnegative integers.")
    result: dict[str, list[ProblemSpec]] = {}
    used: set[str] = set()
    for split_index, (split, count) in enumerate(counts.items()):
        stream_seed = seed + 104729 * (split_index + 1)
        rng = random.Random(stream_seed)
        ood = split == "ood"
        # Independent streams keep all prefixes unchanged when counts change.
        sizes = _balanced_values(random.Random(stream_seed + 11), (4, 5, 6) if ood else tuple(range(4, 13)), count)
        dimensions = _balanced_values(random.Random(stream_seed + 23), (7, 8) if ood else tuple(range(2, 7)), count)
        ranks = _balanced_values(random.Random(stream_seed + 37), (1, 2, 3) if ood else tuple(range(1, 6)), count)
        families = _balanced_values(random.Random(stream_seed + 53), FAMILIES, count)
        specs = []
        for case in range(count):
            n, dims, rank, family = sizes[case], dimensions[case], ranks[case], families[case]
            anisotropy = rng.choice((8.0, 12.0, 20.0) if ood else (1.0, 2.0, 4.0, 8.0))
            rhs_seed = rng.randrange(1, 2**31 - 1)
            frequency = rng.choice((1, max(1, n // 2), n - 1))
            tier = "ood" if ood else ("intro" if n <= 6 and dims <= 3 and rank <= 2 else (
                "advanced" if dims >= 5 or rank >= 4 else "intermediate"
            ))
            while True:
                weights = tuple(
                    anisotropy ** (axis / (dims - 1)) * math.exp(rng.uniform(-0.25, 0.25))
                    for axis in range(dims)
                )
                edge_count = n if family == "symmetric_weighted_cycle" else n - 1
                edges = () if family == "symmetric_neumann_path" else tuple(
                    tuple(math.exp(rng.uniform(-0.8, 0.8)) for _ in range(edge_count))
                    for _ in range(dims)
                )
                spec = ProblemSpec(
                    name=f"{split}_{case:03d}_n{n}_d{dims}_r{rank}",
                    n=n, dims=dims, seed=rhs_seed, anisotropy=anisotropy,
                    rhs_rank=rank, contamination=0.0 if case % 2 == 0 else 0.1,
                    family=family, rhs_frequency=frequency, axis_weights=weights,
                    edge_weights=edges, curriculum_tier=tier,
                )
                group = operator_group_key(spec)
                if group not in used:
                    used.add(group)
                    specs.append(spec)
                    break
        result[split] = specs
    return result


__all__ = [
    "FAMILIES", "ProblemSpec", "Problem", "spec_from_dict", "build_problem",
    "audit_solution", "operator_group_key", "make_splits",
]
