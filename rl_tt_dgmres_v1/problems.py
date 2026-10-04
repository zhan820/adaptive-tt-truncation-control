"""Manufactured index-one problems and independent floating-point TT audits.

The symmetric path Laplacian has eigenvalues
``4 (n-1)^2 sin(k*pi/(2*n))**2``, ``k=0,...,n-1``.  Positive
weighted Kronecker sums therefore have one constant null vector, index one,
and an analytic largest and smallest positive eigenvalue.  The operator is
scaled to spectral norm one.  No full n**dims tensor is formed here.

Reference solutions are evaluation data only.  ``Problem.online_view`` is
the deliberately narrower interface for a solver/controller.  Audit norms
use untruncated TT arithmetic and a QR gauge transformation; they are
floating-point checks, not rigorous error bounds or certified arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch
import tntorch as tn

from adaptive_tt_dgmres.backend.paper_tt_dgmres import (
    neumann_1d_laplacian,
    tt_norm_stable,
    variable_kron_sum_tt_operator,
)
from adaptive_tt_dgmres.backend.paper_tt_gmres import tt_dot, tt_rank1, tt_scale


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    n: int
    dims: int
    seed: int
    anisotropy: float = 1.0
    rhs_rank: int = 2
    contamination: float = 0.1
    family: str = "symmetric_neumann_path"
    rhs_frequency: int = 1
    # Explicit weights take precedence over the geometric anisotropy profile.
    # They are serialized with each split, so the actual operator is recorded.
    axis_weights: tuple[float, ...] = ()


def _weights(spec: ProblemSpec) -> tuple[float, ...]:
    if spec.n < 2 or spec.dims < 1:
        raise ValueError("Require n >= 2 and dims >= 1.")
    if spec.family != "symmetric_neumann_path":
        raise ValueError("Only the symmetric_neumann_path family is implemented.")
    if not math.isfinite(spec.anisotropy) or spec.anisotropy <= 0:
        raise ValueError("anisotropy must be finite and positive.")
    if spec.rhs_rank < 1 or not isinstance(spec.rhs_rank, int):
        raise ValueError("rhs_rank must be a positive integer.")
    if not 1 <= spec.rhs_frequency < spec.n:
        raise ValueError("rhs_frequency must lie in [1, n-1].")
    if not math.isfinite(spec.contamination) or spec.contamination < 0:
        raise ValueError("contamination must be finite and nonnegative.")
    if spec.axis_weights:
        values = tuple(float(value) for value in spec.axis_weights)
        if len(values) != spec.dims:
            raise ValueError("axis_weights must contain one weight per dimension.")
    else:
        values = tuple(
            spec.anisotropy ** (axis / max(spec.dims - 1, 1))
            for axis in range(spec.dims)
        )
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("All axis weights must be finite and strictly positive.")
    if not math.isfinite(math.fsum(values)):
        raise ValueError("Sum of axis weights must be finite.")
    return values


def operator_group_key(spec: ProblemSpec) -> str:
    """Identify the constructed normalized operator, independently of RHS/name.

    The canonical coefficients intentionally ignore common scaling of all
    weights.  Fourteen significant digits also group roundoff-equivalent
    specifications, instead of using a split label or a random seed as a
    surrogate for operator identity.  Axis order is retained.
    """
    weights = _weights(spec)
    total = math.fsum(weights)
    payload = {
        "family": spec.family,
        "n": spec.n,
        "dims": spec.dims,
        "normalized_axis_weights": [format(value / total, ".14g") for value in weights],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return "neumann_path_" + digest[:24]


@dataclass
class Problem:
    spec: ProblemSpec
    operator: Any
    rhs: Any
    reference_solution: Any
    projector: Callable[[Any], Any]
    q0: Any
    operator_norm: float
    rhs_norm: float
    ab_norm: float
    lambda_min_positive: float
    axis_weights: tuple[float, ...]
    normalized_axis_weights: tuple[float, ...]
    unnormalized_operator_norm: float
    operator_group_key: str
    operator_fingerprint: str

    def online_view(self) -> Mapping[str, Any]:
        """Read-only mapping of online inputs, excluding reference and its seed.

        The mapping is immutable; TT tensor objects retain the backend's
        normal mutable representation and are treated as read-only by callers.
        """
        return MappingProxyType({
            "operator": self.operator,
            "rhs": self.rhs,
            "projector": self.projector,
            "q0": self.q0,
            "operator_norm": self.operator_norm,
            "rhs_norm": self.rhs_norm,
            "ab_norm": self.ab_norm,
            "lambda_min_positive": self.lambda_min_positive,
            "index": 1,
        })

    def construction_metadata(self) -> dict[str, Any]:
        """Serializable description of the matrices actually used by matvec."""
        return {
            "family": self.spec.family,
            "n": self.spec.n,
            "dims": self.spec.dims,
            "index": 1,
            "axis_weights": list(self.axis_weights),
            "normalized_axis_weights": list(self.normalized_axis_weights),
            "unnormalized_operator_norm": self.unnormalized_operator_norm,
            "operator_norm": self.operator_norm,
            "lambda_min_positive": self.lambda_min_positive,
            "operator_group_key": self.operator_group_key,
            "operator_fingerprint": self.operator_fingerprint,
        }


def _manufactured_reference(spec: ProblemSpec) -> Any:
    """Construct a frequency-filtered random low-rank TT in the range of A.

    The path eigenbasis is only an n-by-n local matrix.  Removing its constant
    mode on one axis makes the full tensor orthogonal to the unique nullspace;
    transforming individual TT cores preserves their TT rank bounds.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(spec.seed)
    grid = torch.arange(spec.n, dtype=torch.float64) + 0.5
    modes = torch.arange(spec.n, dtype=torch.float64)
    basis = math.sqrt(2.0 / spec.n) * torch.cos(
        math.pi / spec.n * grid[:, None] * modes[None, :]
    )
    basis[:, 0] = 1.0 / math.sqrt(spec.n)
    width = max(1.0, 0.22 * spec.n)
    spectral_filter = torch.exp(-0.5 * ((modes - spec.rhs_frequency) / width) ** 2)
    # A small broad-spectrum component prevents almost single-eigenmode RHSs.
    spectral_filter = 0.05 + 0.95 * spectral_filter
    ranks = [1] + [spec.rhs_rank] * (spec.dims - 1) + [1]
    zero_mean_axis = spec.seed % spec.dims
    cores = []
    for axis in range(spec.dims):
        spectral_core = torch.randn(
            (ranks[axis], spec.n, ranks[axis + 1]),
            generator=generator, dtype=torch.float64,
        ) * spectral_filter[None, :, None]
        if axis == zero_mean_axis:
            spectral_core[:, 0, :] = 0.0
        physical_core = torch.einsum("ik,akb->aib", basis, spectral_core)
        # Remove DCT cosine evaluation roundoff in the constant mode locally.
        if axis == zero_mean_axis:
            physical_core = physical_core - physical_core.mean(dim=1, keepdim=True)
        cores.append(physical_core)
    reference = tn.Tensor(cores)
    norm = tt_norm_stable(reference)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Manufactured reference unexpectedly has zero/nonfinite norm.")
    return tt_scale(reference, 1.0 / norm)


def build_problem(spec: ProblemSpec) -> Problem:
    """Build a normalized symmetric index-one problem entirely in TT form."""
    weights = _weights(spec)
    total_weight = math.fsum(weights)
    normalized_weights = tuple(value / total_weight for value in weights)
    local_max = 4.0 * (spec.n - 1) ** 2 * math.cos(math.pi / (2.0 * spec.n)) ** 2
    local_min_positive = 4.0 * (spec.n - 1) ** 2 * math.sin(math.pi / (2.0 * spec.n)) ** 2
    normalized_local = neumann_1d_laplacian(spec.n, dtype=torch.float64) / local_max
    local_operators = [weight * normalized_local for weight in normalized_weights]
    operator = variable_kron_sum_tt_operator(
        local_operators, name=spec.name, exact_norm=1.0, dtype=torch.float64,
    )
    q0 = tt_rank1([
        torch.full((spec.n,), 1.0 / math.sqrt(spec.n), dtype=torch.float64)
        for _ in range(spec.dims)
    ])

    def projector(vector: Any) -> Any:
        # Exact TT block subtraction, with no extra truncation or rank cap.
        # This adds at most one to each internal rank; the solver controls its
        # own rounding.  A tiny coefficient is not silently thresholded away.
        coefficient = tt_dot(q0, vector)
        return vector - tt_scale(q0, coefficient) if coefficient != 0.0 else vector

    reference = _manufactured_reference(spec)
    ax = operator.matvec(reference)
    ax_norm = tt_norm_stable(ax)
    rhs = ax + tt_scale(q0, spec.contamination * ax_norm) if spec.contamination else ax
    fingerprint = hashlib.sha256()
    fingerprint.update(f"{spec.n}:{spec.dims}:float64".encode("ascii"))
    for matrix in local_operators:
        fingerprint.update(matrix.contiguous().numpy().tobytes())
    return Problem(
        spec=spec, operator=operator, rhs=rhs, reference_solution=reference,
        projector=projector, q0=q0, operator_norm=1.0,
        rhs_norm=tt_norm_stable(rhs), ab_norm=tt_norm_stable(operator.matvec(rhs)),
        lambda_min_positive=min(normalized_weights) * local_min_positive / local_max,
        axis_weights=weights, normalized_axis_weights=normalized_weights,
        unnormalized_operator_norm=total_weight * local_max,
        operator_group_key=operator_group_key(spec),
        operator_fingerprint=fingerprint.hexdigest(),
    )


def _finite_tt(vector: Any) -> bool:
    return all(bool(torch.isfinite(core).all()) for core in vector.cores) and all(
        factor is None or bool(torch.isfinite(factor).all()) for factor in vector.Us
    )


def audit_solution(problem: Problem, x: Any, include_reference: bool = False) -> dict[str, Any]:
    """Recompute r=b-Ax and Ar without TT rounding; return floating-point checks.

    For ||A||=1 and index a=1 the requested transformed backward measure is
    ``eta = ||Ar|| / (||x|| + ||Ab||)``.  Ordinary residuals stay nonzero at the
    Drazin solution when b has nullspace contamination.  Range membership is
    checked separately because eta cannot detect contamination of x by q0.
    """
    names = (
        "residual_norm", "transformed_residual_norm", "x_norm", "eta",
        "relative_transformed_residual", "relative_true_residual", "range_defect",
    )

    def failed() -> dict[str, Any]:
        result: dict[str, Any] = {name: math.inf for name in names}
        result["finite"] = False
        if include_reference:
            result["forward_error"] = math.inf
        return result

    if not _finite_tt(x):
        return failed()
    residual = problem.rhs - problem.operator.matvec(x)
    transformed = problem.operator.matvec(residual)
    if not _finite_tt(residual) or not _finite_tt(transformed):
        return failed()
    residual_norm = tt_norm_stable(residual)
    transformed_norm = tt_norm_stable(transformed)
    x_norm = tt_norm_stable(x)
    tiny = torch.finfo(torch.float64).tiny
    values: dict[str, Any] = {
        "residual_norm": residual_norm,
        "transformed_residual_norm": transformed_norm,
        "x_norm": x_norm,
        "eta": transformed_norm / max(x_norm + problem.ab_norm, tiny),
        "relative_transformed_residual": transformed_norm / max(problem.ab_norm, tiny),
        "relative_true_residual": residual_norm / max(problem.rhs_norm, tiny),
        "range_defect": abs(tt_dot(problem.q0, x)) / max(x_norm, 1.0),
    }
    if include_reference:
        values["forward_error"] = tt_norm_stable(x - problem.reference_solution)
    values["finite"] = all(math.isfinite(value) for value in values.values())
    return values


def make_splits(
    seed: int = 20260929,
    train_count: int = 48,
    validation_count: int = 12,
    test_count: int = 12,
    ood_count: int = 6,
) -> dict[str, list[ProblemSpec]]:
    """Create deterministic, disjoint operator groups with recorded weights.

    In-distribution splits share the same size/anisotropy/rank distributions
    but never an actual normalized operator.  OOD cases increase size,
    dimension, anisotropy, and requested TT rank.  Counts may be reduced for
    smoke runs; any split with at least two cases includes contamination 0
    and 0.1.  The RHS seed does not participate in operator identity.
    """
    counts = {"train": train_count, "validation": validation_count, "test": test_count, "ood": ood_count}
    if any(not isinstance(count, int) or count < 0 for count in counts.values()):
        raise ValueError("Split counts must be nonnegative integers.")
    result: dict[str, list[ProblemSpec]] = {}
    used_operators: set[str] = set()
    for split_index, (split, count) in enumerate(counts.items()):
        # Separate streams preserve a split's entries when another count changes.
        rng = random.Random(seed + 104729 * (split_index + 1))
        specs = []
        for case in range(count):
            ood = split == "ood"
            n = rng.choice((10, 12, 16) if ood else (4, 5, 6, 8))
            dims = rng.choice((4, 5, 6) if ood else (2, 3, 4))
            anisotropy = rng.choice((8.0, 12.0, 20.0) if ood else (1.0, 2.0, 4.0))
            rank = rng.choice((4, 5) if ood else (1, 2, 3))
            rhs_seed = rng.randrange(1, 2**31 - 1)
            frequency = rng.choice((1, max(1, n // 2), n - 1))
            while True:
                # Continuous positive jitter ensures that splitting examples
                # cannot merely relabel the same A with a different RHS seed.
                weights = tuple(
                    anisotropy ** (axis / (dims - 1)) * math.exp(rng.uniform(-0.15, 0.15))
                    for axis in range(dims)
                )
                spec = ProblemSpec(
                    name=f"{split}_{case:03d}_n{n}_d{dims}_r{rank}",
                    n=n, dims=dims, seed=rhs_seed, anisotropy=anisotropy,
                    rhs_rank=rank, contamination=0.0 if case % 2 == 0 else 0.1,
                    rhs_frequency=frequency, axis_weights=weights,
                )
                group = operator_group_key(spec)
                if group not in used_operators:
                    used_operators.add(group)
                    specs.append(spec)
                    break
        result[split] = specs
    return result
