"""The latest paper's isotropic Neumann--Poisson CP-manufactured problems.

This module deliberately does not use V1/V2's random-TT manufactured RHS.
The NumPy random stream, CP/axis loop order, global mean removal, and three
data-preparation roundings follow the paper's ``make_problem_nd`` builder.
Only local n-by-n matrices and TT cores are materialized. V1's reference-free
online interface and independent, untruncated floating-point audit are reused.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
import torch

from adaptive_tt_dgmres.backend.paper_tt_dgmres import (
    tt_norm_stable, tt_round_capped, variable_kron_sum_tt_operator,
)
from adaptive_tt_dgmres.backend.paper_tt_gmres import tt_dot, tt_rank1, tt_scale
from rl_tt_dgmres_v1.problems import Problem as _BaseProblem, audit_solution


BUILDER_SOURCE = {
    "path": "C:/Users/16081/Desktop/论文知识库/pytorch_version/backend_cells.json",
    "cell_id": "33430136",
    "function": "make_problem_nd",
    "sha256": "8df85f8bc82f8f123b7356b5d3e18187769f6b7eac4213bca850d5f84e397133",
}
PAPER_SOURCE = {
    "path": "C:/Users/16081/Desktop/论文知识库/output/上财模板_Nature方法修订_20260929/chapters/ch05_setup.tex",
    "equations": ["eq:exp-L", "eq:exp-A", "eq:exp-reference", "eq:exp-rhs"],
}
FAMILY = "paper_neumann_poisson_cp"


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    n: int
    dims: int
    seed: int
    cp_rank: int = 3
    max_mode: int | None = None
    null_ratio: float = 0.1
    spectral_decay: float = 2.0
    preparation_tolerance: float = 1e-13
    family: str = FAMILY


def _validate(spec: ProblemSpec) -> int:
    if not isinstance(spec.name, str) or not spec.name:
        raise ValueError("name must be a nonempty string")
    for field, minimum in (("n", 3), ("dims", 2), ("seed", 0), ("cp_rank", 1)):
        value = getattr(spec, field)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{field} must be an integer >= {minimum}")
    if spec.family != FAMILY:
        raise ValueError(f"Only {FAMILY} is supported")
    max_mode = spec.n - 1 if spec.max_mode is None else spec.max_mode
    if type(max_mode) is not int or not 1 <= max_mode < spec.n:
        raise ValueError("max_mode must be None or an integer in [1, n-1]")
    for field in ("null_ratio", "spectral_decay", "preparation_tolerance"):
        value = getattr(spec, field)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{field} must be finite and nonnegative")
    if spec.preparation_tolerance >= 1:
        raise ValueError("preparation_tolerance must be less than one")
    return max_mode


def spec_from_dict(values: Mapping[str, Any]) -> ProblemSpec:
    payload = dict(values)
    unknown = payload.keys() - {field.name for field in fields(ProblemSpec)}
    if unknown:
        raise ValueError(f"Unknown problem specification fields: {sorted(unknown)}")
    spec = ProblemSpec(**payload)
    _validate(spec)
    return spec


def _construction_parameters(spec: ProblemSpec) -> dict[str, Any]:
    max_mode = _validate(spec)
    return {
        "n": spec.n, "dims": spec.dims, "seed": spec.seed,
        "cp_rank": spec.cp_rank, "max_mode": max_mode,
        "null_ratio": float(spec.null_ratio), "spectral_decay": float(spec.spectral_decay),
        "preparation_tolerance": float(spec.preparation_tolerance),
    }


def instance_key(spec: ProblemSpec) -> str:
    """Identify numerical construction, independently of the display name.

    The effective max_mode canonicalizes None and n-1. Labels such as family
    do not create spurious distinct instances of the same numerical recipe.
    """
    payload = _construction_parameters(spec)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "paper_cp_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _local_data(spec: ProblemSpec) -> tuple[np.ndarray, float, float]:
    _validate(spec)
    h = 1.0 / (spec.n - 1)
    local = np.diag(np.r_[1.0, np.full(spec.n - 2, 2.0), 1.0])
    local += np.diag(-np.ones(spec.n - 1), 1) + np.diag(-np.ones(spec.n - 1), -1)
    local *= h ** (-2)
    local_max = 2.0 * h ** (-2) * (1.0 + np.cos(np.pi / spec.n))
    raw_norm = float(spec.dims * local_max)
    minimum = float(2.0 * h ** (-2) * (1.0 - np.cos(np.pi / spec.n)) / raw_norm)
    return local / raw_norm, raw_norm, minimum


def _operator_group_from_local(spec: ProblemSpec, local: np.ndarray) -> str:
    # The payload and precision match V2's matrix-based graph identity. There
    # is no family/seed tag: a V2 equal-weight path is the same operator group.
    matrix = [format(float(value), ".13g") if value != 0 else "0" for value in local.reshape(-1)]
    payload = {"n": spec.n, "dims": spec.dims,
               "normalized_local_matrices": [matrix for _ in range(spec.dims)]}
    encoded = json.dumps(payload, sort_keys=True)
    return "graph_laplacian_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def operator_group_key(spec: ProblemSpec) -> str:
    """Identify A from n, d and the normalized local matrices, never the RHS."""
    local, _, _ = _local_data(spec)
    return _operator_group_from_local(spec, local)


@dataclass
class Problem(_BaseProblem):
    instance_key: str = ""

    def construction_metadata(self) -> dict[str, Any]:
        result = super().construction_metadata()
        result.update({
            "instance_key": self.instance_key,
            "builder_source": dict(BUILDER_SOURCE),
            "paper_source": {**PAPER_SOURCE, "equations": list(PAPER_SOURCE["equations"])},
            "construction_parameters": _construction_parameters(self.spec),
            "preparation": {
                "phases": ["reference", "compatible_rhs", "rhs"],
                "relative_tolerance": float(self.spec.preparation_tolerance),
                "rank_cap": None,
                "rounder": "adaptive_tt_dgmres.backend.paper_tt_dgmres.tt_round_capped(max_rank=None)",
                "ab_is_unrounded": True,
            },
            "random_stream": "numpy.default_rng; CP term j then axis then q=0..max_mode",
            "reference_construction": "normalized globally zero-mean weighted CP sum of normalized random cosine factors",
            "cp_rank_is_not_rhs_tt_rank": True,
            "normalization": "sum(I tensor L tensor I) / (d * lambda_max(L))",
            "reference_in_online_view": False,
            "roundoff_scope": "same mathematical builder and preparation budgets; no claim of bitwise TT-core agreement across backends",
        })
        return result


def build_problem(spec: ProblemSpec) -> Problem:
    """Build the paper problem in float64 CPU TT form, without rank clipping."""
    max_mode = _validate(spec)
    local, raw_norm, minimum = _local_data(spec)
    local_tensor = torch.from_numpy(local.copy())
    operator = variable_kron_sum_tt_operator(
        [local_tensor for _ in range(spec.dims)], name=spec.name,
        exact_norm=1.0, dtype=torch.float64, device="cpu",
    )
    qvec = np.ones(spec.n, dtype=float) / np.sqrt(spec.n)
    q0 = tt_rank1([torch.from_numpy(qvec.copy()) for _ in range(spec.dims)])

    def projector(vector: Any) -> Any:
        coefficient = tt_dot(q0, vector)
        return vector - tt_scale(q0, coefficient) if coefficient != 0.0 else vector

    # Match the original generator's draw order, not merely its marginal law.
    rng = np.random.default_rng(int(spec.seed))
    grid = np.arange(spec.n, dtype=float)
    phi = np.empty((spec.n, max_mode + 1), dtype=float)
    phi[:, 0] = 1.0 / np.sqrt(spec.n)
    for q in range(1, max_mode + 1):
        phi[:, q] = np.sqrt(2.0 / spec.n) * np.cos(np.pi * q * (grid + 0.5) / spec.n)
    decay = (1.0 + np.arange(phi.shape[1])) ** float(spec.spectral_decay)
    reference = None
    for j in range(spec.cp_rank):
        factors = []
        for _axis in range(spec.dims):
            coefficients = rng.standard_normal(phi.shape[1]) / decay
            factor = phi @ coefficients
            norm = float(np.linalg.norm(factor))
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("Cosine factor has zero/nonfinite norm")
            factor /= norm
            factors.append(torch.from_numpy(factor.copy()))
        term = tt_scale(tt_rank1(factors), 0.5 ** j)
        reference = term if reference is None else reference + term
    reference = projector(reference)
    reference = tt_round_capped(reference, float(spec.preparation_tolerance), max_rank=None)
    reference_norm = tt_norm_stable(reference)
    if not math.isfinite(reference_norm) or reference_norm <= 0:
        raise ValueError("Manufactured reference has zero/nonfinite norm")
    reference = tt_scale(reference, 1.0 / reference_norm)
    compatible_rhs = tt_round_capped(
        operator.matvec(reference), float(spec.preparation_tolerance), max_rank=None,
    )
    compatible_norm = tt_norm_stable(compatible_rhs)
    rhs = compatible_rhs + tt_scale(q0, float(spec.null_ratio) * compatible_norm)
    rhs = tt_round_capped(rhs, float(spec.preparation_tolerance), max_rank=None)
    rhs_norm = tt_norm_stable(rhs)
    ab_norm = tt_norm_stable(operator.matvec(rhs))  # The paper does not round Ab.
    if not all(math.isfinite(value) and value > 0 for value in (compatible_norm, rhs_norm, ab_norm)):
        raise ValueError("Manufactured RHS or transformed RHS has zero/nonfinite norm")

    fingerprint = hashlib.sha256(f"{spec.n}:{spec.dims}:float64".encode("ascii"))
    for _ in range(spec.dims):
        fingerprint.update(local_tensor.contiguous().numpy().tobytes())
    return Problem(
        spec=spec, operator=operator, rhs=rhs, reference_solution=reference,
        projector=projector, q0=q0, operator_norm=1.0, rhs_norm=rhs_norm, ab_norm=ab_norm,
        lambda_min_positive=minimum, axis_weights=(1.0,) * spec.dims,
        normalized_axis_weights=(1.0 / spec.dims,) * spec.dims,
        unnormalized_operator_norm=raw_norm,
        operator_group_key=_operator_group_from_local(spec, local),
        operator_fingerprint=fingerprint.hexdigest(), instance_key=instance_key(spec),
    )


__all__ = ["BUILDER_SOURCE", "PAPER_SOURCE", "FAMILY", "ProblemSpec", "Problem",
           "spec_from_dict", "build_problem", "operator_group_key", "instance_key", "audit_solution"]
