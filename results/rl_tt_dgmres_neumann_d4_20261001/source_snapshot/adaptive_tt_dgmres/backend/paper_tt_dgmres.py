from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Callable, Dict, List, Sequence

import numpy as np
import torch

from .paper_tt_gmres import (
    TTTensor,
    TTLinearOperator,
    _resolve_operator_norm,
    tt_dot,
    tt_dot_tensor,
    tt_device,
    tt_dtype,
    tt_linear_combination,
    tt_matrix_operator,
    tt_max_rank,
    tt_norm,
    tt_rank1,
    tt_round,
    tt_round_capped as _tt_round_capped_device_safe,
    tt_scalar,
    tt_scale,
    tt_storage_ratio,
    tt_zeros,
)


torch.set_default_dtype(torch.float64)


def tt_round_capped(tensor: TTTensor, eps: float, max_rank: int | None = None) -> TTTensor:
    """Round with the core helper's CUDA-safe exact-zero fallback."""
    return _tt_round_capped_device_safe(tensor, eps, max_rank)


def tt_linear_combination_capped_tree(
    vectors: Sequence[TTTensor],
    coeffs: Sequence[float],
    round_eps: float,
    max_rank: int | None = None,
) -> TTTensor:
    """Form a linear combination with a rank-capped pairwise reduction tree.

    This can reduce peak intermediate TT ranks compared with constructing one
    block sum and rounding only once.  Because it introduces additional
    truncations, callers must select it explicitly rather than inheriting it
    from an unrelated orthogonalization option.
    """

    if not vectors:
        raise ValueError("Expected at least one TT vector.")
    if len(vectors) != len(coeffs):
        raise ValueError("vectors and coeffs must have the same length.")
    active = [
        tt_round_capped(tt_scale(vector, coeff), round_eps, max_rank)
        for coeff, vector in zip(coeffs, vectors)
        if tt_scalar(coeff) != 0.0
    ]
    if not active:
        return tt_zeros(vectors[0].shape, like=vectors[0])
    while len(active) > 1:
        next_level: List[TTTensor] = []
        for offset in range(0, len(active), 2):
            if offset + 1 == len(active):
                next_level.append(active[offset])
            else:
                next_level.append(
                    tt_round_capped(
                        active[offset] + active[offset + 1],
                        round_eps,
                        max_rank,
                    )
                )
        active = next_level
    return active[0]


def tt_norm_stable_tensor(tensor: TTTensor) -> torch.Tensor:
    """Evaluate a TT norm after a lossless QR gauge transformation.

    Direct environment contraction of block-sum cores can suffer catastrophic
    cancellation and even produce a tiny negative norm-square that is clamped
    to zero by the backend.  Orthogonalization changes only the TT gauge, not
    the represented tensor; at the last orthogonality centre the Frobenius norm
    is the Euclidean norm of that core.
    """

    orthogonal = tensor.clone()
    orthogonal.orthogonalize(orthogonal.dim() - 1)
    return torch.linalg.vector_norm(orthogonal.cores[-1])


def tt_norm_stable(tensor: TTTensor) -> float:
    """Backward-compatible host scalar wrapper around the stable TT norm."""
    return tt_scalar(tt_norm_stable_tensor(tensor))


# Backward-compatible private spelling used by the first audited solver patch.
_stable_tt_norm = tt_norm_stable


def tt_apply_power(
    operator: TTLinearOperator,
    vector: TTTensor,
    power: int,
    round_eps: float,
    max_rank: int | None = None,
) -> TTTensor:
    result = vector
    for _ in range(power):
        result = tt_round_capped(operator.matvec(result), round_eps, max_rank)
    return result


def hbar_block(hbar: np.ndarray, ell: int) -> np.ndarray:
    """Return the Arnoldi block Hbar_ell in the paper notation."""
    return hbar[: ell + 1, :ell]


def build_hhat(hbar: np.ndarray, k: int, a: int) -> np.ndarray:
    """Build Hhat_k = Hbar_k Hbar_(k-1) ... Hbar_(k-a)."""
    if k <= a:
        raise ValueError("Need k > a so that Hhat_k is defined.")

    hbar_factors = [hbar_block(hbar, ell) for ell in range(k, k - a - 1, -1)]
    hhat = hbar_factors[0].copy()
    for hbar_ell in hbar_factors[1:]:
        hhat = hhat @ hbar_ell
    return hhat


@dataclass
class TTDGMRESConfig:
    index: int = 1
    restart: int = 20
    max_iterations: int = 100
    tolerance: float = 1e-8
    round_tolerance: float = 1e-10
    orth_round_tolerance: float | None = None
    residual_round_tolerance: float | None = None
    solution_round_tolerance: float | None = None
    transformed_round_tolerance: float | None = None
    operator_norm: float | None = None
    operator_norm_mode: str = "sampled"
    operator_norm_samples: int = 10
    operator_norm_tt_rank: int = 2
    operator_norm_seed: int = 0
    max_cycles: int | None = None
    safe_orth_rounding: bool = False
    tree_solution_combination: bool = False
    reorthogonalize: bool = True
    stopping_metric: str = "a_residual"
    max_tt_rank: int | None = None
    breakdown_tolerance: float = 1e-14

    def resolved(self) -> "TTDGMRESConfig":
        return TTDGMRESConfig(
            index=self.index,
            restart=self.restart,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            round_tolerance=self.round_tolerance,
            orth_round_tolerance=self.round_tolerance if self.orth_round_tolerance is None else self.orth_round_tolerance,
            residual_round_tolerance=self.round_tolerance if self.residual_round_tolerance is None else self.residual_round_tolerance,
            solution_round_tolerance=self.round_tolerance if self.solution_round_tolerance is None else self.solution_round_tolerance,
            transformed_round_tolerance=(
                self.round_tolerance if self.transformed_round_tolerance is None else self.transformed_round_tolerance
            ),
            operator_norm=self.operator_norm,
            operator_norm_mode=self.operator_norm_mode,
            operator_norm_samples=self.operator_norm_samples,
            operator_norm_tt_rank=self.operator_norm_tt_rank,
            operator_norm_seed=self.operator_norm_seed,
            max_cycles=self.max_cycles,
            safe_orth_rounding=self.safe_orth_rounding,
            tree_solution_combination=self.tree_solution_combination,
            reorthogonalize=self.reorthogonalize,
            stopping_metric=self.stopping_metric,
            max_tt_rank=self.max_tt_rank,
            breakdown_tolerance=self.breakdown_tolerance,
        )


@dataclass
class TTDGMRESResult:
    solution: TTTensor
    converged: bool
    iterations: int
    cycles: int
    operator_norm: float
    history: Dict[str, List[float]]
    config: TTDGMRESConfig
    status: str = "iteration_limit"
    breakdown: bool = False

    @property
    def final_transformed_residual_norm(self) -> float:
        values = self.history["transformed_residual_norm"]
        return values[-1] if values else math.nan


def _empty_history() -> Dict[str, List[float]]:
    return {
        "iteration": [],
        "cycle": [],
        "backward_error": [],
        "true_residual_norm": [],
        "transformed_residual_norm": [],
        "relative_transformed_residual_norm": [],
        "projected_transformed_residual_norm": [],
        "relative_solution_error": [],
        "solution_rank": [],
        "last_vector_rank": [],
        "last_vector_ratio": [],
        "basis_ratio": [],
        "solution_numcoef": [],
        "last_vector_numcoef": [],
        "basis_numcoef": [],
        "basis_vector_count": [],
        "basis_storage_bytes": [],
        "dense_basis_storage_bytes": [],
    }


def _append_history(
    history: Dict[str, List[float]],
    iteration: int,
    cycle: int,
    backward_eta: float,
    true_residual_norm: float,
    transformed_residual_norm: float,
    relative_transformed_residual_norm: float,
    projected_transformed_residual_norm: float,
    relative_solution_error: float,
    solution: TTTensor,
    current_basis: Sequence[TTTensor],
) -> None:
    last_vector = current_basis[-1]
    dense_size = float(last_vector.numel())
    element_bytes = int(last_vector.cores[0].element_size())
    solution_numcoef = int(solution.numcoef())
    last_vector_numcoef = int(last_vector.numcoef())
    basis_numcoef = sum(int(vector.numcoef()) for vector in current_basis)
    basis_vector_count = len(current_basis)

    history["iteration"].append(float(iteration))
    history["cycle"].append(float(cycle))
    history["backward_error"].append(float(backward_eta))
    history["true_residual_norm"].append(float(true_residual_norm))
    history["transformed_residual_norm"].append(float(transformed_residual_norm))
    history["relative_transformed_residual_norm"].append(float(relative_transformed_residual_norm))
    history["projected_transformed_residual_norm"].append(float(projected_transformed_residual_norm))
    history["relative_solution_error"].append(float(relative_solution_error))
    history["solution_rank"].append(float(tt_max_rank(solution)))
    history["last_vector_rank"].append(float(tt_max_rank(last_vector)))
    history["last_vector_ratio"].append(float(tt_storage_ratio(last_vector)))
    history["basis_ratio"].append(basis_numcoef / (basis_vector_count * dense_size))
    history["solution_numcoef"].append(float(solution_numcoef))
    history["last_vector_numcoef"].append(float(last_vector_numcoef))
    history["basis_numcoef"].append(float(basis_numcoef))
    history["basis_vector_count"].append(float(basis_vector_count))
    history["basis_storage_bytes"].append(float(basis_numcoef * element_bytes))
    history["dense_basis_storage_bytes"].append(
        float(basis_vector_count * int(dense_size) * element_bytes)
    )


def _relative_solution_error(
    solution: TTTensor,
    reference_solution: TTTensor | None,
    round_eps: float,
    reference_norm: float | None = None,
) -> float:
    if reference_solution is None:
        return math.nan
    denom = (
        _stable_tt_norm(reference_solution)
        if reference_norm is None
        else float(reference_norm)
    )
    if denom == 0.0:
        return math.nan
    # Do not round away the very error this diagnostic is meant to measure.
    # ``round_eps`` remains in the signature for API compatibility.
    del round_eps
    diff = solution - reference_solution
    return _stable_tt_norm(diff) / denom


def _compute_residual_metrics(
    operator: TTLinearOperator,
    rhs: TTTensor,
    x: TTTensor,
    index: int,
    residual_round_eps: float,
    transformed_round_eps: float,
    operator_norm: float,
    max_rank: int | None = None,
) -> tuple[float, float, float]:
    # The ordinary residual is cheap enough to contract before TT rounding.
    # Reporting the rounded norm here used to make both the backward error and
    # the so-called ``true_residual_norm`` optimistic when a rank cap was active.
    residual_unrounded = rhs - operator.matvec(x)
    residual_norm = _stable_tt_norm(residual_unrounded)
    denom = operator_norm * _stable_tt_norm(x) + _stable_tt_norm(rhs)
    backward_eta = residual_norm if denom == 0.0 else residual_norm / denom
    residual = tt_round_capped(residual_unrounded, residual_round_eps, max_rank)
    transformed_residual = tt_apply_power(operator, residual, index, transformed_round_eps, max_rank)
    transformed_residual_norm = tt_norm_stable(transformed_residual)
    return backward_eta, residual_norm, transformed_residual_norm


_TRANSFORM_AUDIT_MAX_BYTES = 512 * 1024 * 1024


def _tt_core_storage_bytes(tensor: TTTensor) -> int:
    return sum(int(core.numel()) * int(core.element_size()) for core in tensor.cores)


def _certified_transformed_residual_norm(
    operator: TTLinearOperator,
    residual_unrounded: TTTensor,
    index: int,
    diagnostic_rank_hint: int | None,
) -> tuple[float, bool, float, float]:
    """Contract ``||A^index r||`` without TT truncation when it is safe.

    The previous stopping test measured a separately rounded residual and could
    therefore certify convergence after discarding the component that mattered.
    Here the audit chain starts from the *unrounded* residual and never calls a
    TT rounding routine.  A conservative storage/rank guard prevents an exact
    audit from turning into an uncontrolled high-index rank explosion.  If the
    guard fires, the value is deliberately unavailable and callers must not
    declare convergence from a rounded surrogate.
    """

    if index < 0:
        raise ValueError("index must be nonnegative.")

    transformed = residual_unrounded
    if index == 0:
        rank = float(tt_max_rank(transformed))
        return _stable_tt_norm(transformed), True, rank, rank

    rank_limit = max(384, 2 * int(diagnostic_rank_hint or 0))
    last_pre_rank = float(tt_max_rank(transformed))
    last_rank = last_pre_rank
    if (
        last_rank > rank_limit
        or _tt_core_storage_bytes(transformed) > _TRANSFORM_AUDIT_MAX_BYTES
    ):
        return math.nan, False, last_pre_rank, last_rank

    for _ in range(index):
        # Do not apply A to an intermediate that has already crossed the audit
        # budget.  For an arbitrary callback operator its output rank cannot be
        # predicted safely, so the result is checked immediately after matvec as
        # well and is never contracted when it is outside the budget.
        last_pre_rank = float(tt_max_rank(transformed))
        if (
            last_pre_rank > rank_limit
            or _tt_core_storage_bytes(transformed) > _TRANSFORM_AUDIT_MAX_BYTES
        ):
            return math.nan, False, last_pre_rank, last_pre_rank
        transformed = operator.matvec(transformed)
        last_rank = float(tt_max_rank(transformed))
        if (
            last_rank > rank_limit
            or _tt_core_storage_bytes(transformed) > _TRANSFORM_AUDIT_MAX_BYTES
        ):
            return math.nan, False, last_pre_rank, last_rank

    return _stable_tt_norm(transformed), True, last_pre_rank, last_rank


def _audited_residual_metrics(
    *,
    operator: TTLinearOperator,
    rhs: TTTensor,
    x: TTTensor,
    index: int,
    residual_round_eps: float,
    transformed_round_eps: float,
    operator_norm: float,
    max_rank: int | None,
    rhs_norm: float | None = None,
    certify_transformed: bool = True,
) -> Dict[str, float]:
    """Return exact ordinary-residual metrics plus a certified A-residual.

    ``transformed_residual_norm`` is exact when the untruncated audit fits the
    safety budget.  Otherwise it remains a rounded diagnostic for plotting and
    ``transformed_residual_certified`` is zero, which forbids its use as a stop
    certificate.
    """

    residual_unrounded = rhs - operator.matvec(x)
    residual_norm = _stable_tt_norm(residual_unrounded)
    resolved_rhs_norm = (
        _stable_tt_norm(rhs) if rhs_norm is None else float(rhs_norm)
    )
    denominator = operator_norm * _stable_tt_norm(x) + resolved_rhs_norm
    backward_eta = (
        residual_norm if denominator == 0.0 else residual_norm / denominator
    )

    if certify_transformed:
        certified_norm, certified, audit_pre_rank, audit_rank = (
            _certified_transformed_residual_norm(
                operator,
                residual_unrounded,
                index,
                max_rank,
            )
        )
    else:
        certified_norm = math.nan
        certified = False
        audit_pre_rank = float(tt_max_rank(residual_unrounded))
        audit_rank = audit_pre_rank
    residual_rounded = tt_round_capped(
        residual_unrounded,
        residual_round_eps,
        max_rank,
    )
    transformed_rounded = tt_apply_power(
        operator,
        residual_rounded,
        index,
        transformed_round_eps,
        max_rank,
    )
    rounded_norm = tt_norm_stable(transformed_rounded)
    reported_transformed_norm = certified_norm if certified else rounded_norm
    return {
        "backward_error": float(backward_eta),
        "residual_norm": residual_norm,
        "transformed_residual_norm": float(reported_transformed_norm),
        "rounded_transformed_residual_norm": rounded_norm,
        "transformed_residual_certified": float(certified),
        "audit_transformed_pre_rank": float(audit_pre_rank),
        "audit_transformed_rank": float(audit_rank),
    }


def _arnoldi_invariance_is_certified(
    operator: TTLinearOperator,
    basis: Sequence[TTTensor],
    hbar: np.ndarray,
    k: int,
) -> bool:
    """Numerically certify ``A V_k = V_k H_k`` before using ``H_k`` powers.

    A small final Arnoldi norm alone is not such a certificate, especially after
    TT rounding.  This independent, unrounded check is only needed for the
    exceptional ``k <= index`` lucky-breakdown update.
    """

    if k <= 0 or len(basis) < k:
        return False
    tolerance = 256.0 * np.finfo(float).eps * max(1, k)
    for column in range(k):
        image = operator.matvec(basis[column])
        approximation = tt_linear_combination(
            basis[:k],
            hbar[:k, column],
            0.0,
            max_rank=None,
        )
        defect_norm = _stable_tt_norm(image - approximation)
        scale = max(_stable_tt_norm(image), np.finfo(float).tiny)
        if not np.isfinite(defect_norm) or defect_norm > tolerance * scale:
            return False
    return True


def tt_dgmres_paper(
    operator: TTLinearOperator,
    rhs: TTTensor,
    config: TTDGMRESConfig | None = None,
    x0: TTTensor | None = None,
    reference_solution: TTTensor | None = None,
    solution_projector: Callable[[TTTensor], TTTensor] | None = None,
) -> TTDGMRESResult:
    cfg = (config or TTDGMRESConfig()).resolved()
    if cfg.index < 0:
        raise ValueError("index must be nonnegative.")
    if cfg.restart <= cfg.index:
        raise ValueError("restart must be strictly larger than index.")

    rhs_device = tt_device(rhs)
    rhs_dtype = tt_dtype(rhs)

    def require_problem_layout(label: str, tensor: TTTensor) -> None:
        if tt_device(tensor) != rhs_device or tt_dtype(tensor) != rhs_dtype:
            raise ValueError(
                f"{label} must use device={rhs_device} and dtype={rhs_dtype}; "
                f"got device={tt_device(tensor)} and dtype={tt_dtype(tensor)}."
            )

    if x0 is not None:
        require_problem_layout("x0", x0)
    if reference_solution is not None:
        require_problem_layout("reference_solution", reference_solution)

    operator_norm = _resolve_operator_norm(operator, cfg)  # type: ignore[arg-type]
    x = (
        tt_zeros(rhs.shape, like=rhs)
        if x0 is None
        else tt_round_capped(
            x0,
            cfg.solution_round_tolerance,
            cfg.max_tt_rank,
        )
    )
    if solution_projector is not None:
        x = solution_projector(x)
        require_problem_layout("solution_projector output", x)
    rhs_norm = _stable_tt_norm(rhs)
    reference_norm = (
        _stable_tt_norm(reference_solution)
        if reference_solution is not None
        else None
    )
    history = _empty_history()
    history["diagnostic_metric_valid_for_stopping"] = []
    history["diagnostic_transformed_residual_certified"] = []
    history["diagnostic_rounded_transformed_residual_norm"] = []
    total_iterations = 0
    cycles = 0
    converged = False
    breakdown_terminated = False
    breakdown = False
    status = "iteration_limit"

    initial_metrics = _audited_residual_metrics(
        operator=operator,
        rhs=rhs,
        x=x,
        index=cfg.index,
        residual_round_eps=cfg.residual_round_tolerance,
        transformed_round_eps=cfg.transformed_round_tolerance,
        operator_norm=operator_norm,
        max_rank=cfg.max_tt_rank,
        rhs_norm=rhs_norm,
        certify_transformed=cfg.stopping_metric in {"a_residual", "relative_a_residual"},
    )
    initial_transformed_residual_norm = float(
        initial_metrics["transformed_residual_norm"]
    )

    def metric_is_certified(metrics: Dict[str, float]) -> bool:
        if cfg.stopping_metric == "backward_error":
            # The ordinary residual is contracted before any TT truncation.
            return True
        if cfg.stopping_metric == "a_residual":
            return bool(metrics["transformed_residual_certified"])
        if cfg.stopping_metric == "relative_a_residual":
            return bool(
                initial_metrics["transformed_residual_certified"]
                and metrics["transformed_residual_certified"]
            )
        raise ValueError(f"Unsupported stopping_metric: {cfg.stopping_metric}")

    def diagnostic_stop_value(metrics: Dict[str, float]) -> float:
        if cfg.stopping_metric == "a_residual":
            return float(metrics["transformed_residual_norm"])
        if cfg.stopping_metric == "relative_a_residual":
            if initial_transformed_residual_norm == 0.0:
                return 0.0 if metrics["transformed_residual_norm"] == 0.0 else math.inf
            return (
                float(metrics["transformed_residual_norm"])
                / initial_transformed_residual_norm
            )
        if cfg.stopping_metric == "backward_error":
            return float(metrics["backward_error"])
        raise ValueError(f"Unsupported stopping_metric: {cfg.stopping_metric}")

    def append_diagnostic(
        *,
        metrics: Dict[str, float],
        x_eval: TTTensor,
        current_basis: Sequence[TTTensor],
        iteration: int,
        cycle: int,
        projected_residual: float,
    ) -> None:
        transformed_norm = float(metrics["transformed_residual_norm"])
        relative_transformed_norm = (
            0.0
            if initial_transformed_residual_norm == 0.0
            and transformed_norm == 0.0
            else (
                math.inf
                if initial_transformed_residual_norm == 0.0
                else transformed_norm / initial_transformed_residual_norm
            )
        )
        _append_history(
            history=history,
            iteration=iteration,
            cycle=cycle,
            backward_eta=float(metrics["backward_error"]),
            true_residual_norm=float(metrics["residual_norm"]),
            transformed_residual_norm=transformed_norm,
            relative_transformed_residual_norm=relative_transformed_norm,
            projected_transformed_residual_norm=projected_residual,
            relative_solution_error=_relative_solution_error(
                x_eval,
                reference_solution,
                cfg.solution_round_tolerance,
                reference_norm=reference_norm,
            ),
            solution=x_eval,
            current_basis=current_basis,
        )
        history["diagnostic_metric_valid_for_stopping"].append(
            float(metric_is_certified(metrics))
        )
        history["diagnostic_transformed_residual_certified"].append(
            float(metrics["transformed_residual_certified"])
        )
        history["diagnostic_rounded_transformed_residual_norm"].append(
            float(metrics["rounded_transformed_residual_norm"])
        )

    # Test the supplied initial guess before constructing an Arnoldi basis.  In
    # particular, a small but nonzero residual must not force one extra cycle.
    initial_algorithm_residual = tt_round_capped(
        rhs - operator.matvec(x),
        cfg.residual_round_tolerance,
        cfg.max_tt_rank,
    )
    initial_algorithm_transformed = tt_apply_power(
        operator,
        initial_algorithm_residual,
        cfg.index,
        cfg.transformed_round_tolerance,
        cfg.max_tt_rank,
    )
    initial_projected_residual = tt_norm_stable(initial_algorithm_transformed)
    if (
        diagnostic_stop_value(initial_metrics) < cfg.tolerance
        and metric_is_certified(initial_metrics)
    ):
        append_diagnostic(
            metrics=initial_metrics,
            x_eval=x,
            current_basis=[initial_algorithm_transformed],
            iteration=0,
            cycle=0,
            projected_residual=initial_projected_residual,
        )
        converged = True
        status = "initial_residual_target"
        return TTDGMRESResult(
            solution=x,
            converged=converged,
            iterations=0,
            cycles=0,
            operator_norm=operator_norm,
            history=history,
            config=cfg,
            status=status,
            breakdown=False,
        )

    max_cycles = cfg.max_cycles
    if max_cycles is None:
        max_cycles = max(1, math.ceil(cfg.max_iterations / cfg.restart))

    while (
        total_iterations < cfg.max_iterations
        and cycles < max_cycles
        and not converged
        and not breakdown_terminated
    ):
        cycles += 1
        cycle_start_x = x
        residual0 = tt_round_capped(rhs - operator.matvec(cycle_start_x), cfg.residual_round_tolerance, cfg.max_tt_rank)
        transformed_residual0 = tt_apply_power(
            operator,
            residual0,
            cfg.index,
            cfg.transformed_round_tolerance,
            cfg.max_tt_rank,
        )
        beta_tensor = tt_norm_stable_tensor(transformed_residual0)
        beta = tt_scalar(beta_tensor)
        if beta == 0.0:
            restart_metrics = _audited_residual_metrics(
                operator=operator,
                rhs=rhs,
                x=cycle_start_x,
                index=cfg.index,
                residual_round_eps=cfg.residual_round_tolerance,
                transformed_round_eps=cfg.transformed_round_tolerance,
                operator_norm=operator_norm,
                max_rank=cfg.max_tt_rank,
                rhs_norm=rhs_norm,
                certify_transformed=cfg.stopping_metric
                in {"a_residual", "relative_a_residual"},
            )
            append_diagnostic(
                metrics=restart_metrics,
                x_eval=cycle_start_x,
                current_basis=[transformed_residual0],
                iteration=total_iterations,
                cycle=cycles,
                projected_residual=0.0,
            )
            converged = bool(
                diagnostic_stop_value(restart_metrics) < cfg.tolerance
                and metric_is_certified(restart_metrics)
            )
            breakdown_terminated = not converged
            if converged:
                status = "residual_target_at_cycle_start"
            else:
                status = "zero_transformed_residual_without_target"
                breakdown = True
            break

        basis: List[TTTensor] = [
            tt_scale(transformed_residual0, torch.reciprocal(beta_tensor))
        ]
        hbar = torch.zeros(
            (cfg.restart + 1, cfg.restart),
            device=rhs_device,
            dtype=rhs_dtype,
        )
        x_cycle = cycle_start_x

        for local_k in range(cfg.restart):
            if total_iterations >= cfg.max_iterations:
                break

            w = tt_round_capped(operator.matvec(basis[local_k]), cfg.round_tolerance, cfg.max_tt_rank)
            arnoldi_image_norm_tensor = tt_norm_stable_tensor(w)
            for i in range(local_k + 1):
                hij = tt_dot_tensor(basis[i], w)
                hbar[i, local_k] += hij
                w = w + tt_scale(basis[i], -hij)
                if cfg.safe_orth_rounding:
                    w = tt_round_capped(w, cfg.orth_round_tolerance, cfg.max_tt_rank)

            if cfg.reorthogonalize:
                for i in range(local_k + 1):
                    correction_hij = tt_dot_tensor(basis[i], w)
                    hbar[i, local_k] += correction_hij
                    w = w + tt_scale(basis[i], -correction_hij)
                    if cfg.safe_orth_rounding:
                        w = tt_round_capped(
                            w,
                            cfg.orth_round_tolerance,
                            cfg.max_tt_rank,
                        )

            # Safe orthogonalization already rounded the result of the final
            # MGS subtraction.  Only the delayed-rounding branch needs a TT-SVD
            # after the loop.
            if not cfg.safe_orth_rounding:
                w = tt_round_capped(
                    w,
                    cfg.orth_round_tolerance,
                    cfg.max_tt_rank,
                )
            h_next_tensor = tt_norm_stable_tensor(w)
            hbar[local_k + 1, local_k] = h_next_tensor

            total_iterations += 1
            k = local_k + 1
            # One deliberately batched device-to-host synchronization per
            # Arnoldi column.  The large TT/MGS work remains on the GPU while
            # the tiny, potentially rank-deficient least-squares problem keeps
            # NumPy's SVD-based semantics on the CPU.
            packed_hessenberg = torch.cat(
                (
                    hbar[: k + 1, :k].reshape(-1),
                    arnoldi_image_norm_tensor.reshape(1),
                )
            ).detach().cpu().numpy()
            hbar_cpu = packed_hessenberg[:-1].reshape(k + 1, k)
            arnoldi_image_norm = float(packed_hessenberg[-1])
            h_next = float(hbar_cpu[k, k - 1])
            breakdown_scale = max(arnoldi_image_norm, np.finfo(float).tiny)
            near_zero_candidate = (
                h_next <= cfg.breakdown_tolerance * breakdown_scale
            )
            exact_breakdown = bool(
                near_zero_candidate and h_next == 0.0
            )
            if not near_zero_candidate:
                basis.append(tt_scale(w, torch.reciprocal(h_next_tensor)))

            current_basis = list(basis)
            projected_residual = beta

            lucky_invariant_breakdown = bool(
                near_zero_candidate
                and k <= cfg.index
                and _arnoldi_invariance_is_certified(
                    operator,
                    basis,
                    hbar_cpu,
                    k,
                )
            )

            if k > cfg.index:
                # Even when h_(k+1,k) is merely small, the ordinary rectangular
                # DGMRES problem is still the valid candidate construction.  A
                # power of H_k would assume an invariant subspace that TT
                # rounding has not established.
                hhat = build_hhat(hbar_cpu, k, cfg.index)
                e1 = np.zeros(k + 1, dtype=float)
                e1[0] = beta
                xi, *_ = np.linalg.lstsq(hhat, e1, rcond=None)
                projected_residual = float(np.linalg.norm(e1 - hhat @ xi))
                combination_vectors = basis[: k - cfg.index]
                if cfg.tree_solution_combination:
                    correction = tt_linear_combination_capped_tree(
                        combination_vectors,
                        xi,
                        cfg.solution_round_tolerance,
                        max_rank=cfg.max_tt_rank,
                    )
                else:
                    correction = tt_linear_combination(
                        combination_vectors,
                        xi,
                        cfg.solution_round_tolerance,
                        max_rank=cfg.max_tt_rank,
                    )
                x_eval = tt_round_capped(
                    cycle_start_x + correction,
                    cfg.solution_round_tolerance,
                    cfg.max_tt_rank,
                )
                x_cycle = x_eval
            elif lucky_invariant_breakdown:
                invariant_hessenberg = hbar_cpu[:k, :k]
                transformed_hessenberg = np.linalg.matrix_power(
                    invariant_hessenberg,
                    cfg.index + 1,
                )
                e1 = np.zeros(k, dtype=float)
                e1[0] = beta
                xi, *_ = np.linalg.lstsq(
                    transformed_hessenberg,
                    e1,
                    rcond=None,
                )
                projected_residual = float(
                    np.linalg.norm(e1 - transformed_hessenberg @ xi)
                )
                combination_vectors = basis[:k]
                if cfg.tree_solution_combination:
                    correction = tt_linear_combination_capped_tree(
                        combination_vectors,
                        xi,
                        cfg.solution_round_tolerance,
                        max_rank=cfg.max_tt_rank,
                    )
                else:
                    correction = tt_linear_combination(
                        combination_vectors,
                        xi,
                        cfg.solution_round_tolerance,
                        max_rank=cfg.max_tt_rank,
                    )
                x_eval = tt_round_capped(
                    cycle_start_x + correction,
                    cfg.solution_round_tolerance,
                    cfg.max_tt_rank,
                )
                x_cycle = x_eval
            else:
                x_eval = cycle_start_x

            if solution_projector is not None:
                x_eval = solution_projector(x_eval)
                require_problem_layout("solution_projector output", x_eval)
                x_cycle = x_eval

            diagnostic_metrics = _audited_residual_metrics(
                operator=operator,
                rhs=rhs,
                x=x_eval,
                index=cfg.index,
                residual_round_eps=cfg.residual_round_tolerance,
                transformed_round_eps=cfg.transformed_round_tolerance,
                operator_norm=operator_norm,
                max_rank=cfg.max_tt_rank,
                rhs_norm=rhs_norm,
                certify_transformed=cfg.stopping_metric
                in {"a_residual", "relative_a_residual"},
            )
            append_diagnostic(
                metrics=diagnostic_metrics,
                x_eval=x_eval,
                current_basis=current_basis,
                iteration=total_iterations,
                cycle=cycles,
                projected_residual=projected_residual,
            )

            if (
                diagnostic_stop_value(diagnostic_metrics) < cfg.tolerance
                and metric_is_certified(diagnostic_metrics)
            ):
                x = x_eval
                x_cycle = x_eval
                converged = True
                if near_zero_candidate:
                    breakdown = True
                    status = (
                        "residual_target_at_breakdown"
                        if exact_breakdown
                        else "residual_target_at_numerical_breakdown"
                    )
                else:
                    status = "residual_target"
                break

            if near_zero_candidate:
                x = x_eval
                x_cycle = x_eval
                breakdown_terminated = True
                breakdown = True
                status = (
                    "arnoldi_breakdown"
                    if exact_breakdown
                    else "numerical_breakdown"
                )
                break

        x = x_cycle

    if (
        not converged
        and not breakdown
        and cycles >= max_cycles
        and total_iterations < cfg.max_iterations
    ):
        status = "cycle_limit"

    return TTDGMRESResult(
        solution=x,
        converged=converged,
        iterations=total_iterations,
        cycles=cycles,
        operator_norm=operator_norm,
        history=history,
        config=cfg,
        status=status,
        breakdown=breakdown,
    )


def _problem_device(device: torch.device | str | None) -> torch.device:
    """Resolve an optional problem-construction device without global state."""

    return torch.device("cpu") if device is None else torch.device(device)


def singular_diagonal_tt_operator(
    local_diagonals: Sequence[torch.Tensor],
    name: str = "DiagSingular",
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> TTLinearOperator:
    if not local_diagonals:
        raise ValueError("At least one local diagonal is required.")
    resolved_device = (
        local_diagonals[0].device if device is None else torch.device(device)
    )
    resolved_dtype = local_diagonals[0].dtype if dtype is None else dtype
    diagonals = [
        values.to(device=resolved_device, dtype=resolved_dtype)
        for values in local_diagonals
    ]
    cores = [
        torch.diag(values)
        .reshape(1, values.numel(), values.numel(), 1)
        .clone()
        for values in diagonals
    ]
    exact_norm = float(
        np.prod([float(torch.max(torch.abs(values))) for values in diagonals])
    )
    return tt_matrix_operator(
        cores,
        name=name,
        exact_norm=exact_norm,
        device=resolved_device,
        dtype=resolved_dtype,
    )


def diagonal_drazin_demo_problem(
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[TTLinearOperator, TTTensor, TTTensor]:
    resolved_device = _problem_device(device)
    local_diagonals = [
        torch.tensor(
            [0.0, 1.0, 1.5, 2.0], device=resolved_device, dtype=dtype
        ),
        torch.tensor(
            [1.0, 0.5, 1.25, 1.75], device=resolved_device, dtype=dtype
        ),
        torch.tensor(
            [2.0, 1.0, 0.75, 1.5], device=resolved_device, dtype=dtype
        ),
    ]
    operator = singular_diagonal_tt_operator(
        local_diagonals,
        name="DiagSingular3D",
        device=resolved_device,
        dtype=dtype,
    )

    y = tt_rank1(
        [
            torch.tensor(
                [0.2, 0.5, 0.8, 1.1], device=resolved_device, dtype=dtype
            ),
            torch.tensor(
                [1.0, 0.6, 0.4, 0.2], device=resolved_device, dtype=dtype
            ),
            torch.tensor(
                [0.3, 0.7, 1.0, 1.2], device=resolved_device, dtype=dtype
            ),
        ],
        device=resolved_device,
        dtype=dtype,
    )
    x_exact = tt_round(operator.matvec(y), 1e-14)

    null_first = torch.tensor(
        [1.0, 0.0, 0.0, 0.0], device=resolved_device, dtype=dtype
    )
    null_vec = tt_rank1(
        [
            null_first,
            torch.ones(4, device=resolved_device, dtype=dtype),
            torch.ones(4, device=resolved_device, dtype=dtype),
        ],
        device=resolved_device,
        dtype=dtype,
    )

    ax_exact = tt_round(operator.matvec(x_exact), 1e-14)
    alpha = 0.01 * tt_norm(ax_exact) / tt_norm(null_vec)
    rhs = tt_round(ax_exact + alpha * null_vec, 1e-14)
    return operator, rhs, x_exact


def diagonal_drazin_demo_problem_nd(
    mode_size: int = 8,
    dims: int = 10,
    perturbation_ratio: float = 0.01,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[TTLinearOperator, TTTensor, TTTensor]:
    if mode_size < 4:
        raise ValueError("mode_size must be at least 4.")
    if dims < 2:
        raise ValueError("dims must be at least 2.")

    resolved_device = _problem_device(device)
    grid = torch.linspace(
        0.0, 1.0, mode_size, device=resolved_device, dtype=dtype
    )
    local_diagonals = []
    factors = []
    for axis in range(dims):
        diagonal = 0.8 + 0.2 * (axis + 1) + grid + 0.1 * (axis + 1) * grid**2
        diagonal[0] = 0.0
        local_diagonals.append(diagonal.clone())

        factor = 0.4 + torch.cos((axis + 1) * np.pi * grid) + 0.2 * (axis + 1) * grid
        factors.append(factor.clone())

    operator = singular_diagonal_tt_operator(
        local_diagonals,
        name=f"DiagSingular{dims}D",
        device=resolved_device,
        dtype=dtype,
    )
    y = tt_rank1(factors, device=resolved_device, dtype=dtype)
    x_exact = tt_round(operator.matvec(y), 1e-14)

    null_first = torch.zeros(mode_size, device=resolved_device, dtype=dtype)
    null_first[0] = 1.0
    null_vec = tt_rank1(
        [null_first]
        + [
            torch.ones(mode_size, device=resolved_device, dtype=dtype)
            for _ in range(dims - 1)
        ],
        device=resolved_device,
        dtype=dtype,
    )

    ax_exact = tt_round(operator.matvec(x_exact), 1e-14)
    alpha = perturbation_ratio * tt_norm(ax_exact) / tt_norm(null_vec)
    rhs = tt_round(ax_exact + alpha * null_vec, 1e-14)
    return operator, rhs, x_exact


def one_axis_singular_diagonal_operator(
    mode_size: int = 8,
    dims: int = 12,
    zero_axis: int = 0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> TTLinearOperator:
    if mode_size < 3:
        raise ValueError("mode_size must be at least 3.")
    if dims < 2:
        raise ValueError("dims must be at least 2.")
    if not 0 <= zero_axis < dims:
        raise ValueError("zero_axis must satisfy 0 <= zero_axis < dims.")

    resolved_device = _problem_device(device)
    eye = torch.eye(mode_size, device=resolved_device, dtype=dtype)
    diagonal = torch.linspace(
        0.75, 1.25, mode_size, device=resolved_device, dtype=dtype
    )
    diagonal[0] = 0.0

    cores: List[torch.Tensor] = []
    for axis in range(dims):
        core = torch.zeros(
            (1, mode_size, mode_size, 1),
            device=resolved_device,
            dtype=dtype,
        )
        core[0, :, :, 0] = torch.diag(diagonal) if axis == zero_axis else eye
        cores.append(core)

    return tt_matrix_operator(
        cores,
        name=f"OneAxisDiagSingular{dims}D",
        exact_norm=float(torch.max(torch.abs(diagonal))),
        device=resolved_device,
        dtype=dtype,
    )


def massive_diagonal_drazin_demo_problem(
    mode_size: int = 8,
    dims: int = 12,
    perturbation_ratio: float = 0.01,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[TTLinearOperator, TTTensor, TTTensor]:
    resolved_device = _problem_device(device)
    operator = one_axis_singular_diagonal_operator(
        mode_size=mode_size,
        dims=dims,
        device=resolved_device,
        dtype=dtype,
    )

    grid = torch.linspace(
        0.0, 1.0, mode_size, device=resolved_device, dtype=dtype
    )
    factors = []
    for axis in range(dims):
        factor = 0.8 + torch.cos((axis + 1) * np.pi * grid) + 0.1 * grid
        factors.append(factor.clone())
    y = tt_rank1(factors, device=resolved_device, dtype=dtype)

    x_exact = tt_round(operator.matvec(y), 1e-14)

    null_first = torch.zeros(mode_size, device=resolved_device, dtype=dtype)
    null_first[0] = 1.0
    null_vec = tt_rank1(
        [null_first]
        + [
            torch.ones(mode_size, device=resolved_device, dtype=dtype)
            for _ in range(dims - 1)
        ],
        device=resolved_device,
        dtype=dtype,
    )

    ax_exact = tt_round(operator.matvec(x_exact), 1e-14)
    alpha = perturbation_ratio * tt_norm(ax_exact) / tt_norm(null_vec)
    rhs = tt_round(ax_exact + alpha * null_vec, 1e-14)
    return operator, rhs, x_exact


def neumann_1d_laplacian(
    n: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if n < 2:
        raise ValueError("n must be at least 2.")
    resolved_device = _problem_device(device)
    h = 1.0 / (n - 1)
    main = 2.0 * torch.ones(n, device=resolved_device, dtype=dtype)
    main[0] = 1.0
    main[-1] = 1.0
    off = -1.0 * torch.ones(n - 1, device=resolved_device, dtype=dtype)
    matrix = torch.diag(main) + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    return matrix / (h * h)


def paper_neumann_1d_poisson_stencil(
    n: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Second-order Neumann stencil matching the report example."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    resolved_device = _problem_device(device)
    h = 1.0 / (n - 1)
    matrix = torch.zeros((n, n), device=resolved_device, dtype=dtype)
    matrix[0, 0] = -2.0
    matrix[0, 1] = 2.0
    for i in range(1, n - 1):
        matrix[i, i - 1] = 1.0
        matrix[i, i] = -2.0
        matrix[i, i + 1] = 1.0
    matrix[-1, -2] = 2.0
    matrix[-1, -1] = -2.0
    return matrix / (h * h)


def nonsymmetric_first_derivative_neumann(
    n: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if n < 2:
        raise ValueError("n must be at least 2.")
    resolved_device = _problem_device(device)
    h = 1.0 / (n - 1)
    matrix = torch.zeros((n, n), device=resolved_device, dtype=dtype)
    matrix[0, 0] = -1.0
    matrix[0, 1] = 1.0
    for i in range(1, n - 1):
        matrix[i, i - 1] = -0.5
        matrix[i, i + 1] = 0.5
    matrix[-1, -2] = -1.0
    matrix[-1, -1] = 1.0
    return matrix / h


def neumann_operator_norm_exact(n: int, dims: int = 3) -> float:
    lap = neumann_1d_laplacian(n).detach().cpu().numpy()
    lam_max = float(np.max(np.linalg.eigvalsh(lap)))
    return float(dims * lam_max)


def kron_sum_tt_operator(
    local_operator: torch.Tensor,
    dims: int,
    *,
    name: str,
    exact_norm: float | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> TTLinearOperator:
    if dims < 1:
        raise ValueError("dims must be positive.")
    if local_operator.ndim != 2 or local_operator.shape[0] != local_operator.shape[1]:
        raise ValueError("local_operator must be a square matrix.")
    resolved_device = (
        local_operator.device if device is None else torch.device(device)
    )
    resolved_dtype = local_operator.dtype if dtype is None else dtype
    local_operator = local_operator.to(
        device=resolved_device,
        dtype=resolved_dtype,
    )
    n = int(local_operator.shape[0])
    eye = torch.eye(n, device=resolved_device, dtype=resolved_dtype)

    if dims == 1:
        core = local_operator.reshape(1, n, n, 1).clone()
        return tt_matrix_operator(
            [core],
            name=name,
            exact_norm=exact_norm,
            device=resolved_device,
            dtype=resolved_dtype,
        )

    cores: List[torch.Tensor] = []

    first = torch.zeros(
        (1, n, n, 2), device=resolved_device, dtype=resolved_dtype
    )
    first[0, :, :, 0] = eye
    first[0, :, :, 1] = local_operator
    cores.append(first)

    for _ in range(dims - 2):
        middle = torch.zeros(
            (2, n, n, 2), device=resolved_device, dtype=resolved_dtype
        )
        middle[0, :, :, 0] = eye
        middle[0, :, :, 1] = local_operator
        middle[1, :, :, 1] = eye
        cores.append(middle)

    last = torch.zeros(
        (2, n, n, 1), device=resolved_device, dtype=resolved_dtype
    )
    last[0, :, :, 0] = local_operator
    last[1, :, :, 0] = eye
    cores.append(last)

    return tt_matrix_operator(
        cores,
        name=name,
        exact_norm=exact_norm,
        device=resolved_device,
        dtype=resolved_dtype,
    )


def variable_kron_sum_tt_operator(
    local_operators: Sequence[torch.Tensor],
    *,
    name: str,
    exact_norm: float | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> TTLinearOperator:
    dims = len(local_operators)
    if dims < 1:
        raise ValueError("At least one local operator is required.")
    n = int(local_operators[0].shape[0])
    if any(tuple(op.shape) != (n, n) for op in local_operators):
        raise ValueError("All local operators must have the same square shape.")
    resolved_device = (
        local_operators[0].device if device is None else torch.device(device)
    )
    resolved_dtype = local_operators[0].dtype if dtype is None else dtype
    if device is None and any(op.device != resolved_device for op in local_operators):
        raise ValueError("All local operators must be on the same device.")
    if dtype is None and any(op.dtype != resolved_dtype for op in local_operators):
        raise ValueError("All local operators must have the same dtype.")
    prepared_operators = [
        op.to(device=resolved_device, dtype=resolved_dtype)
        for op in local_operators
    ]

    eye = torch.eye(n, device=resolved_device, dtype=resolved_dtype)
    if dims == 1:
        core = prepared_operators[0].reshape(1, n, n, 1).clone()
        return tt_matrix_operator(
            [core],
            name=name,
            exact_norm=exact_norm,
            device=resolved_device,
            dtype=resolved_dtype,
        )

    cores: List[torch.Tensor] = []

    first = torch.zeros(
        (1, n, n, 2), device=resolved_device, dtype=resolved_dtype
    )
    first[0, :, :, 0] = eye
    first[0, :, :, 1] = prepared_operators[0]
    cores.append(first)

    for axis in range(1, dims - 1):
        middle = torch.zeros(
            (2, n, n, 2), device=resolved_device, dtype=resolved_dtype
        )
        middle[0, :, :, 0] = eye
        middle[0, :, :, 1] = prepared_operators[axis]
        middle[1, :, :, 1] = eye
        cores.append(middle)

    last = torch.zeros(
        (2, n, n, 1), device=resolved_device, dtype=resolved_dtype
    )
    last[0, :, :, 0] = prepared_operators[-1]
    last[1, :, :, 0] = eye
    cores.append(last)

    return tt_matrix_operator(
        cores,
        name=name,
        exact_norm=exact_norm,
        device=resolved_device,
        dtype=resolved_dtype,
    )


def neumann_poisson_operator_3d(
    n: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> TTLinearOperator:
    resolved_device = _problem_device(device)
    eye = torch.eye(n, device=resolved_device, dtype=dtype)
    lap = neumann_1d_laplacian(n, device=resolved_device, dtype=dtype)

    core1 = torch.zeros((1, n, n, 2), device=resolved_device, dtype=dtype)
    core1[0, :, :, 0] = eye
    core1[0, :, :, 1] = lap

    core2 = torch.zeros((2, n, n, 2), device=resolved_device, dtype=dtype)
    core2[0, :, :, 0] = eye
    core2[0, :, :, 1] = lap
    core2[1, :, :, 1] = eye

    core3 = torch.zeros((2, n, n, 1), device=resolved_device, dtype=dtype)
    core3[0, :, :, 0] = lap
    core3[1, :, :, 0] = eye

    exact_norm = neumann_operator_norm_exact(n)
    return tt_matrix_operator(
        [core1, core2, core3],
        name="Poisson3DNeumann",
        exact_norm=exact_norm,
        device=resolved_device,
        dtype=dtype,
    )


def paper_neumann_poisson_operator_3d(
    n: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> TTLinearOperator:
    resolved_device = _problem_device(device)
    eye = torch.eye(n, device=resolved_device, dtype=dtype)
    local_op = paper_neumann_1d_poisson_stencil(
        n, device=resolved_device, dtype=dtype
    )

    core1 = torch.zeros((1, n, n, 2), device=resolved_device, dtype=dtype)
    core1[0, :, :, 0] = eye
    core1[0, :, :, 1] = local_op

    core2 = torch.zeros((2, n, n, 2), device=resolved_device, dtype=dtype)
    core2[0, :, :, 0] = eye
    core2[0, :, :, 1] = local_op
    core2[1, :, :, 1] = eye

    core3 = torch.zeros((2, n, n, 1), device=resolved_device, dtype=dtype)
    core3[0, :, :, 0] = local_op
    core3[1, :, :, 0] = eye

    norm_bound = 3.0 * local_operator_norm_exact(local_op)
    return tt_matrix_operator(
        [core1, core2, core3],
        name="PaperPoisson3DNeumann",
        exact_norm=norm_bound,
        device=resolved_device,
        dtype=dtype,
    )


def neumann_poisson_drazin_demo_problem(
    n: int = 17,
    perturbation_ratio: float = 0.01,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[TTLinearOperator, TTTensor, TTTensor]:
    resolved_device = _problem_device(device)
    operator = neumann_poisson_operator_3d(
        n, device=resolved_device, dtype=dtype
    )

    grid = torch.linspace(0.0, 1.0, n, device=resolved_device, dtype=dtype)
    factors = [
        0.5 + torch.cos(np.pi * grid),
        1.0 + grid - grid**2,
        0.25 + torch.sin(2.0 * np.pi * grid) + grid**2,
    ]
    y = tt_rank1(factors, device=resolved_device, dtype=dtype)

    x_exact = tt_round(operator.matvec(y), 1e-12)
    null_vec = tt_rank1(
        [
            torch.ones(n, device=resolved_device, dtype=dtype),
            torch.ones(n, device=resolved_device, dtype=dtype),
            torch.ones(n, device=resolved_device, dtype=dtype),
        ],
        device=resolved_device,
        dtype=dtype,
    )

    ax_exact = tt_round(operator.matvec(x_exact), 1e-12)
    alpha = perturbation_ratio * tt_norm(ax_exact) / tt_norm(null_vec)
    rhs = tt_round(ax_exact + alpha * null_vec, 1e-12)
    return operator, rhs, x_exact


def paper_neumann_poisson_demo_problem(
    n: int = 17,
    perturbation_ratio: float = 0.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[TTLinearOperator, TTTensor, TTTensor]:
    resolved_device = _problem_device(device)
    operator = paper_neumann_poisson_operator_3d(
        n, device=resolved_device, dtype=dtype
    )

    grid = torch.linspace(0.0, 1.0, n, device=resolved_device, dtype=dtype)
    factors = [
        0.5 + torch.cos(np.pi * grid),
        1.0 + grid - grid**2,
        0.25 + torch.sin(2.0 * np.pi * grid) + grid**2,
    ]
    y = tt_rank1(factors, device=resolved_device, dtype=dtype)

    x_exact = tt_round(operator.matvec(y), 1e-12)
    null_vec = tt_rank1(
        [
            torch.ones(n, device=resolved_device, dtype=dtype),
            torch.ones(n, device=resolved_device, dtype=dtype),
            torch.ones(n, device=resolved_device, dtype=dtype),
        ],
        device=resolved_device,
        dtype=dtype,
    )

    ax_exact = tt_round(operator.matvec(x_exact), 1e-12)
    alpha = perturbation_ratio * tt_norm(ax_exact) / tt_norm(null_vec)
    rhs = tt_round(ax_exact + alpha * null_vec, 1e-12)
    return operator, rhs, x_exact


def neumann_poisson_operator_nd(
    n: int,
    dims: int = 5,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> TTLinearOperator:
    resolved_device = _problem_device(device)
    lap = neumann_1d_laplacian(n, device=resolved_device, dtype=dtype)
    exact_norm = neumann_operator_norm_exact(n, dims=dims)
    return kron_sum_tt_operator(
        lap,
        dims=dims,
        name=f"Poisson{dims}DNeumann",
        exact_norm=exact_norm,
        device=resolved_device,
        dtype=dtype,
    )


def paper_neumann_poisson_operator_nd(
    n: int,
    dims: int = 5,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> TTLinearOperator:
    resolved_device = _problem_device(device)
    local_op = paper_neumann_1d_poisson_stencil(
        n, device=resolved_device, dtype=dtype
    )
    norm_bound = float(dims) * local_operator_norm_exact(local_op)
    return kron_sum_tt_operator(
        local_op,
        dims=dims,
        name=f"PaperPoisson{dims}DNeumann",
        exact_norm=norm_bound,
        device=resolved_device,
        dtype=dtype,
    )


def neumann_poisson_drazin_demo_problem_nd(
    n: int = 17,
    dims: int = 5,
    perturbation_ratio: float = 0.01,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[TTLinearOperator, TTTensor, TTTensor]:
    resolved_device = _problem_device(device)
    operator = neumann_poisson_operator_nd(
        n=n, dims=dims, device=resolved_device, dtype=dtype
    )

    grid = torch.linspace(0.0, 1.0, n, device=resolved_device, dtype=dtype)
    base_factors = [
        0.5 + torch.cos(np.pi * grid),
        1.0 + grid - grid**2,
        0.25 + torch.sin(2.0 * np.pi * grid) + grid**2,
        0.75 + torch.cos(3.0 * np.pi * grid) + 0.5 * grid,
        0.6 + torch.sin(np.pi * grid) + grid**3,
    ]
    factors = [base_factors[idx % len(base_factors)].clone() for idx in range(dims)]
    y = tt_rank1(factors, device=resolved_device, dtype=dtype)

    x_exact = tt_round(operator.matvec(y), 1e-12)
    null_vec = tt_rank1(
        [
            torch.ones(n, device=resolved_device, dtype=dtype)
            for _ in range(dims)
        ],
        device=resolved_device,
        dtype=dtype,
    )

    ax_exact = tt_round(operator.matvec(x_exact), 1e-12)
    alpha = perturbation_ratio * tt_norm(ax_exact) / tt_norm(null_vec)
    rhs = tt_round(ax_exact + alpha * null_vec, 1e-12)
    return operator, rhs, x_exact


def paper_neumann_poisson_demo_problem_nd(
    n: int = 17,
    dims: int = 5,
    perturbation_ratio: float = 0.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[TTLinearOperator, TTTensor, TTTensor]:
    resolved_device = _problem_device(device)
    operator = paper_neumann_poisson_operator_nd(
        n=n, dims=dims, device=resolved_device, dtype=dtype
    )

    grid = torch.linspace(0.0, 1.0, n, device=resolved_device, dtype=dtype)
    base_factors = [
        0.5 + torch.cos(np.pi * grid),
        1.0 + grid - grid**2,
        0.25 + torch.sin(2.0 * np.pi * grid) + grid**2,
        0.75 + torch.cos(3.0 * np.pi * grid) + 0.5 * grid,
        0.6 + torch.sin(np.pi * grid) + grid**3,
    ]
    factors = [base_factors[idx % len(base_factors)].clone() for idx in range(dims)]
    y = tt_rank1(factors, device=resolved_device, dtype=dtype)

    x_exact = tt_round(operator.matvec(y), 1e-12)
    null_vec = tt_rank1(
        [
            torch.ones(n, device=resolved_device, dtype=dtype)
            for _ in range(dims)
        ],
        device=resolved_device,
        dtype=dtype,
    )

    ax_exact = tt_round(operator.matvec(x_exact), 1e-12)
    alpha = perturbation_ratio * tt_norm(ax_exact) / tt_norm(null_vec)
    rhs = tt_round(ax_exact + alpha * null_vec, 1e-12)
    return operator, rhs, x_exact


def complex_diagonal_kron_sum_drazin_demo_problem(
    mode_size: int = 8,
    dims: int = 6,
    perturbation_ratio: float = 0.01,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[TTLinearOperator, TTTensor, TTTensor]:
    if mode_size < 4:
        raise ValueError("mode_size must be at least 4.")
    if dims < 2:
        raise ValueError("dims must be at least 2.")

    resolved_device = _problem_device(device)
    grid = torch.linspace(
        0.0, 1.0, mode_size, device=resolved_device, dtype=dtype
    )
    local_operators = []
    factors = []
    exact_norm = 0.0
    for axis in range(dims):
        diagonal = 0.2 * (axis + 1) + grid + 0.35 * (axis + 1) * grid**2
        diagonal[0] = 0.0
        local_operators.append(torch.diag(diagonal))
        exact_norm += float(torch.max(torch.abs(diagonal)))

        factor = 0.7 + torch.cos((axis + 1) * np.pi * grid) + 0.15 * (axis + 1) * grid
        factors.append(factor.clone())

    operator = variable_kron_sum_tt_operator(
        local_operators,
        name=f"ComplexDiagKronSum{dims}D",
        exact_norm=exact_norm,
        device=resolved_device,
        dtype=dtype,
    )
    y = tt_rank1(factors, device=resolved_device, dtype=dtype)
    x_exact = tt_round(operator.matvec(y), 1e-12)

    e0 = torch.zeros(mode_size, device=resolved_device, dtype=dtype)
    e0[0] = 1.0
    null_vec = tt_rank1(
        [e0.clone() for _ in range(dims)],
        device=resolved_device,
        dtype=dtype,
    )

    ax_exact = tt_round(operator.matvec(x_exact), 1e-12)
    alpha = perturbation_ratio * tt_norm(ax_exact) / tt_norm(null_vec)
    rhs = tt_round(ax_exact + alpha * null_vec, 1e-12)
    return operator, rhs, x_exact


def nonsymmetric_neumann_operator_nd(
    n: int = 8,
    dims: int = 6,
    base_convection: float = 0.08,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> TTLinearOperator:
    if n < 2:
        raise ValueError("n must be at least 2.")
    if dims < 2:
        raise ValueError("dims must be at least 2.")

    resolved_device = _problem_device(device)
    local_operators = []
    norm_bound = 0.0
    lap = neumann_1d_laplacian(n, device=resolved_device, dtype=dtype)
    deriv = nonsymmetric_first_derivative_neumann(
        n, device=resolved_device, dtype=dtype
    )
    for axis in range(dims):
        strength = base_convection * (1.0 + 0.25 * axis)
        local_op = lap + strength * deriv
        local_operators.append(local_op)
        norm_bound += local_operator_norm_exact(local_op)

    return variable_kron_sum_tt_operator(
        local_operators,
        name=f"NonSymNeumannKronSum{dims}D",
        exact_norm=norm_bound,
        device=resolved_device,
        dtype=dtype,
    )


def nonsymmetric_neumann_drazin_demo_problem_nd(
    n: int = 8,
    dims: int = 6,
    base_convection: float = 0.08,
    perturbation_ratio: float = 0.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[TTLinearOperator, TTTensor, TTTensor]:
    resolved_device = _problem_device(device)
    operator = nonsymmetric_neumann_operator_nd(
        n=n,
        dims=dims,
        base_convection=base_convection,
        device=resolved_device,
        dtype=dtype,
    )

    grid = torch.linspace(0.0, 1.0, n, device=resolved_device, dtype=dtype)
    base_factors = [
        0.5 + torch.cos(np.pi * grid),
        1.0 + grid - grid**2,
        0.25 + torch.sin(2.0 * np.pi * grid) + grid**2,
        0.75 + torch.cos(3.0 * np.pi * grid) + 0.5 * grid,
        0.6 + torch.sin(np.pi * grid) + grid**3,
        0.9 + 0.3 * torch.cos(2.0 * np.pi * grid) + grid**2,
    ]
    factors = [base_factors[axis % len(base_factors)].clone() for axis in range(dims)]
    y = tt_rank1(factors, device=resolved_device, dtype=dtype)

    x_exact = tt_round(operator.matvec(y), 1e-12)
    null_vec = tt_rank1(
        [
            torch.ones(n, device=resolved_device, dtype=dtype)
            for _ in range(dims)
        ],
        device=resolved_device,
        dtype=dtype,
    )

    ax_exact = tt_round(operator.matvec(x_exact), 1e-12)
    alpha = perturbation_ratio * tt_norm(ax_exact) / tt_norm(null_vec)
    rhs = tt_round(ax_exact + alpha * null_vec, 1e-12)
    return operator, rhs, x_exact


def kron_sum_norm_exact(local_operator: torch.Tensor, dims: int = 3) -> float:
    op = local_operator.detach().cpu().numpy()
    n = op.shape[0]
    eye_np = np.eye(n)
    total = np.zeros((n**dims, n**dims), dtype=np.float64)
    for pos in range(dims):
        term = np.array([[1.0]], dtype=np.float64)
        for axis in range(dims):
            factor = op if axis == pos else eye_np
            term = np.kron(term, factor)
        total += term
    return float(np.linalg.norm(total, ord=2))


def local_operator_norm_exact(local_operator: torch.Tensor) -> float:
    return float(
        np.linalg.norm(local_operator.detach().cpu().numpy(), ord=2)
    )


def kron_sum_norm_bound(local_operator: torch.Tensor, dims: int = 3) -> float:
    # For larger nonsymmetric problems we avoid forming the full dense kron-sum.
    # This bound is only used to scale the backward error and keeps the demo scalable.
    return float(dims * local_operator_norm_exact(local_operator))


def nonsymmetric_singular_operator_3d(
    n: int,
    convection_strength: float = 0.15,
    norm_strategy: str = "auto",
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> TTLinearOperator:
    resolved_device = _problem_device(device)
    eye = torch.eye(n, device=resolved_device, dtype=dtype)
    local_op = neumann_1d_laplacian(
        n, device=resolved_device, dtype=dtype
    ) + convection_strength * nonsymmetric_first_derivative_neumann(
        n, device=resolved_device, dtype=dtype
    )

    core1 = torch.zeros((1, n, n, 2), device=resolved_device, dtype=dtype)
    core1[0, :, :, 0] = eye
    core1[0, :, :, 1] = local_op

    core2 = torch.zeros((2, n, n, 2), device=resolved_device, dtype=dtype)
    core2[0, :, :, 0] = eye
    core2[0, :, :, 1] = local_op
    core2[1, :, :, 1] = eye

    core3 = torch.zeros((2, n, n, 1), device=resolved_device, dtype=dtype)
    core3[0, :, :, 0] = local_op
    core3[1, :, :, 0] = eye

    if norm_strategy == "exact":
        operator_norm = kron_sum_norm_exact(local_op, dims=3)
    elif norm_strategy == "bound":
        operator_norm = kron_sum_norm_bound(local_op, dims=3)
    elif norm_strategy == "auto":
        operator_norm = kron_sum_norm_exact(local_op, dims=3) if n <= 17 else kron_sum_norm_bound(local_op, dims=3)
    elif norm_strategy == "none":
        operator_norm = None
    else:
        raise ValueError(f"Unknown norm_strategy: {norm_strategy}")

    return tt_matrix_operator(
        [core1, core2, core3],
        name="Poisson3DNonSymSingular",
        exact_norm=operator_norm,
        device=resolved_device,
        dtype=dtype,
    )


def nonsymmetric_poisson_drazin_demo_problem(
    n: int = 7,
    convection_strength: float = 0.15,
    perturbation_ratio: float = 0.01,
    norm_strategy: str = "auto",
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[TTLinearOperator, TTTensor, TTTensor]:
    resolved_device = _problem_device(device)
    operator = nonsymmetric_singular_operator_3d(
        n=n,
        convection_strength=convection_strength,
        norm_strategy=norm_strategy,
        device=resolved_device,
        dtype=dtype,
    )

    grid = torch.linspace(0.0, 1.0, n, device=resolved_device, dtype=dtype)
    factors = [
        0.5 + torch.cos(np.pi * grid),
        1.0 + grid - grid**2,
        0.25 + torch.sin(2.0 * np.pi * grid) + grid**2,
    ]
    y = tt_rank1(factors, device=resolved_device, dtype=dtype)

    x_exact = tt_round(operator.matvec(y), 1e-12)
    null_vec = tt_rank1(
        [
            torch.ones(n, device=resolved_device, dtype=dtype),
            torch.ones(n, device=resolved_device, dtype=dtype),
            torch.ones(n, device=resolved_device, dtype=dtype),
        ],
        device=resolved_device,
        dtype=dtype,
    )

    ax_exact = tt_round(operator.matvec(x_exact), 1e-12)
    alpha = perturbation_ratio * tt_norm(ax_exact) / tt_norm(null_vec)
    rhs = tt_round(ax_exact + alpha * null_vec, 1e-12)
    return operator, rhs, x_exact


def default_dgmres_config() -> TTDGMRESConfig:
    return TTDGMRESConfig(
        index=1,
        restart=10,
        max_iterations=30,
        tolerance=1e-8,
        round_tolerance=1e-10,
        operator_norm_mode="exact",
    )


def default_phase2_config() -> TTDGMRESConfig:
    return TTDGMRESConfig(
        index=1,
        restart=20,
        max_iterations=60,
        tolerance=1e-8,
        round_tolerance=1e-10,
        operator_norm_mode="exact",
        stopping_metric="relative_a_residual",
    )


def default_phase2_nonsym_config() -> TTDGMRESConfig:
    return TTDGMRESConfig(
        index=1,
        # The nonsymmetric singular test is noticeably more restart-sensitive.
        # A longer cycle keeps enough Krylov information to drive ||A^a r|| down.
        restart=40,
        max_iterations=120,
        tolerance=1e-8,
        round_tolerance=1e-10,
        operator_norm_mode="exact",
    )


def default_phase3_large_config() -> TTDGMRESConfig:
    return TTDGMRESConfig(
        index=1,
        restart=30,
        max_iterations=90,
        tolerance=1e-8,
        round_tolerance=1e-10,
        operator_norm_mode="exact",
        stopping_metric="relative_a_residual",
    )


def default_phase3_large_diag_config() -> TTDGMRESConfig:
    return TTDGMRESConfig(
        index=1,
        restart=30,
        max_iterations=240,
        tolerance=1e-8,
        round_tolerance=1e-10,
        operator_norm_mode="exact",
        stopping_metric="a_residual",
        max_tt_rank=50,
    )


def default_phase4_massive_diag_config() -> TTDGMRESConfig:
    return TTDGMRESConfig(
        index=1,
        restart=12,
        max_iterations=36,
        tolerance=1e-8,
        round_tolerance=1e-11,
        operator_norm_mode="exact",
        stopping_metric="a_residual",
        max_tt_rank=20,
    )


def default_phase5_complex_diag_config() -> TTDGMRESConfig:
    return TTDGMRESConfig(
        index=1,
        restart=30,
        max_iterations=180,
        tolerance=1e-8,
        round_tolerance=1e-10,
        operator_norm_mode="exact",
        stopping_metric="a_residual",
        max_tt_rank=60,
    )


def default_phase6_complex_8x12_config() -> TTDGMRESConfig:
    return TTDGMRESConfig(
        index=1,
        restart=30,
        max_iterations=360,
        tolerance=1e-8,
        round_tolerance=1e-10,
        operator_norm_mode="exact",
        stopping_metric="a_residual",
        max_tt_rank=80,
    )


def default_phase7_nonsym_complex_config() -> TTDGMRESConfig:
    return TTDGMRESConfig(
        index=1,
        restart=30,
        max_iterations=180,
        tolerance=1e-8,
        round_tolerance=1e-9,
        operator_norm_mode="exact",
        stopping_metric="a_residual",
        max_tt_rank=80,
    )


def summary_dict(result: TTDGMRESResult) -> Dict[str, float | int | bool | str]:
    history = result.history
    return {
        "converged": result.converged,
        "status": result.status,
        "breakdown": result.breakdown,
        "iterations": result.iterations,
        "cycles": result.cycles,
        "final_backward_error": history["backward_error"][-1] if history["backward_error"] else math.nan,
        "final_true_residual_norm": history["true_residual_norm"][-1] if history["true_residual_norm"] else math.nan,
        "final_transformed_residual_norm": (
            history["transformed_residual_norm"][-1] if history["transformed_residual_norm"] else math.nan
        ),
        "final_relative_transformed_residual_norm": (
            history["relative_transformed_residual_norm"][-1]
            if history.get("relative_transformed_residual_norm")
            else math.nan
        ),
        "final_relative_solution_error": (
            history["relative_solution_error"][-1] if history["relative_solution_error"] else math.nan
        ),
        "max_solution_rank": max(history["solution_rank"]) if history["solution_rank"] else 0,
        "max_last_vector_rank": max(history["last_vector_rank"]) if history["last_vector_rank"] else 0,
        "peak_basis_numcoef": max(history.get("basis_numcoef", []) or [0.0]),
        "peak_basis_storage_bytes": max(history.get("basis_storage_bytes", []) or [0.0]),
        "peak_dense_basis_storage_bytes": max(history.get("dense_basis_storage_bytes", []) or [0.0]),
    }


def save_single_run_plot(result: TTDGMRESResult, output_path: Path | str) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = result.history

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].semilogy(history["iteration"], history["transformed_residual_norm"], marker="x")
    axes[0].set_title(r"Transformed Residual $\|A^a r_k\|$")
    axes[0].set_xlabel("iteration")
    axes[0].grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    axes[1].semilogy(history["iteration"], history["backward_error"], marker="x")
    axes[1].set_title("Backward Error")
    axes[1].set_xlabel("iteration")
    axes[1].grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    axes[2].plot(history["iteration"], history["solution_rank"], marker="x")
    axes[2].set_title("Solution TT-rank")
    axes[2].set_xlabel("iteration")
    axes[2].grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    axes[3].semilogy(history["iteration"], history["relative_solution_error"], marker="x")
    axes[3].set_title("Relative Solution Error")
    axes[3].set_xlabel("iteration")
    axes[3].grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.suptitle("Single TT-DGMRES Run", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")
    return output_path


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "phase2"

    if mode == "phase1":
        config = default_dgmres_config()
        operator, rhs, x_exact = diagonal_drazin_demo_problem()
        output_dir = Path("outputs_tt_dgmres_demo")
        title = "TT-DGMRES Phase 1"
    elif mode == "phase2":
        config = default_phase2_config()
        operator, rhs, x_exact = paper_neumann_poisson_demo_problem(n=7, perturbation_ratio=0.0)
        output_dir = Path("outputs_tt_dgmres_phase2")
        title = "TT-DGMRES Phase 2 (Paper Neumann-Poisson)"
    elif mode == "phase2_nonsym":
        config = default_phase2_nonsym_config()
        operator, rhs, x_exact = nonsymmetric_poisson_drazin_demo_problem(
            n=7,
            convection_strength=0.15,
            perturbation_ratio=0.01,
        )
        output_dir = Path("outputs_tt_dgmres_phase2_nonsym")
        title = "TT-DGMRES Phase 2 (Nonsymmetric Singular)"
    elif mode == "phase3_large":
        config = default_phase3_large_config()
        operator, rhs, x_exact = paper_neumann_poisson_demo_problem_nd(
            n=17,
            dims=5,
            perturbation_ratio=0.0,
        )
        output_dir = Path("outputs_tt_dgmres_phase3_large")
        title = "TT-DGMRES Phase 3 (Large-Scale Paper Neumann-Poisson)"
    elif mode == "phase3_large_diag":
        config = default_phase3_large_diag_config()
        operator, rhs, x_exact = diagonal_drazin_demo_problem_nd(
            mode_size=8,
            dims=5,
            perturbation_ratio=0.01,
        )
        output_dir = Path("outputs_tt_dgmres_phase3_large_diag")
        title = "TT-DGMRES Phase 3 (Large-Scale Diagonal Singular)"
    elif mode == "phase4_massive_diag":
        config = default_phase4_massive_diag_config()
        operator, rhs, x_exact = massive_diagonal_drazin_demo_problem(
            mode_size=8,
            dims=12,
            perturbation_ratio=0.01,
        )
        output_dir = Path("outputs_tt_dgmres_phase4_massive_diag")
        title = "TT-DGMRES Phase 4 (Massive Diagonal Singular)"
    elif mode == "phase5_complex_diag":
        config = default_phase5_complex_diag_config()
        operator, rhs, x_exact = complex_diagonal_kron_sum_drazin_demo_problem(
            mode_size=8,
            dims=6,
            perturbation_ratio=0.01,
        )
        output_dir = Path("outputs_tt_dgmres_phase5_complex_diag")
        title = "TT-DGMRES Phase 5 (Complex Diagonal Kron-Sum Singular)"
    elif mode == "phase6_complex_8x12":
        config = default_phase6_complex_8x12_config()
        operator, rhs, x_exact = complex_diagonal_kron_sum_drazin_demo_problem(
            mode_size=8,
            dims=12,
            perturbation_ratio=0.01,
        )
        output_dir = Path("outputs_tt_dgmres_phase6_complex_8x12")
        title = "TT-DGMRES Phase 6 (Complex 8^12 Diagonal Kron-Sum Singular)"
    elif mode == "phase7_nonsym_complex":
        config = default_phase7_nonsym_complex_config()
        operator, rhs, x_exact = nonsymmetric_neumann_drazin_demo_problem_nd(
            n=7,
            dims=4,
            base_convection=0.08,
            perturbation_ratio=0.0,
        )
        output_dir = Path("outputs_tt_dgmres_phase7_nonsym_complex")
        title = "TT-DGMRES Phase 7 (Complex Nonsymmetric Singular)"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    result = tt_dgmres_paper(operator, rhs, config=config, reference_solution=x_exact)

    summary = summary_dict(result)
    print("Config:")
    print(config)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))

    output_dir.mkdir(exist_ok=True)
    history_path = output_dir / "paper_tt_dgmres_history.json"
    history_path.write_text(json.dumps(result.history, indent=2), encoding="utf-8")
    print(f"\nSaved history to {history_path}")

    plot_path = save_single_run_plot(result, output_dir / "paper_tt_dgmres_plot.png")
    print(f"Run label: {title}")
    if "ipykernel" in sys.modules:
        try:
            from IPython.display import Image, display

            display(Image(filename=str(plot_path)))
        except Exception:
            pass


if __name__ == "__main__":
    main()
