from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch

from .paper_tt_dgmres import (
    TTDGMRESConfig,
    TTDGMRESResult,
    _arnoldi_invariance_is_certified,
    _append_history,
    _certified_transformed_residual_norm,
    _compute_residual_metrics,
    _empty_history,
    _relative_solution_error,
    _resolve_operator_norm,
    _stable_tt_norm,
    build_hhat,
    summary_dict,
    tt_apply_power,
    tt_linear_combination,
    tt_linear_combination_capped_tree,
    tt_norm_stable_tensor,
    tt_round_capped,
)
from .paper_tt_gmres import (
    TTLinearOperator,
    TTTensor,
    tt_device,
    tt_dot_tensor,
    tt_dtype,
    tt_max_rank,
    tt_scalar,
    tt_scale,
    tt_zeros,
)
from .tt_dgmres_neumann_poisson_example import (
    build_neumann_poisson_problem,
    default_config,
    make_neumann_range_projector,
)


torch.set_default_dtype(torch.float64)


def _tt_core_storage_bytes(tensor: TTTensor) -> int:
    """Return the logical storage occupied by a TT tensor's cores."""

    return sum(int(core.numel()) * int(core.element_size()) for core in tensor.cores)


def _diagnostic_residual_metrics(
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
    """Reconstruct residuals and audit whether the diagnostic rank cap is active."""

    residual_unrounded = rhs - operator.matvec(x)
    residual_pre_rank = tt_max_rank(residual_unrounded)
    # Ordinary-residual and backward-error stopping must use the untruncated
    # tensor.  The rounded copy below is retained only for rank diagnostics and
    # for a non-certifying plotting fallback.
    residual_norm = _stable_tt_norm(residual_unrounded)
    residual = tt_round_capped(residual_unrounded, residual_round_eps, max_rank)
    residual_rank = tt_max_rank(residual)
    residual_cap_hit = bool(
        max_rank is not None
        and residual_pre_rank > max_rank
        and residual_rank >= max_rank
    )
    resolved_rhs_norm = (
        _stable_tt_norm(rhs) if rhs_norm is None else float(rhs_norm)
    )
    denom = operator_norm * _stable_tt_norm(x) + resolved_rhs_norm
    backward_eta = residual_norm if denom == 0.0 else residual_norm / denom

    transformed = residual
    transformed_pre_rank = residual_pre_rank
    transformed_rank = residual_rank
    transformed_cap_hit = False
    for _ in range(index):
        transformed_unrounded = operator.matvec(transformed)
        transformed_pre_rank = tt_max_rank(transformed_unrounded)
        transformed = tt_round_capped(
            transformed_unrounded,
            transformed_round_eps,
            max_rank,
        )
        transformed_rank = tt_max_rank(transformed)
        transformed_cap_hit = transformed_cap_hit or bool(
            max_rank is not None
            and transformed_pre_rank > max_rank
            and transformed_rank >= max_rank
        )

    rounded_transformed_norm = _stable_tt_norm(transformed)
    if certify_transformed:
        (
            certified_transformed_norm,
            transformed_certified,
            audit_pre_rank,
            audit_rank,
        ) = _certified_transformed_residual_norm(
            operator,
            residual_unrounded,
            index,
            max_rank,
        )
    else:
        certified_transformed_norm = float("nan")
        transformed_certified = False
        audit_pre_rank = float(residual_pre_rank)
        audit_rank = audit_pre_rank
    reported_transformed_norm = (
        certified_transformed_norm
        if transformed_certified
        else rounded_transformed_norm
    )

    return {
        "backward_error": float(backward_eta),
        "residual_norm": float(residual_norm),
        "transformed_residual_norm": float(reported_transformed_norm),
        "rounded_transformed_residual_norm": rounded_transformed_norm,
        "transformed_residual_certified": float(transformed_certified),
        "audit_transformed_pre_rank": float(audit_pre_rank),
        "audit_transformed_rank": float(audit_rank),
        "residual_pre_round_rank": float(residual_pre_rank),
        "residual_rank": float(residual_rank),
        "transformed_pre_round_rank": float(transformed_pre_rank),
        "transformed_rank": float(transformed_rank),
        "residual_rank_cap_hit": float(residual_cap_hit),
        "transformed_rank_cap_hit": float(transformed_cap_hit),
        "rank_cap_hit": float(residual_cap_hit or transformed_cap_hit),
    }


def _diagnostic_metric_is_valid(
    *,
    stopping_metric: str,
    current_metrics: Dict[str, float],
    initial_metrics: Dict[str, float],
) -> bool:
    """Return whether the tensors used by the selected stop metric avoided a cap.

    A transformed-residual cap must not invalidate an ordinary-residual or
    backward-error test: those two metrics use only the residual tensor.  The
    initial transformed residual matters only for the relative transformed
    residual because it supplies that metric's denominator.
    """

    if stopping_metric in {"relative_true_residual", "backward_error"}:
        # These values are contracted from the untruncated residual.
        return True
    if stopping_metric == "a_residual":
        return not (
            bool(current_metrics["residual_rank_cap_hit"])
            or bool(current_metrics["transformed_rank_cap_hit"])
            or not bool(current_metrics.get("transformed_residual_certified", 1.0))
        )
    if stopping_metric == "relative_a_residual":
        return not (
            bool(initial_metrics["residual_rank_cap_hit"])
            or bool(initial_metrics["transformed_rank_cap_hit"])
            or not bool(initial_metrics.get("transformed_residual_certified", 1.0))
            or bool(current_metrics["residual_rank_cap_hit"])
            or bool(current_metrics["transformed_rank_cap_hit"])
            or not bool(current_metrics.get("transformed_residual_certified", 1.0))
        )
    raise ValueError(f"Unsupported stopping_metric: {stopping_metric}")


def _diagnostic_stop_value(
    *,
    stopping_metric: str,
    metrics: Dict[str, float],
    initial_transformed_residual_norm: float,
    rhs_norm: float,
) -> float:
    if stopping_metric == "a_residual":
        return float(metrics["transformed_residual_norm"])
    if stopping_metric == "relative_a_residual":
        denominator = max(initial_transformed_residual_norm, np.finfo(float).tiny)
        return float(metrics["transformed_residual_norm"]) / denominator
    if stopping_metric == "relative_true_residual":
        return float(metrics["residual_norm"]) / rhs_norm
    if stopping_metric == "backward_error":
        return float(metrics["backward_error"])
    raise ValueError(f"Unsupported stopping_metric: {stopping_metric}")


@torch.inference_mode()
def tt_dgmres_sparse_diagnostics(
    *,
    operator: TTLinearOperator,
    rhs: TTTensor,
    config: TTDGMRESConfig,
    reference_solution: TTTensor | None = None,
    diagnostic_interval: int = 5,
    diagnostic_max_tt_rank: int | None = None,
    iteration_observer: Callable[[Dict[str, int]], None] | None = None,
    solution_projector: Callable[[TTTensor], TTTensor] | None = None,
) -> TTDGMRESResult:
    """Run TT-DGMRES while computing expensive diagnostics only every few steps."""

    if diagnostic_interval <= 0:
        raise ValueError("diagnostic_interval must be positive.")

    cfg = config.resolved()
    metric_max_rank = (
        cfg.max_tt_rank
        if diagnostic_max_tt_rank is None
        else int(diagnostic_max_tt_rank)
    )
    if metric_max_rank is not None and metric_max_rank < 1:
        raise ValueError("diagnostic_max_tt_rank must be positive when provided.")
    if (
        metric_max_rank is not None
        and cfg.max_tt_rank is not None
        and metric_max_rank < cfg.max_tt_rank
    ):
        raise ValueError(
            "diagnostic_max_tt_rank must be at least the algorithm max_tt_rank."
        )
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

    operator_norm = _resolve_operator_norm(operator, cfg)  # type: ignore[arg-type]
    rhs_norm = max(_stable_tt_norm(rhs), np.finfo(float).tiny)
    reference_norm = (
        _stable_tt_norm(reference_solution)
        if reference_solution is not None
        else None
    )
    x = tt_zeros(rhs.shape, like=rhs)
    if solution_projector is not None:
        x = solution_projector(x)
        require_problem_layout("solution_projector output", x)
    history = _empty_history()
    for field in (
        "algorithm_inner_products",
        "diagnostic_residual_pre_round_rank",
        "diagnostic_residual_rank",
        "diagnostic_transformed_pre_round_rank",
        "diagnostic_transformed_residual_rank",
        "diagnostic_residual_rank_cap_hit",
        "diagnostic_transformed_rank_cap_hit",
        "diagnostic_rank_cap_hit",
        "diagnostic_initial_rank_cap_hit",
        "diagnostic_metric_valid_for_stopping",
        "diagnostic_initial_transformed_residual_norm",
        "diagnostic_transformed_residual_certified",
        "diagnostic_rounded_transformed_residual_norm",
        "diagnostic_audit_transformed_pre_rank",
        "diagnostic_audit_transformed_rank",
        "cumulative_diagnostic_seconds",
    ):
        history[field] = []

    algorithm_inner_products = 0
    total_iterations = 0
    cycles = 0
    converged = False
    breakdown_terminated = False
    breakdown = False
    status = "iteration_limit"

    diagnostic_started = time.perf_counter()
    initial_metrics = _diagnostic_residual_metrics(
        operator=operator,
        rhs=rhs,
        x=x,
        index=cfg.index,
        residual_round_eps=cfg.residual_round_tolerance,
        transformed_round_eps=cfg.transformed_round_tolerance,
        operator_norm=operator_norm,
        rhs_norm=rhs_norm,
        max_rank=metric_max_rank,
        certify_transformed=cfg.stopping_metric
        in {"a_residual", "relative_a_residual"},
    )
    cumulative_diagnostic_seconds = time.perf_counter() - diagnostic_started
    initial_transformed_residual_norm = float(
        initial_metrics["transformed_residual_norm"]
    )
    initial_diagnostic_rank_cap_hit = bool(initial_metrics["rank_cap_hit"])
    initial_projected_transformed_residual_norm: float | None = None

    def append_diagnostic(
        *,
        metrics: Dict[str, float],
        x_eval: TTTensor,
        current_basis: List[TTTensor],
        iteration: int,
        cycle: int,
        projected_residual: float,
    ) -> bool:
        transformed_norm = float(metrics["transformed_residual_norm"])
        relative_transformed_norm = (
            0.0
            if initial_transformed_residual_norm == 0.0
            and transformed_norm == 0.0
            else (
                float("inf")
                if initial_transformed_residual_norm == 0.0
                else transformed_norm / initial_transformed_residual_norm
            )
        )
        metric_valid = _diagnostic_metric_is_valid(
            stopping_metric=cfg.stopping_metric,
            current_metrics=metrics,
            initial_metrics=initial_metrics,
        )
        _append_history(
            history=history,
            iteration=iteration,
            cycle=cycle,
            backward_eta=metrics["backward_error"],
            true_residual_norm=metrics["residual_norm"],
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
        history["algorithm_inner_products"].append(float(algorithm_inner_products))
        history["diagnostic_residual_pre_round_rank"].append(
            metrics["residual_pre_round_rank"]
        )
        history["diagnostic_residual_rank"].append(metrics["residual_rank"])
        history["diagnostic_transformed_pre_round_rank"].append(
            metrics["transformed_pre_round_rank"]
        )
        history["diagnostic_transformed_residual_rank"].append(
            metrics["transformed_rank"]
        )
        history["diagnostic_residual_rank_cap_hit"].append(
            metrics["residual_rank_cap_hit"]
        )
        history["diagnostic_transformed_rank_cap_hit"].append(
            metrics["transformed_rank_cap_hit"]
        )
        history["diagnostic_rank_cap_hit"].append(float(bool(metrics["rank_cap_hit"])))
        history["diagnostic_initial_rank_cap_hit"].append(
            float(initial_diagnostic_rank_cap_hit)
        )
        history["diagnostic_metric_valid_for_stopping"].append(float(metric_valid))
        history["diagnostic_initial_transformed_residual_norm"].append(
            initial_transformed_residual_norm
        )
        history["diagnostic_transformed_residual_certified"].append(
            metrics.get("transformed_residual_certified", 0.0)
        )
        history["diagnostic_rounded_transformed_residual_norm"].append(
            metrics.get(
                "rounded_transformed_residual_norm",
                metrics["transformed_residual_norm"],
            )
        )
        history["diagnostic_audit_transformed_pre_rank"].append(
            metrics.get(
                "audit_transformed_pre_rank",
                metrics["transformed_pre_round_rank"],
            )
        )
        history["diagnostic_audit_transformed_rank"].append(
            metrics.get("audit_transformed_rank", metrics["transformed_rank"])
        )
        history["cumulative_diagnostic_seconds"].append(
            float(cumulative_diagnostic_seconds)
        )
        return metric_valid

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
    initial_projected_value = _stable_tt_norm(initial_algorithm_transformed)
    initial_stop_value = _diagnostic_stop_value(
        stopping_metric=cfg.stopping_metric,
        metrics=initial_metrics,
        initial_transformed_residual_norm=initial_transformed_residual_norm,
        rhs_norm=rhs_norm,
    )
    initial_metric_valid = _diagnostic_metric_is_valid(
        stopping_metric=cfg.stopping_metric,
        current_metrics=initial_metrics,
        initial_metrics=initial_metrics,
    )
    if initial_stop_value < cfg.tolerance and initial_metric_valid:
        append_diagnostic(
            metrics=initial_metrics,
            x_eval=x,
            current_basis=[initial_algorithm_transformed],
            iteration=0,
            cycle=0,
            projected_residual=initial_projected_value,
        )
        return TTDGMRESResult(
            solution=x,
            converged=True,
            iterations=0,
            cycles=0,
            operator_norm=operator_norm,
            history=history,
            config=cfg,
            status="initial_residual_target",
            breakdown=False,
        )

    max_cycles = cfg.max_cycles
    if max_cycles is None:
        max_cycles = max(1, int(np.ceil(cfg.max_iterations / cfg.restart)))

    while (
        total_iterations < cfg.max_iterations
        and cycles < max_cycles
        and not converged
        and not breakdown_terminated
    ):
        cycles += 1
        cycle_start_x = x
        residual0 = tt_round_capped(
            rhs - operator.matvec(cycle_start_x),
            cfg.residual_round_tolerance,
            cfg.max_tt_rank,
        )
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
            diagnostic_step_started = time.perf_counter()
            restart_metrics = _diagnostic_residual_metrics(
                operator=operator,
                rhs=rhs,
                x=cycle_start_x,
                index=cfg.index,
                residual_round_eps=cfg.residual_round_tolerance,
                transformed_round_eps=cfg.transformed_round_tolerance,
                operator_norm=operator_norm,
                rhs_norm=rhs_norm,
                max_rank=metric_max_rank,
                certify_transformed=cfg.stopping_metric
                in {"a_residual", "relative_a_residual"},
            )
            cumulative_diagnostic_seconds += time.perf_counter() - diagnostic_step_started
            restart_metric_valid = append_diagnostic(
                metrics=restart_metrics,
                x_eval=cycle_start_x,
                current_basis=[transformed_residual0],
                iteration=total_iterations,
                cycle=cycles,
                projected_residual=0.0,
            )
            restart_stop_value = _diagnostic_stop_value(
                stopping_metric=cfg.stopping_metric,
                metrics=restart_metrics,
                initial_transformed_residual_norm=initial_transformed_residual_norm,
                rhs_norm=rhs_norm,
            )
            converged = bool(
                restart_stop_value < cfg.tolerance and restart_metric_valid
            )
            breakdown_terminated = not converged
            if converged:
                status = "residual_target_at_cycle_start"
            else:
                status = "zero_transformed_residual_without_target"
                breakdown = True
            break

        if initial_projected_transformed_residual_norm is None:
            initial_projected_transformed_residual_norm = beta

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

            w = tt_round_capped(
                operator.matvec(basis[local_k]),
                cfg.round_tolerance,
                cfg.max_tt_rank,
            )
            arnoldi_image_norm_tensor = tt_norm_stable_tensor(w)
            for i in range(local_k + 1):
                hij = tt_dot_tensor(basis[i], w)
                algorithm_inner_products += 1
                hbar[i, local_k] += hij
                w = w + tt_scale(basis[i], -hij)
                if cfg.safe_orth_rounding:
                    w = tt_round_capped(
                        w,
                        cfg.orth_round_tolerance,
                        cfg.max_tt_rank,
                    )

            if cfg.reorthogonalize:
                for i in range(local_k + 1):
                    correction_hij = tt_dot_tensor(basis[i], w)
                    algorithm_inner_products += 1
                    hbar[i, local_k] += correction_hij
                    w = w + tt_scale(basis[i], -correction_hij)
                    if cfg.safe_orth_rounding:
                        w = tt_round_capped(
                            w,
                            cfg.orth_round_tolerance,
                            cfg.max_tt_rank,
                        )

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
            # Keep the TT contractions and MGS coefficients on the problem
            # device, then synchronize exactly once for the small CPU
            # least-squares problem and breakdown decisions.
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
            has_breakdown = bool(
                h_next <= cfg.breakdown_tolerance * breakdown_scale
            )
            exact_breakdown = bool(
                has_breakdown and h_next == 0.0
            )
            if not has_breakdown:
                basis.append(tt_scale(w, torch.reciprocal(h_next_tensor)))

            projected_residual = beta
            xi: np.ndarray | None = None
            lucky_invariant_breakdown = bool(
                has_breakdown
                and k <= cfg.index
                and _arnoldi_invariance_is_certified(
                    operator,
                    basis,
                    hbar_cpu,
                    k,
                )
            )

            if k > cfg.index:
                hhat = build_hhat(hbar_cpu, k, cfg.index)
                e1 = np.zeros(k + 1, dtype=float)
                e1[0] = beta
                xi, *_ = np.linalg.lstsq(hhat, e1, rcond=None)
                projected_residual = float(np.linalg.norm(e1 - hhat @ xi))
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

            projected_relative = (
                projected_residual
                if initial_projected_transformed_residual_norm in (None, 0.0)
                else projected_residual / initial_projected_transformed_residual_norm
            )
            at_cycle_end = local_k == cfg.restart - 1
            at_max_iterations = total_iterations >= cfg.max_iterations
            at_diagnostic_step = total_iterations % diagnostic_interval == 0
            projected_converged = projected_relative < cfg.tolerance
            should_evaluate = bool(
                at_diagnostic_step
                or at_cycle_end
                or at_max_iterations
                or has_breakdown
                or projected_converged
            )

            if iteration_observer is not None:
                iteration_observer(
                    {
                        "iteration": int(total_iterations),
                        "cycle": int(cycles),
                        "local_k": int(local_k),
                        "perf_counter_ns": int(time.perf_counter_ns()),
                        "basis_tt_core_bytes": int(
                            sum(_tt_core_storage_bytes(vector) for vector in basis)
                        ),
                        "accepted_solution_tt_core_bytes": int(
                            _tt_core_storage_bytes(x_cycle)
                        ),
                        "last_vector_tt_core_bytes": int(
                            _tt_core_storage_bytes(basis[-1])
                        ),
                        "work_tt_core_bytes": int(_tt_core_storage_bytes(w)),
                        "hbar_bytes": int(hbar.numel())
                        * int(hbar.element_size()),
                    }
                )

            if should_evaluate:
                diagnostic_step_started = time.perf_counter()
                if k > cfg.index:
                    assert xi is not None
                    combination_vectors = basis[: k - cfg.index]
                elif lucky_invariant_breakdown:
                    assert xi is not None
                    combination_vectors = basis[:k]
                else:
                    combination_vectors = []

                if combination_vectors:
                    if cfg.tree_solution_combination:
                        correction = tt_linear_combination_capped_tree(
                            combination_vectors,
                            xi,
                            cfg.solution_round_tolerance,
                            cfg.max_tt_rank,
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
                else:
                    x_eval = cycle_start_x

                if solution_projector is not None:
                    x_eval = solution_projector(x_eval)
                    require_problem_layout("solution_projector output", x_eval)

                diagnostic_metrics = _diagnostic_residual_metrics(
                    operator=operator,
                    rhs=rhs,
                    x=x_eval,
                    index=cfg.index,
                    residual_round_eps=cfg.residual_round_tolerance,
                    transformed_round_eps=cfg.transformed_round_tolerance,
                    operator_norm=operator_norm,
                    rhs_norm=rhs_norm,
                    max_rank=metric_max_rank,
                    certify_transformed=cfg.stopping_metric
                    in {"a_residual", "relative_a_residual"},
                )
                cumulative_diagnostic_seconds += time.perf_counter() - diagnostic_step_started
                diagnostic_metric_valid = append_diagnostic(
                    metrics=diagnostic_metrics,
                    x_eval=x_eval,
                    current_basis=list(basis),
                    iteration=total_iterations,
                    cycle=cycles,
                    projected_residual=projected_residual,
                )

                if (
                    at_cycle_end
                    or at_max_iterations
                    or has_breakdown
                    or projected_converged
                ):
                    x_cycle = x_eval

                stop_value = _diagnostic_stop_value(
                    stopping_metric=cfg.stopping_metric,
                    metrics=diagnostic_metrics,
                    initial_transformed_residual_norm=initial_transformed_residual_norm,
                    rhs_norm=rhs_norm,
                )
                if stop_value < cfg.tolerance and diagnostic_metric_valid:
                    x_cycle = x_eval
                    x = x_eval
                    converged = True
                    if has_breakdown:
                        breakdown = True
                        status = (
                            "residual_target_at_breakdown"
                            if exact_breakdown
                            else "residual_target_at_numerical_breakdown"
                        )
                    else:
                        status = "residual_target"
                    break

            if has_breakdown:
                x = x_cycle
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


def run_fast_case(
    *,
    n: int = 12,
    dims: int = 7,
    restart: int = 20,
    max_iterations: int = 300,
    tolerance: float = 1e-8,
    round_tolerance: float = 1e-8,
    diagnostic_interval: int = 10,
    max_tt_rank: int | None = 120,
    safe_orth_rounding: bool = True,
    tree_solution_combination: bool = True,
    output_dir: Path | str = "outputs_tt_dgmres_seventh_n12_d7_safe",
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = build_neumann_poisson_problem(n=n, dims=dims, round_tolerance=1e-12)
    config = default_config(
        problem.operator_norm_lower_bound,
        restart=restart,
        max_iterations=max_iterations,
        tolerance=tolerance,
        round_tolerance=round_tolerance,
    )
    config.max_tt_rank = max_tt_rank
    config.safe_orth_rounding = safe_orth_rounding
    config.tree_solution_combination = tree_solution_combination

    start = time.perf_counter()
    result = tt_dgmres_sparse_diagnostics(
        operator=problem.operator,
        rhs=problem.rhs,
        config=config,
        reference_solution=problem.reference_solution,
        diagnostic_interval=diagnostic_interval,
        solution_projector=make_neumann_range_projector(n, dims),
    )
    elapsed_seconds = time.perf_counter() - start

    summary = summary_dict(result)
    summary.update(
        {
            "n": n,
            "dims": dims,
            "restart": restart,
            "diagnostic_interval": diagnostic_interval,
            "max_tt_rank": max_tt_rank,
            "safe_orth_rounding": safe_orth_rounding,
            "tree_solution_combination": tree_solution_combination,
            "projection_scans": 1,
            "history_points": len(result.history["iteration"]),
            "elapsed_seconds": elapsed_seconds,
        }
    )

    case_dir = output_dir / f"restart_{restart}"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "history.json").write_text(json.dumps(result.history, indent=2), encoding="utf-8")
    (case_dir / "summary.json").write_text(
        json.dumps({"config": asdict(config), "summary": summary}, indent=2),
        encoding="utf-8",
    )
    full_curve_path = save_single_run_full_curves(
        result.history,
        case_dir / f"restart_{restart}_full_curves.png",
        title=f"TT-DGMRES Full Curves, Example 5.1, n={n}, d={dims}, restart={restart}",
        rank_cap=max_tt_rank,
    )

    return {"result": result, "summary": summary, "case_dir": case_dir, "full_curve_path": full_curve_path}


def run_restart_sweep(
    restarts: Iterable[int] = (20,),
    *,
    n: int = 12,
    dims: int = 7,
    max_iterations: int = 300,
    tolerance: float = 1e-8,
    round_tolerance: float = 1e-8,
    diagnostic_interval: int = 10,
    max_tt_rank: int | None = 120,
    safe_orth_rounding: bool = True,
    tree_solution_combination: bool = True,
    output_dir: Path | str = "outputs_tt_dgmres_seventh_n12_d7_safe",
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for restart in restarts:
        artifact = run_fast_case(
            n=n,
            dims=dims,
            restart=restart,
            max_iterations=max_iterations,
            tolerance=tolerance,
            round_tolerance=round_tolerance,
            diagnostic_interval=diagnostic_interval,
            max_tt_rank=max_tt_rank,
            safe_orth_rounding=safe_orth_rounding,
            tree_solution_combination=tree_solution_combination,
            output_dir=output_dir,
        )
        rows.append(artifact["summary"])
        print(
            "restart={restart:2d}  converged={converged}  iterations={iterations:3d}  "
            "error={error:.3e}  time={time:.2f}s  full_curve={full_curve}".format(
                restart=restart,
                converged=artifact["summary"]["converged"],
                iterations=artifact["summary"]["iterations"],
                error=artifact["summary"]["final_relative_solution_error"],
                time=artifact["summary"]["elapsed_seconds"],
                full_curve=artifact["full_curve_path"],
            )
        )

    csv_path = output_dir / "restart_sweep_summary.csv"
    fieldnames = [
        "restart",
        "converged",
        "iterations",
        "cycles",
        "history_points",
        "final_relative_transformed_residual_norm",
        "final_backward_error",
        "final_relative_solution_error",
        "max_solution_rank",
        "max_last_vector_rank",
        "max_tt_rank",
        "safe_orth_rounding",
        "tree_solution_combination",
        "projection_scans",
        "elapsed_seconds",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    summary_path = output_dir / "restart_sweep_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    plot_path = save_restart_sweep_plot(rows, output_dir / "restart_sweep_comparison.png")

    return {"rows": rows, "csv_path": csv_path, "summary_path": summary_path, "plot_path": plot_path}


def save_restart_sweep_plot(rows: List[Dict[str, Any]], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    restarts = [int(row["restart"]) for row in rows]
    times = [float(row["elapsed_seconds"]) for row in rows]
    errors = [float(row["final_relative_solution_error"]) for row in rows]
    iterations = [int(row["iterations"]) for row in rows]
    ranks = [float(row["max_solution_rank"]) for row in rows]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    axes[0].plot(restarts, times, marker="o")
    axes[0].set_title("Elapsed Time")
    axes[0].set_xlabel("restart")
    axes[0].set_ylabel("seconds")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(restarts, iterations, marker="o")
    axes[1].set_title("Iterations")
    axes[1].set_xlabel("restart")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(restarts, errors, marker="o")
    axes[2].set_title("Relative Solution Error")
    axes[2].set_xlabel("restart")
    axes[2].grid(True, which="both", alpha=0.3)

    axes[3].plot(restarts, ranks, marker="o")
    axes[3].set_title("Max Solution TT-rank")
    axes[3].set_xlabel("restart")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Seventh Version Fast Diagnostics: Restart Sweep")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_single_run_full_curves(
    history: Dict[str, List[float]],
    output_path: Path | str,
    *,
    title: str,
    rank_cap: int | None = None,
) -> Path:
    """Save detailed convergence/rank curves for one TT-DGMRES run."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    iterations = history.get("iteration", [])
    if not iterations:
        raise ValueError("Cannot plot full curves because history is empty.")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()

    axes[0].semilogy(iterations, history["relative_transformed_residual_norm"], marker="o", linewidth=1.6)
    axes[0].set_title("Relative Transformed Residual")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel(r"$||A^a r_k|| / ||A^a r_0||$")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogy(iterations, history["relative_solution_error"], marker="o", linewidth=1.6)
    axes[1].set_title("Relative Solution Error")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(r"$||x_k-x_{ref}||/||x_{ref}||$")
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].semilogy(iterations, history["backward_error"], marker="o", linewidth=1.6)
    axes[2].set_title("Backward Error")
    axes[2].set_xlabel("iteration")
    axes[2].grid(True, which="both", alpha=0.3)

    axes[3].semilogy(iterations, history["projected_transformed_residual_norm"], marker="o", linewidth=1.6)
    axes[3].set_title("Projected Transformed Residual")
    axes[3].set_xlabel("iteration")
    axes[3].grid(True, which="both", alpha=0.3)

    axes[4].plot(iterations, history["solution_rank"], marker="o", linewidth=1.6, label="solution rank")
    axes[4].plot(iterations, history["last_vector_rank"], marker="x", linewidth=1.4, label="last Arnoldi vector rank")
    if rank_cap is not None:
        axes[4].axhline(rank_cap, color="red", linestyle="--", linewidth=1.0, label=f"rank cap = {rank_cap}")
    axes[4].set_title("TT-rank Evolution")
    axes[4].set_xlabel("iteration")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    axes[5].semilogy(iterations, history["basis_ratio"], marker="o", linewidth=1.6, label="basis storage ratio")
    axes[5].semilogy(iterations, history["last_vector_ratio"], marker="x", linewidth=1.4, label="last vector storage ratio")
    axes[5].set_title("TT Storage Ratio")
    axes[5].set_xlabel("iteration")
    axes[5].grid(True, which="both", alpha=0.3)
    axes[5].legend()

    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    run_restart_sweep()


if __name__ == "__main__":
    main()
