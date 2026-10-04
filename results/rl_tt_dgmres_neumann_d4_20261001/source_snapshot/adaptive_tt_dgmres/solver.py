"""Adaptive hooks around an isolated Drazin TT-DGMRES backend.

The archived baseline and its diagnostic implementation are left unchanged.
Policy callbacks never receive reference solution errors.
"""
from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from .backend.paper_tt_dgmres import (
    TTDGMRESConfig, TTDGMRESResult, _arnoldi_invariance_is_certified,
    _append_history, _empty_history, _relative_solution_error,
    _resolve_operator_norm, _stable_tt_norm, build_hhat,
    tt_norm_stable_tensor, tt_round_capped,
)
from .backend.paper_tt_gmres import (
    TTLinearOperator, TTTensor, tt_device, tt_dtype, tt_dot_tensor,
    tt_scalar, tt_scale, tt_zeros,
)
from .backend.tt_dgmres_seventh_fast_restart_sweep import (
    _diagnostic_residual_metrics, _diagnostic_metric_is_valid,
    _diagnostic_stop_value, _tt_core_storage_bytes,
)


def _validate_tolerance(name, value):
    if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(name + " must be finite and positive.")


@torch.inference_mode()
def tt_dgmres_adaptive(
    *,
    operator: TTLinearOperator,
    rhs: TTTensor,
    config: TTDGMRESConfig,
    reference_solution: TTTensor | None = None,
    diagnostic_interval: int = 5,
    diagnostic_max_tt_rank: int | None = None,
    diagnostic_residual_round_tolerance: float | None = None,
    diagnostic_transformed_round_tolerance: float | None = None,
    iteration_observer: Callable[[Dict[str, int]], None] | None = None,
    solution_projector: Callable[[TTTensor], TTTensor] | None = None,
    cycle_controller: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
    cycle_observer: Callable[[Dict[str, Any]], None] | None = None,
    algorithm_rounder: Callable[[TTTensor, float, int | None, str], TTTensor] | None = None,
    initial_solution: TTTensor | None = None,
) -> TTDGMRESResult:
    """Run Drazin TT-DGMRES with cycle controls and independent residual audits.

    Controllers see solver state but never reference solutions or reference errors.
    ``algorithm_rounder`` receives every algorithm truncation; diagnostics use
    the original independent backend and frozen diagnostic tolerances. Callback
    tensor arguments are read-only by contract: clone before modifying cores.
    A policy may update restart and the three named algorithm tolerances only.
    This hook is not a proof of convergence or a certified global error budget.
    """

    if diagnostic_interval <= 0:
        raise ValueError("diagnostic_interval must be positive.")

    cfg = config.resolved()
    diagnostic_residual_eps = (cfg.residual_round_tolerance
                               if diagnostic_residual_round_tolerance is None
                               else diagnostic_residual_round_tolerance)
    diagnostic_transformed_eps = (cfg.transformed_round_tolerance
                                  if diagnostic_transformed_round_tolerance is None
                                  else diagnostic_transformed_round_tolerance)
    diagnostic_solution_eps = cfg.solution_round_tolerance
    _validate_tolerance("diagnostic_residual_round_tolerance", diagnostic_residual_eps)
    _validate_tolerance("diagnostic_transformed_round_tolerance", diagnostic_transformed_eps)
    for field in ("round_tolerance", "orth_round_tolerance", "residual_round_tolerance",
                  "solution_round_tolerance", "transformed_round_tolerance"):
        _validate_tolerance(field, getattr(cfg, field))
    if cfg.max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if cfg.max_cycles is not None and cfg.max_cycles < 1:
        raise ValueError("max_cycles must be positive when provided.")

    def algorithm_round(tensor, eps, max_rank, phase):
        result = (tt_round_capped(tensor, eps, max_rank) if algorithm_rounder is None
                  else algorithm_rounder(tensor, eps, max_rank, phase))
        require_problem_layout("algorithm_rounder output", result)
        if tuple(result.shape) != tuple(tensor.shape):
            raise ValueError("algorithm_rounder must preserve the tensor shape.")
        return result

    def algorithm_power(operator, vector, power, eps, max_rank, phase):
        result = vector
        for _ in range(power):
            result = algorithm_round(operator.matvec(result), eps, max_rank, phase)
        return result

    def algorithm_combination(vectors, coeffs, eps, max_rank, tree):
        if not vectors:
            raise ValueError("Expected at least one TT vector.")
        if tree:
            active = [algorithm_round(tt_scale(v, c), eps, max_rank, "combination_leaf")
                      for c, v in zip(coeffs, vectors) if tt_scalar(c) != 0.0]
            if not active:
                return tt_zeros(vectors[0].shape, like=vectors[0])
            while len(active) > 1:
                next_level = []
                for i in range(0, len(active), 2):
                    next_level.append(active[i] if i + 1 == len(active) else
                                      algorithm_round(active[i] + active[i + 1], eps,
                                                      max_rank, "combination_tree"))
                active = next_level
            return active[0]
        value = tt_zeros(vectors[0].shape, like=vectors[0])
        for c, v in zip(coeffs, vectors):
            value = value + tt_scale(v, c)
        return algorithm_round(value, eps, max_rank, "combination")
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
    x = tt_zeros(rhs.shape, like=rhs) if initial_solution is None else initial_solution.clone()
    require_problem_layout("initial_solution", x)
    if tuple(x.shape) != tuple(rhs.shape):
        raise ValueError("initial_solution must have the same shape as rhs.")
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
        residual_round_eps=diagnostic_residual_eps,
        transformed_round_eps=diagnostic_transformed_eps,
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
                diagnostic_solution_eps,
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

    initial_algorithm_residual = algorithm_round(
        rhs - operator.matvec(x),
        cfg.residual_round_tolerance,
        cfg.max_tt_rank,
        "initial_residual",
    )
    initial_algorithm_transformed = algorithm_power(
        operator,
        initial_algorithm_residual,
        cfg.index,
        cfg.transformed_round_tolerance,
        cfg.max_tt_rank,
        "initial_power",
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
        max_cycles = cfg.max_iterations

    latest_metrics = dict(initial_metrics)
    latest_projected_residual = initial_projected_value

    def state_context():
        return {
            "x": x, "cycle": cycles, "total_iterations": total_iterations,
            "metrics": dict(latest_metrics),
            "projected_residual": float(latest_projected_residual),
            "index": cfg.index, "restart": cfg.restart,
            "round_tolerance": cfg.round_tolerance,
            "orth_round_tolerance": cfg.orth_round_tolerance,
            "solution_round_tolerance": cfg.solution_round_tolerance,
            "max_tt_rank": cfg.max_tt_rank, "max_iterations": cfg.max_iterations,
            "tolerance": cfg.tolerance, "rhs_norm": rhs_norm,
            "initial_transformed_residual_norm": initial_transformed_residual_norm,
            "stopping_metric": cfg.stopping_metric,
        }

    def finish_cycle(started, start_iteration, action):
        event = state_context()
        event.update({
            "cycle_iterations": total_iterations - start_iteration,
            "converged": converged, "status": status,
            "elapsed_seconds": time.perf_counter() - started,
            "action": dict(action),
        })
        if cycle_observer is not None:
            cycle_observer(event)

    while (
        total_iterations < cfg.max_iterations
        and cycles < max_cycles
        and not converged
        and not breakdown_terminated
    ):
        cycles += 1
        cycle_started = time.perf_counter()
        cycle_start_iteration = total_iterations
        action = {} if cycle_controller is None else cycle_controller(state_context())
        if not isinstance(action, dict):
            raise TypeError("cycle_controller must return a dict.")
        allowed = {"restart", "round_tolerance", "orth_round_tolerance", "solution_round_tolerance"}
        unknown = set(action) - allowed
        if unknown:
            raise ValueError("Unsupported cycle_controller action keys: " + str(sorted(unknown)))
        if "restart" in action:
            restart_value = action["restart"]
            if isinstance(restart_value, bool) or int(restart_value) != restart_value:
                raise ValueError("restart must be an integer strictly larger than index.")
            if int(restart_value) <= cfg.index:
                raise ValueError("restart must be strictly larger than index.")
            cfg.restart = int(restart_value)
        for name in allowed - {"restart"}:
            if name in action:
                _validate_tolerance(name, action[name])
                setattr(cfg, name, float(action[name]))
        cycle_restart = cfg.restart
        cycle_start_x = x
        residual0 = algorithm_round(
            rhs - operator.matvec(cycle_start_x),
            cfg.residual_round_tolerance,
            cfg.max_tt_rank,
            "cycle_residual",
        )
        transformed_residual0 = algorithm_power(
            operator,
            residual0,
            cfg.index,
            cfg.transformed_round_tolerance,
            cfg.max_tt_rank,
            "cycle_power",
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
                residual_round_eps=diagnostic_residual_eps,
                transformed_round_eps=diagnostic_transformed_eps,
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
            latest_metrics = dict(restart_metrics)
            latest_projected_residual = 0.0
            finish_cycle(cycle_started, cycle_start_iteration, action)
            break

        if initial_projected_transformed_residual_norm is None:
            initial_projected_transformed_residual_norm = beta

        basis: List[TTTensor] = [
            tt_scale(transformed_residual0, torch.reciprocal(beta_tensor))
        ]
        hbar = torch.zeros(
            (cycle_restart + 1, cycle_restart),
            device=rhs_device,
            dtype=rhs_dtype,
        )
        x_cycle = cycle_start_x

        for local_k in range(cycle_restart):
            if total_iterations >= cfg.max_iterations:
                break

            w = algorithm_round(
                operator.matvec(basis[local_k]),
                cfg.round_tolerance,
                cfg.max_tt_rank,
                "matvec",
            )
            arnoldi_image_norm_tensor = tt_norm_stable_tensor(w)
            for i in range(local_k + 1):
                hij = tt_dot_tensor(basis[i], w)
                algorithm_inner_products += 1
                hbar[i, local_k] += hij
                w = w + tt_scale(basis[i], -hij)
                if cfg.safe_orth_rounding:
                    w = algorithm_round(
                        w,
                        cfg.orth_round_tolerance,
                        cfg.max_tt_rank,
                        "orth",
                    )

            if cfg.reorthogonalize:
                for i in range(local_k + 1):
                    correction_hij = tt_dot_tensor(basis[i], w)
                    algorithm_inner_products += 1
                    hbar[i, local_k] += correction_hij
                    w = w + tt_scale(basis[i], -correction_hij)
                    if cfg.safe_orth_rounding:
                        w = algorithm_round(
                            w,
                            cfg.orth_round_tolerance,
                            cfg.max_tt_rank,
                            "orth",
                        )

            if not cfg.safe_orth_rounding:
                w = algorithm_round(
                    w,
                    cfg.orth_round_tolerance,
                    cfg.max_tt_rank,
                    "orth",
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
            at_cycle_end = local_k == cycle_restart - 1
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
                    correction = algorithm_combination(
                        combination_vectors, xi, cfg.solution_round_tolerance,
                        cfg.max_tt_rank, cfg.tree_solution_combination,
                    )
                    x_eval = algorithm_round(
                        cycle_start_x + correction,
                        cfg.solution_round_tolerance,
                        cfg.max_tt_rank,
                        "solution",
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
                    residual_round_eps=diagnostic_residual_eps,
                    transformed_round_eps=diagnostic_transformed_eps,
                    operator_norm=operator_norm,
                    rhs_norm=rhs_norm,
                    max_rank=metric_max_rank,
                    certify_transformed=cfg.stopping_metric
                    in {"a_residual", "relative_a_residual"},
                )
                cumulative_diagnostic_seconds += time.perf_counter() - diagnostic_step_started
                latest_metrics = dict(diagnostic_metrics)
                latest_projected_residual = projected_residual
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
        finish_cycle(cycle_started, cycle_start_iteration, action)

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

