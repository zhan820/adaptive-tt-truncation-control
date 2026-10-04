from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch

from adaptive_tt_dgmres.backend.paper_tt_dgmres import TTDGMRESConfig, _stable_tt_norm
from adaptive_tt_dgmres.backend.paper_tt_gmres import tt_dot_tensor, tt_max_rank, tt_zeros
from adaptive_tt_dgmres.rounding import BudgetRoundError, BudgetRounder
from adaptive_tt_dgmres.solver import tt_dgmres_adaptive
from .problems import audit_solution


ACTIONS = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2)
FEATURE_NAMES = (
    "log10_eta_over_target", "log10_relative_transformed_residual",
    "previous_log_reduction", "log2_solution_rank", "log1p_solution_core_bytes",
    "previous_log10_delta", "previous_cycle_time_fraction", "iteration_fraction",
    "log10_checked_over_projected", "previous_max_round_relative_error",
    "previous_max_round_rank", "previous_basis_core_bytes", "tensor_order",
    "mode_size", "has_previous_cycle", "elapsed_time_fraction",
)


@dataclass(frozen=True)
class EpisodeConfig:
    update_dim: int = 12
    max_iterations: int = 156
    tolerance: float = 1e-6
    fixed_round_tolerance: float = 1e-10
    diagnostic_tolerance: float = 1e-12
    range_tolerance: float = 1e-9
    max_tt_rank: int = 96
    diagnostic_max_rank: int = 512
    seconds_budget: float = 20.0
    failure_penalty: float = 2.0
    scope: str = "matvec"

    def validate(self):
        if self.update_dim < 1 or self.max_iterations <= 1:
            raise ValueError("Require positive update dimension and at least two Arnoldi steps.")
        if self.scope not in {"matvec", "arnoldi"}:
            raise ValueError("scope must be matvec or arnoldi.")
        if self.max_tt_rank < 1 or self.diagnostic_max_rank < self.max_tt_rank:
            raise ValueError("Invalid rank limits.")
        for name in ("tolerance", "fixed_round_tolerance", "diagnostic_tolerance",
                     "range_tolerance", "seconds_budget", "failure_penalty"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive.")


class EpisodeStop(RuntimeError):
    pass


def _log(value):
    if not math.isfinite(float(value)):
        raise FloatingPointError("Nonfinite value before observation scaling.")
    return math.log10(max(1e-30, min(1e30, float(value))))


def _bytes(tensor):
    return sum(c.numel() * c.element_size() for c in tensor.cores)


class ConservativeRounder(BudgetRounder):
    """Keep a near-cancelled orthogonalization vector instead of compressing it.

    An identity operation is a valid zero-truncation-error choice. It avoids the
    old backend's cancellation-prone zero fallback without enlarging tolerances.
    The exception is allowed only when existing ranks already satisfy the cap,
    and is identical for learned and nonlearned policies.
    """

    def __call__(self, tensor, eps, max_rank=None, phase="round"):
        started = time.perf_counter()
        norm = _stable_tt_norm(tensor) if phase == "orth" else math.inf
        preserve = (phase == "orth" and (max_rank is None or tt_max_rank(tensor) <= max_rank)
                    and norm < 1e-10)
        inspection_seconds = time.perf_counter() - started
        if preserve:
            started = time.perf_counter()
            rounded = tensor.clone()
            ranks = [int(c.shape[-1]) for c in tensor.cores[:-1]]
            self.logs.append({
                "cycle": self.cycle, "phase": phase, "requested_eps": float(eps), "effective_eps": 0.0,
                "input_norm": norm, "actual_absolute_error": 0.0, "actual_relative_error": 0.0,
                "noise_slack_absolute": 0.0, "noise_slack_relative": 0.0,
                "rank_before": max(ranks + [1]), "rank_after": max(ranks + [1]),
                "ranks_before": ranks, "ranks_after": ranks,
                "rank_cap_hit": bool(max_rank is not None and max(ranks + [1]) >= max_rank),
                "within_requested": True, "within_effective": True, "within_budget": True,
                "measured_within_requested": True, "measured_within_effective": True,
                "audit_acceptance": "within_tolerance", "audit_method": "identity_core_copy_no_truncation",
                "status": "accepted", "round_seconds": time.perf_counter() - started,
                "audit_seconds": 0.0,
            })
        else:
            rounded = super().__call__(tensor, eps, max_rank, phase)
        record = self.logs[-1]
        record["preserved_near_cancellation"] = bool(preserve)
        record["near_cancellation_check_seconds"] = inspection_seconds
        return rounded


class CycleController:
    """Online scalar observations only; no reference solution is accepted."""

    def __init__(self, online, config, actions, policy, rounder, started):
        self.online, self.config = online, config
        self.actions, self.policy, self.rounder = tuple(actions), policy, rounder
        self.started = self.last_end = started
        self.last_x = tt_zeros(online["rhs"].shape, like=online["rhs"])
        self.records, self.transitions = [], []
        self.active = None
        self.previous = {}
        self.previous_delta = min(actions)
        self.iterations = 0
        self.peak_basis_bytes = 0
        self.cycle_basis_bytes = 0
        self.feature_seconds = self.inference_seconds = 0.0
        self.status = "iteration_limit"
        self.last_observation = None

    def _eta(self, context):
        norm_x = _stable_tt_norm(context["x"])
        return context["metrics"]["transformed_residual_norm"] / max(
            norm_x + self.online["ab_norm"], 1e-300)

    def _observation(self, context):
        started = time.perf_counter()
        metrics = context["metrics"]
        transformed = float(metrics["transformed_residual_norm"])
        x = context["x"]
        observation = [
            _log(self._eta(context) / self.config.tolerance) / 8,
            _log(transformed / self.online["ab_norm"]) / 16,
            self.previous.get("log_reduction", 0.0) / 8,
            math.log2(1 + tt_max_rank(x)) / 8,
            math.log1p(_bytes(x)) / 20,
            (_log(self.previous_delta) + 5) / 4,
            self.previous.get("seconds", 0.0) / self.config.seconds_budget,
            context["total_iterations"] / self.config.max_iterations,
            _log(transformed / max(context["projected_residual"], 1e-300)) / 12,
            (_log(self.previous.get("round_error", 1e-16)) + 8) / 8,
            math.log2(1 + self.previous.get("round_rank", 1)) / 8,
            math.log1p(self.previous.get("basis_bytes", 0)) / 20,
            len(x.shape) / 10,
            max(x.shape) / 32,
            float(bool(self.previous)),
            (time.perf_counter() - self.started) / self.config.seconds_budget,
        ]
        if not np.isfinite(observation).all():
            raise FloatingPointError("Nonfinite online observation.")
        self.feature_seconds += time.perf_counter() - started
        return np.clip(observation, -4, 4).astype(np.float32).tolist()

    def on_start(self, context):
        if time.perf_counter() - self.started >= self.config.seconds_budget:
            self.status = "time_budget"
            raise EpisodeStop(self.status)
        observation = self._observation(context)
        if self.transitions:
            self.transitions[-1]["next_state"] = list(observation)
            self.records[-1]["next_state"] = list(observation)
        started = time.perf_counter()
        action = int(self.policy(observation))
        self.inference_seconds += time.perf_counter() - started
        if action < 0 or action >= len(self.actions):
            raise ValueError("Policy returned an invalid action index.")
        self.active = {
            "state": observation, "action": action, "delta": self.actions[action],
            "cycle": context["cycle"], "iteration_start": context["total_iterations"],
            "transformed_start": float(context["metrics"]["transformed_residual_norm"]),
            "round_start": len(self.rounder.logs),
        }
        self.rounder.cycle = int(context["cycle"])
        self.cycle_basis_bytes = 0
        self.last_observation = observation
        changes = {"round_tolerance": self.actions[action]}
        if self.config.scope == "arnoldi":
            changes["orth_round_tolerance"] = self.actions[action]
        return changes

    def on_iteration(self, event):
        self.iterations = event["iteration"]
        self.cycle_basis_bytes = max(self.cycle_basis_bytes, event["basis_tt_core_bytes"])
        self.peak_basis_bytes = max(self.peak_basis_bytes, self.cycle_basis_bytes)
        if time.perf_counter() - self.started >= self.config.seconds_budget:
            self.status = "time_budget"
            raise EpisodeStop(self.status)

    def on_end(self, context):
        self.last_x = context["x"]
        self.iterations = context["total_iterations"]
        metrics = context["metrics"]
        eta = self._eta(context)
        range_defect = abs(float(tt_dot_tensor(self.online["q0"], self.last_x))) / max(
            _stable_tt_norm(self.last_x), 1.0)
        transformed = float(metrics["transformed_residual_norm"])
        valid = bool(metrics.get("transformed_residual_certified", False))
        logs = self.rounder.logs[self.active["round_start"]:]
        self.previous_delta = self.active["delta"]
        self.previous = {
            "log_reduction": math.log(max(self.active["transformed_start"], 1e-300) /
                                      max(transformed, 1e-300)),
            "seconds": time.perf_counter() - self.last_end,
            "round_error": max([r["actual_relative_error"] for r in logs] + [0]),
            "round_rank": max([r["rank_after"] for r in logs] + [1]),
            "basis_bytes": self.cycle_basis_bytes,
        }
        next_state = self._observation(context)
        ended = time.perf_counter()
        elapsed = ended - self.last_end
        row = {**self.active, "iteration_end": self.iterations, "seconds": elapsed,
               "eta": eta, "range_defect": range_defect,
               "relative_transformed_residual": transformed / self.online["ab_norm"],
               "independent_transformed_available": valid, "completed": True,
               "basis_core_bytes": self.cycle_basis_bytes,
               "solution_rank": tt_max_rank(self.last_x), "next_state": next_state}
        self.records.append(row)
        self.transitions.append({"state": row["state"], "action": row["action"],
                                 "reward": -elapsed / self.config.seconds_budget,
                                 "next_state": next_state, "done": False})
        self.active = None
        self.last_end, self.last_observation = ended, next_state
        if ended - self.started > self.config.seconds_budget:
            self.status = "time_budget"
            raise EpisodeStop(self.status)
        if not valid:
            self.status = "diagnostic_unavailable"
            raise EpisodeStop(self.status)
        if eta <= self.config.tolerance and range_defect <= self.config.range_tolerance:
            self.status = "eta_target"
            raise EpisodeStop(self.status)

    def finalize(self, wall_seconds, accepted):
        remainder = max(0.0, wall_seconds - sum(r["seconds"] for r in self.records))
        if self.active is not None:
            row = {**self.active, "iteration_end": self.iterations, "seconds": remainder,
                   "completed": False, "next_state": self.last_observation,
                   "basis_core_bytes": self.cycle_basis_bytes}
            self.records.append(row)
            self.transitions.append({"state": row["state"], "action": row["action"],
                                     "reward": -remainder / self.config.seconds_budget,
                                     "next_state": row["next_state"], "done": True})
        elif self.transitions:
            self.transitions[-1]["reward"] -= remainder / self.config.seconds_budget
            self.records[-1]["seconds"] += remainder
        if self.transitions:
            self.transitions[-1]["done"] = True
            if not accepted:
                self.transitions[-1]["reward"] -= self.config.failure_penalty


def run_episode(problem, config: EpisodeConfig, policy: Callable, actions=ACTIONS,
                *, evaluate_reference=True):
    config.validate()
    if not actions or any(not math.isfinite(a) or a <= 0 for a in actions):
        raise ValueError("Actions must be positive finite tolerances.")
    started = time.perf_counter()
    online = problem.online_view()
    rounder = ConservativeRounder(adaptive=False)
    controller = CycleController(online, config, actions, policy, rounder, started)
    cfg = TTDGMRESConfig(
        index=1, restart=config.update_dim + 1, max_iterations=config.max_iterations,
        max_cycles=config.max_iterations, tolerance=config.tolerance,
        round_tolerance=min(actions), orth_round_tolerance=config.fixed_round_tolerance,
        solution_round_tolerance=config.fixed_round_tolerance,
        residual_round_tolerance=config.fixed_round_tolerance,
        transformed_round_tolerance=config.fixed_round_tolerance,
        operator_norm=1.0, max_tt_rank=config.max_tt_rank,
        safe_orth_rounding=True, tree_solution_combination=True,
        reorthogonalize=True, stopping_metric="relative_a_residual",
    )
    result = None
    error = None
    try:
        result = tt_dgmres_adaptive(
            operator=online["operator"], rhs=online["rhs"], config=cfg,
            diagnostic_interval=config.max_iterations,
            diagnostic_max_tt_rank=config.diagnostic_max_rank,
            diagnostic_residual_round_tolerance=config.diagnostic_tolerance,
            diagnostic_transformed_round_tolerance=config.diagnostic_tolerance,
            solution_projector=online["projector"], cycle_controller=controller.on_start,
            cycle_observer=controller.on_end, iteration_observer=controller.on_iteration,
            algorithm_rounder=rounder,
        )
        controller.last_x, controller.iterations = result.solution, result.iterations
        controller.status = result.status
    except EpisodeStop:
        pass
    except (BudgetRoundError, FloatingPointError) as exc:
        controller.status = "rounding_rejected" if isinstance(exc, BudgetRoundError) else "nonfinite"
        error = str(exc)
    audit_started = time.perf_counter()
    audit = audit_solution(problem, controller.last_x, include_reference=False)
    audit_seconds = time.perf_counter() - audit_started
    wall_seconds = time.perf_counter() - started
    if wall_seconds > config.seconds_budget:
        controller.status = "time_budget"
    accepted = bool(audit["finite"] and audit["eta"] <= config.tolerance and
                    audit["range_defect"] <= config.range_tolerance and
                    controller.status not in {"time_budget", "rounding_rejected", "nonfinite",
                                              "diagnostic_unavailable"})
    controller.finalize(wall_seconds, accepted)
    forward_error = None
    if evaluate_reference:
        forward_error = _stable_tt_norm(controller.last_x - problem.reference_solution) / max(
            _stable_tt_norm(problem.reference_solution), 1e-300)
    stats = rounder.cumulative_stats()
    summary = {
        "problem": asdict(problem.spec), "config": asdict(config), "accepted": accepted,
        "status": "eta_target" if accepted else controller.status, "error": error,
        "wall_seconds": wall_seconds, "iterations": controller.iterations,
        "cycles": len(controller.records), "eta": audit["eta"],
        "relative_transformed_residual": audit["relative_transformed_residual"],
        "relative_true_residual": audit["relative_true_residual"],
        "range_defect": audit["range_defect"], "forward_error": forward_error,
        "peak_basis_core_bytes": controller.peak_basis_bytes,
        "peak_round_rank": stats["max_rank"], "feature_seconds": controller.feature_seconds,
        "inference_seconds": controller.inference_seconds, "final_audit_seconds": audit_seconds,
        "round_seconds": stats["round_seconds"], "round_audit_seconds": stats["audit_seconds"],
        "near_cancellation_check_seconds": sum(r.get("near_cancellation_check_seconds", 0.0)
                                               for r in rounder.logs),
        "preserved_near_cancellation_calls": sum(r.get("preserved_near_cancellation", False)
                                                 for r in rounder.logs),
        "reward_sum": sum(row["reward"] for row in controller.transitions),
        "actions": list(actions), "action_indices": [r["action"] for r in controller.records],
        "orthogonalization": "fixed_reorthogonalization_identical_for_all_policies",
        "acceptance": "independent_transformed_eta_and_range_float64_not_interval_certificate",
    }
    return {"summary": summary, "cycles": controller.records,
            "transitions": controller.transitions, "round_calls": rounder.logs,
            "history": result.history if result is not None else {}}
