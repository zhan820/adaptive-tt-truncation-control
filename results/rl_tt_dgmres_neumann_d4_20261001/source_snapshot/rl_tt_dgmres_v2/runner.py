from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np

from adaptive_tt_dgmres.backend.paper_tt_dgmres import TTDGMRESConfig, _stable_tt_norm
from adaptive_tt_dgmres.backend.paper_tt_gmres import tt_dot_tensor, tt_max_rank, tt_zeros
from .problems import audit_solution
from .rounding import FastRounder, RoundFailure
from .solver import tt_dgmres_adaptive


@dataclass(frozen=True)
class Action:
    update_dim: int
    matvec_eps: float
    orth_eps: float
    solution_eps: float


PROFILES = ((1e-8, 1e-10, 1e-10), (1e-5, 1e-8, 1e-8), (1e-3, 1e-6, 1e-7))
ACTIONS = tuple(Action(m, *p) for m in (8, 16, 24) for p in PROFILES)
RECOVERY_ACTION = Action(24, 1e-12, 1e-12, 1e-12)
FEATURE_NAMES = (
    "log_eta_over_target", "log_relative_transformed", "previous_log_reduction",
    "log_solution_rank", "log_solution_bytes", "previous_log_delta",
    "previous_cycle_time_fraction", "iteration_fraction", "log_residual_gap",
    "previous_round_error", "previous_round_rank", "previous_basis_bytes",
    "tensor_order", "mode_size", "has_previous_cycle", "elapsed_time_fraction",
    "previous_update_dim", "stagnation_count", "fallback_count", "rank_cap_fraction",
)


@dataclass(frozen=True)
class EpisodeConfig:
    max_iterations: int = 240
    tolerance: float = 1e-6
    fixed_round_tolerance: float = 1e-10
    diagnostic_tolerance: float = 1e-12
    range_tolerance: float = 1e-9
    max_tt_rank: int = 96
    recovery_max_tt_rank: int = 256
    diagnostic_max_rank: int = 512
    seconds_budget: float = 30.0
    failure_penalty: float = 5.0
    max_recoveries: int = 3
    reorthogonalization: str = "selective"
    orth_error_mode: str = "arnoldi_budget"
    rounding_audit: str = "tail"
    safety_filter: bool = True
    growth_limit: float = 1.5
    projected_check_factor: float = 10.0
    direct_breakdown_correction: bool = True

    def validate(self):
        if self.max_iterations < 2 or self.max_recoveries < 0:
            raise ValueError("Invalid iteration/recovery limit.")
        if not 1 <= self.max_tt_rank <= self.recovery_max_tt_rank <= self.diagnostic_max_rank:
            raise ValueError("Rank caps must be positive and nondecreasing.")
        for name in ("tolerance", "fixed_round_tolerance", "diagnostic_tolerance",
                     "range_tolerance", "seconds_budget", "failure_penalty", "growth_limit",
                     "projected_check_factor"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + " must be finite and positive.")
        if self.reorthogonalization not in {"selective", "always", "never"}:
            raise ValueError("Invalid orthogonalization policy.")
        if self.orth_error_mode not in {"relative", "arnoldi_budget"}:
            raise ValueError("Invalid orthogonalization error mode.")
        if self.rounding_audit not in {"tail", "full"}:
            raise ValueError("Invalid rounding audit mode.")


class StopEpisode(RuntimeError):
    pass


class RetryCycle(RuntimeError):
    pass


def _log(value):
    if not math.isfinite(float(value)):
        raise FloatingPointError("Nonfinite observation.")
    return math.log10(max(float(value), 1e-300))


def _bytes(tensor):
    return sum(c.numel() * c.element_size() for c in tensor.cores)


def heuristic_action(observation):
    # Only present/past online measurements; no problem family or reference data.
    profile = 1 if observation[0] > 0.30 else 0
    length = 2 if observation[2] < 0.12 and observation[14] else 1
    return 3 * length + profile


class Controller:
    def __init__(self, online, config, policy, rounder, started):
        self.online, self.config, self.policy = online, config, policy
        self.rounder, self.started, self.last_end = rounder, started, started
        self.last_x = tt_zeros(online["rhs"].shape, like=online["rhs"])
        self.records, self.transitions, self.recoveries = [], [], []
        self.audit_history = []
        self.breakdown_restarts = []
        self.active = None
        self.previous = {}
        self.last_observation = None
        self.last_transformed = online["ab_norm"]
        self.previous_action = ACTIONS[3]
        self.rank_cap = config.max_tt_rank
        self.iterations = self.offset = self.solver_iterations = self.cycle_number = 0
        self.peak_basis_bytes = self.cycle_basis_bytes = 0
        self.stagnation = self.force_safe = self.shield_count = 0
        self.feature_seconds = self.inference_seconds = 0.0
        self.status = "iteration_limit"
        self.orth_passes = self.reorth_steps = 0
        self.direct_correction_matvecs = 0

    def eta(self, context):
        return context["metrics"]["transformed_residual_norm"] / max(
            _stable_tt_norm(context["x"]) + self.online["ab_norm"], 1e-300)

    def observation(self, context):
        started = time.perf_counter()
        transformed = context["metrics"]["transformed_residual_norm"]
        x = context["x"]
        values = [
            _log(self.eta(context) / self.config.tolerance) / 8,
            _log(transformed / max(self.online["ab_norm"], 1e-300)) / 16,
            self.previous.get("log_reduction", 0.0) / 8,
            math.log2(1 + tt_max_rank(x)) / 8, math.log1p(_bytes(x)) / 20,
            (_log(self.previous_action.matvec_eps) + 5) / 4,
            self.previous.get("seconds", 0.0) / self.config.seconds_budget,
            self.iterations / self.config.max_iterations,
            _log(transformed / max(context["projected_residual"], 1e-300)) / 12,
            (_log(self.previous.get("round_error", 1e-16)) + 8) / 8,
            math.log2(1 + self.previous.get("round_rank", 1)) / 8,
            math.log1p(self.previous.get("basis_bytes", 0)) / 20,
            len(x.shape) / 10, max(x.shape) / 32, float(bool(self.previous)),
            (time.perf_counter() - self.started) / self.config.seconds_budget,
            self.previous_action.update_dim / 24, min(self.stagnation, 4) / 4,
            len(self.recoveries) / max(self.config.max_recoveries, 1),
            self.rank_cap / self.config.recovery_max_tt_rank,
        ]
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite policy state.")
        self.feature_seconds += time.perf_counter() - started
        return np.clip(values, -4, 4).astype(np.float32).tolist()

    def deadline(self):
        if time.perf_counter() - self.started >= self.config.seconds_budget:
            self.status = "time_budget"
            raise StopEpisode(self.status)

    def on_start(self, context):
        self.deadline()
        self.iterations = self.offset + context["total_iterations"]
        obs = self.observation(context)
        if self.transitions:
            self.transitions[-1]["next_state"] = list(obs)
            self.records[-1]["next_state"] = list(obs)
        tic = time.perf_counter()
        proposed = int(self.policy(obs))
        self.inference_seconds += time.perf_counter() - tic
        if not 0 <= proposed < len(ACTIONS):
            raise ValueError("Invalid action index.")
        executed = proposed
        if self.force_safe:
            executed = -1
        elif self.config.safety_filter:
            eta_ratio = self.eta(context) / self.config.tolerance
            if eta_ratio < 10 or self.stagnation >= 2:
                executed = 3 * (proposed // 3)
            elif eta_ratio < 1000 and proposed % 3 == 2:
                executed = proposed - 1
        self.shield_count += int(executed != proposed)
        action = RECOVERY_ACTION if executed == -1 else ACTIONS[executed]
        self.previous_action = action
        self.cycle_number += 1
        self.rounder.cycle = self.cycle_number
        self.cycle_basis_bytes = 0
        self.active = {
            "cycle": self.cycle_number, "state": obs, "action": proposed,
            "executed_action": executed, "controls": asdict(action),
            "rank_cap": self.rank_cap, "iteration_start": self.iterations,
            "residual_eps": min(self.config.fixed_round_tolerance, action.matvec_eps)
                            if self.force_safe else self.config.fixed_round_tolerance,
            "transformed_eps": min(self.config.fixed_round_tolerance, action.matvec_eps)
                               if self.force_safe else self.config.fixed_round_tolerance,
            "transformed_start": context["metrics"]["transformed_residual_norm"],
            "round_start": len(self.rounder.logs),
        }
        self.last_observation = obs
        return {"restart": action.update_dim + 1, "round_tolerance": action.matvec_eps,
                "orth_round_tolerance": action.orth_eps,
                "solution_round_tolerance": action.solution_eps}

    def on_iteration(self, event):
        self.solver_iterations = event["iteration"]
        self.iterations = self.offset + self.solver_iterations
        self.cycle_basis_bytes = max(self.cycle_basis_bytes, event["basis_tt_core_bytes"])
        self.peak_basis_bytes = max(self.peak_basis_bytes, self.cycle_basis_bytes)
        self.orth_passes += event.get("orthogonalization_passes", 0)
        self.reorth_steps += int(event.get("reorthogonalized", False))
        self.direct_correction_matvecs += int(event.get("direct_breakdown_extra_matvecs", 0))
        self.deadline()

    def record(self, completed, next_state, **extra):
        if self.active is None:
            return
        ended = time.perf_counter()
        elapsed = ended - self.last_end
        row = {**self.active, "iteration_end": self.iterations, "seconds": elapsed,
               "completed": completed, "next_state": list(next_state),
               "basis_core_bytes": self.cycle_basis_bytes, **extra}
        self.records.append(row)
        self.transitions.append({"state": row["state"], "action": row["action"],
                                 "reward": -elapsed / self.config.seconds_budget,
                                 "next_state": list(next_state), "done": False})
        self.active = None
        self.last_end = ended

    def on_end(self, context):
        self.iterations = self.offset + context["total_iterations"]
        transformed = float(context["metrics"]["transformed_residual_norm"])
        valid = bool(context["metrics"].get("transformed_residual_certified", False))
        self.audit_history.append({"iteration": self.iterations,
                                   "cycle": self.cycle_number,
                                   "metrics": dict(context["metrics"]),
                                   "projected_residual": context["projected_residual"]})
        if not valid or not math.isfinite(transformed):
            raise RetryCycle("diagnostic_unavailable")
        initial = max(self.active["transformed_start"], 1e-300)
        if transformed > self.config.growth_limit * initial:
            raise RetryCycle("residual_growth")
        x = context["x"]
        eta = self.eta(context)
        range_defect = abs(float(tt_dot_tensor(self.online["q0"], x))) / max(_stable_tt_norm(x), 1.0)
        if not math.isfinite(range_defect) or range_defect > self.config.range_tolerance:
            raise RetryCycle("range_violation")
        logs = self.rounder.logs[self.active["round_start"]:]
        ratio = transformed / initial
        self.stagnation = self.stagnation + 1 if ratio > 0.95 else 0
        self.previous = {
            "log_reduction": math.log(initial / max(transformed, 1e-300)),
            "seconds": time.perf_counter() - self.last_end,
            "round_error": max([(r.get("actual_relative_error")
                                  if r.get("actual_relative_error") is not None
                                  else r.get("tail_relative_error", 0.0)) for r in logs] + [0]),
            "round_rank": max([r.get("rank_after", 1) for r in logs] + [1]),
            "basis_bytes": self.cycle_basis_bytes,
        }
        # Commit only a checked cycle. Rejected cycles cannot overwrite this checkpoint.
        self.last_x, self.last_transformed = x, transformed
        next_state = self.observation(context)
        self.record(True, next_state, eta=eta, range_defect=range_defect,
                    relative_transformed_residual=transformed / max(self.online["ab_norm"], 1e-300))
        self.last_observation = next_state
        self.deadline()
        if eta <= self.config.tolerance:
            self.status = "eta_target"
            raise StopEpisode(self.status)

    def on_candidate(self, context):
        # A loose projected trigger only schedules an independent check. It never
        # changes the final target or accepts a projected residual as the true one.
        if (self.active is not None and context.get("metric_valid_for_stopping", False)
                and self.eta(context) <= self.config.tolerance):
            self.on_end(context)

    def recover(self, reason):
        self.record(False, self.last_observation, rejection=reason) if self.active else None
        if len(self.recoveries) >= self.config.max_recoveries:
            return False
        previous_cap = self.rank_cap
        self.rank_cap = min(self.config.recovery_max_tt_rank, max(self.rank_cap + 16, 2 * self.rank_cap))
        self.force_safe = 1
        self.recoveries.append({"reason": reason, "iteration": self.iterations,
                                "previous_rank_cap": previous_cap, "rank_cap": self.rank_cap,
                                "checkpoint": "last_checked_cycle", "fallback_action": -1,
                                "fallback_controls": asdict(RECOVERY_ACTION)})
        self.offset = self.iterations
        self.solver_iterations = 0
        return True

    def finalize(self, wall_seconds, accepted):
        if self.active:
            self.record(False, self.last_observation, rejection=self.status)
        if self.transitions:
            remainder = wall_seconds - sum(r["seconds"] for r in self.records)
            self.records[-1]["seconds"] += remainder
            self.transitions[-1]["reward"] -= remainder / self.config.seconds_budget
            self.transitions[-1]["done"] = True
            if not accepted:
                self.transitions[-1]["reward"] -= self.config.failure_penalty


def run_episode(problem, config, policy, *, evaluate_reference=True):
    config.validate()
    started = time.perf_counter()
    online = problem.online_view()
    rounder = FastRounder(audit_mode=config.rounding_audit)
    controller = Controller(online, config, policy, rounder, started)
    error, history = None, []
    while controller.iterations < config.max_iterations:
        attempt_start_residual = controller.last_transformed
        remaining = config.max_iterations - controller.iterations
        if remaining < 2:
            break
        cfg = TTDGMRESConfig(
            index=1, restart=17, max_iterations=remaining, max_cycles=remaining,
            tolerance=config.tolerance, round_tolerance=1e-8,
            orth_round_tolerance=config.fixed_round_tolerance,
            solution_round_tolerance=config.fixed_round_tolerance,
            residual_round_tolerance=min(config.fixed_round_tolerance, RECOVERY_ACTION.matvec_eps)
                                     if controller.force_safe else config.fixed_round_tolerance,
            transformed_round_tolerance=min(config.fixed_round_tolerance, RECOVERY_ACTION.matvec_eps)
                                        if controller.force_safe else config.fixed_round_tolerance,
            operator_norm=1.0, max_tt_rank=controller.rank_cap, safe_orth_rounding=True,
            tree_solution_combination=True, reorthogonalize=True, stopping_metric="relative_a_residual",
        )
        try:
            controller.deadline()
            result = tt_dgmres_adaptive(
                operator=online["operator"], rhs=online["rhs"], config=cfg,
                diagnostic_interval=config.max_iterations,
                diagnostic_max_tt_rank=config.diagnostic_max_rank,
                diagnostic_residual_round_tolerance=config.diagnostic_tolerance,
                diagnostic_transformed_round_tolerance=config.diagnostic_tolerance,
                solution_projector=online["projector"], cycle_controller=controller.on_start,
                cycle_observer=controller.on_end, iteration_observer=controller.on_iteration,
                algorithm_rounder=rounder, initial_solution=controller.last_x,
                reorthogonalization=config.reorthogonalization,
                candidate_observer=controller.on_candidate,
                projected_check_factor=config.projected_check_factor,
                orth_error_mode=config.orth_error_mode,
                direct_breakdown_correction=config.direct_breakdown_correction,
            )
            controller.last_x = result.solution
            controller.iterations = controller.offset + result.iterations
            controller.status = result.status
            history.append(result.history)
            if (not result.converged and result.breakdown and result.iterations > 0
                    and controller.iterations < config.max_iterations):
                # A checked, improving partial DGMRES correction at an invariant
                # space is a normal restart, not a failed truncation. It must not
                # spend the rank-cap recovery budget or enlarge ranks needlessly.
                if controller.last_transformed < 0.99 * attempt_start_residual:
                    controller.breakdown_restarts.append({"iteration": controller.iterations,
                                                          "status": result.status,
                                                          "residual": controller.last_transformed})
                    controller.offset = controller.iterations
                    controller.solver_iterations = 0
                    continue
                if controller.recover("nonprogressing_breakdown"):
                    continue
            break
        except StopEpisode:
            break
        except (RoundFailure, RetryCycle, FloatingPointError) as exc:
            error = str(exc)
            controller.status = "rounding_rejected" if isinstance(exc, RoundFailure) else "cycle_rejected"
            if not controller.recover(error):
                break
    audit_started = time.perf_counter()
    audit = audit_solution(problem, controller.last_x, include_reference=False)
    audit_seconds = time.perf_counter() - audit_started
    wall_seconds = time.perf_counter() - started
    if wall_seconds > config.seconds_budget:
        controller.status = "time_budget"
    accepted = bool(audit["finite"] and audit["eta"] <= config.tolerance and
                    audit["range_defect"] <= config.range_tolerance and
                    controller.status != "time_budget")
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
        "relative_true_residual": audit["relative_true_residual"], "range_defect": audit["range_defect"],
        "forward_error": forward_error, "peak_basis_core_bytes": controller.peak_basis_bytes,
        "peak_round_rank": stats["max_rank"], "feature_seconds": controller.feature_seconds,
        "inference_seconds": controller.inference_seconds, "final_audit_seconds": audit_seconds,
        "round_seconds": stats["round_seconds"], "round_audit_seconds": stats["audit_seconds"],
        "reward_sum": sum(t["reward"] for t in controller.transitions),
        "action_indices": [r["action"] for r in controller.records],
        "executed_action_indices": [r["executed_action"] for r in controller.records],
        "recoveries": len(controller.recoveries), "shield_interventions": controller.shield_count,
        "breakdown_restarts": len(controller.breakdown_restarts),
        "recovery_policy": "switch_to_conservative_action_for_remaining_solve",
        "orthogonalization_passes": controller.orth_passes, "reorthogonalized_steps": controller.reorth_steps,
        "direct_correction_matvecs": controller.direct_correction_matvecs,
        "acceptance": "independent_untruncated_float64_transformed_eta_and_range_not_interval_bound",
    }
    return {"summary": summary, "cycles": controller.records, "transitions": controller.transitions,
            "recoveries": controller.recoveries, "round_calls": rounder.logs,
            "breakdown_restarts": controller.breakdown_restarts,
            "history": controller.audit_history, "solver_history": history}
