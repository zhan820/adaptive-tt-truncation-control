"""Shared numerical engine for fixed and learned V3 policies.

Every solver call executes one restart cycle.  This makes a checkpoint a full
outer-cycle state rather than pretending that a saved solution contains the
hidden state of a multi-cycle solver invocation.  V2 numerical kernels and
safeguards are reused unchanged for every policy.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from adaptive_tt_dgmres.backend.paper_tt_dgmres import TTDGMRESConfig, _stable_tt_norm
from rl_tt_dgmres_v2.problems import audit_solution
from rl_tt_dgmres_v2.rounding import FastRounder, RoundFailure
from rl_tt_dgmres_v2.solver import tt_dgmres_adaptive
from rl_tt_dgmres_v2.runner import (
    ACTIONS, PROFILES, RECOVERY_ACTION, Action, EpisodeConfig,
    Controller as V2Controller, FEATURE_NAMES as V2_FEATURE_NAMES,
    RetryCycle, StopEpisode, heuristic_action,
)


FEATURE_NAMES = V2_FEATURE_NAMES + (
    "rhs_log_rank_min", "rhs_log_rank_mean", "rhs_log_rank_max",
    "rhs_log_rank_std", "rhs_log_core_bytes",
    "ab_log_rank_min", "ab_log_rank_mean", "ab_log_rank_max",
    "ab_log_rank_std", "ab_log_core_bytes", "log_rhs_norm",
    "log_ab_over_rhs_norm",
)
FEATURE_DIM = len(FEATURE_NAMES)
CHECKPOINT_VERSION = 1
NUMERICAL_ENGINE = "v2_kernels_v3_single_cycle_scheduler_shared_by_all_policies"


def _tensor_structure(tensor):
    ranks = np.asarray([int(c.shape[2]) for c in tensor.cores[:-1]] or [1], dtype=float)
    values = [float(ranks.min()), float(ranks.mean()), float(ranks.max()), float(ranks.std())]
    return [math.log2(1.0 + value) / 8 for value in values] + [
        math.log1p(sum(c.numel() * c.element_size() for c in tensor.cores)) / 20,
    ]


def _input_features(online):
    """Use only online input tensors; one additional matvec is timed equally."""
    ab = online["operator"].matvec(online["rhs"])
    rhs_norm = max(float(online["rhs_norm"]), 1e-300)
    ab_norm = max(float(online["ab_norm"]), 1e-300)
    values = _tensor_structure(online["rhs"]) + _tensor_structure(ab) + [
        math.log10(rhs_norm) / 12, math.log10(ab_norm / rhs_norm) / 12,
    ]
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite input feature.")
    return np.clip(values, -4, 4).astype(np.float32).tolist()


@dataclass
class CycleCheckpoint:
    """Training-only complete state at a pre-initialization cycle boundary.

    ``observation`` is filled after that cycle's common initialization.  TT
    tensors, problem identifiers and controller history are not policy inputs.
    A checkpoint resumes with its unspent time and iteration budgets.
    """

    x: object
    config: dict
    shape: tuple
    elapsed_seconds: float
    input_features: list
    controller_state: dict
    observation: list = field(default_factory=list)
    cycle_initialization_seconds: float = 0.0
    version: int = CHECKPOINT_VERSION

    @property
    def features(self):
        return self.observation


_STATE_FIELDS = (
    "last_transformed", "previous", "previous_action", "rank_cap", "iterations",
    "cycle_number", "peak_basis_bytes", "stagnation", "force_safe", "shield_count",
    "recoveries", "breakdown_restarts", "orth_passes", "reorth_steps",
    "direct_correction_matvecs",
)


class Controller(V2Controller):
    def __init__(self, online, config, policy, rounder, started, *,
                 initial_checkpoint=None, checkpoint_observer=None,
                 compute_policy_features=True):
        super().__init__(online, config, policy, rounder, started)
        self.compute_policy_features = bool(compute_policy_features)
        self.elapsed_before = 0.0
        self.initial_iterations = 0
        self.initial_cycles = 0
        self.checkpoint_observer = checkpoint_observer
        self.checkpoint_collection_seconds = 0.0
        self.initialization_seconds = 0.0
        self.pending_checkpoint = None
        self.attempt_started = started
        self.first_initialization_seconds = 0.0
        self.attempt_count = 0
        tic = time.perf_counter()
        if initial_checkpoint is None:
            self.input_features = (_input_features(online) if self.compute_policy_features
                                   else [0.0] * (FEATURE_DIM - len(V2_FEATURE_NAMES)))
            self.input_feature_seconds = (time.perf_counter() - tic
                                          if self.compute_policy_features else 0.0)
            self.restoration_seconds = 0.0
        else:
            self._restore(initial_checkpoint)
            self.input_feature_seconds = 0.0
            self.restoration_seconds = time.perf_counter() - tic

    def _restore(self, checkpoint):
        if not isinstance(checkpoint, CycleCheckpoint) or checkpoint.version != CHECKPOINT_VERSION:
            raise ValueError("Unsupported cycle checkpoint.")
        if checkpoint.config != asdict(self.config):
            raise ValueError("Checkpoint configuration must match exactly, including total budgets.")
        if tuple(self.online["rhs"].shape) != checkpoint.shape:
            raise ValueError("Checkpoint tensor shape does not match the problem.")
        if not 0 <= checkpoint.elapsed_seconds < self.config.seconds_budget:
            raise ValueError("Checkpoint has no remaining time budget.")
        if len(checkpoint.input_features) != FEATURE_DIM - len(V2_FEATURE_NAMES):
            raise ValueError("Checkpoint feature dimensions do not match.")
        self.last_x = checkpoint.x.clone()
        for name in _STATE_FIELDS:
            setattr(self, name, copy.deepcopy(checkpoint.controller_state[name]))
        self.input_features = list(checkpoint.input_features)
        self.elapsed_before = checkpoint.elapsed_seconds
        self.initial_iterations = self.iterations
        self.initial_cycles = self.cycle_number
        self.offset = self.iterations
        self.solver_iterations = 0

    def elapsed(self):
        return self.elapsed_before + time.perf_counter() - self.started

    def deadline(self):
        if self.elapsed() >= self.config.seconds_budget:
            self.status = "time_budget"
            raise StopEpisode(self.status)

    def observation(self, context):
        if not self.compute_policy_features:
            return [0.0] * FEATURE_DIM
        values = super().observation(context)
        values[15] = min(4.0, self.elapsed() / self.config.seconds_budget)
        return values + list(self.input_features)

    def begin_attempt(self):
        self.deadline()
        self.offset = self.iterations
        self.solver_iterations = 0
        if self.checkpoint_observer is not None:
            tic = time.perf_counter()
            elapsed_before_clone = self.elapsed()
            self.pending_checkpoint = CycleCheckpoint(
                x=self.last_x.clone(), config=asdict(self.config),
                shape=tuple(self.online["rhs"].shape), elapsed_seconds=elapsed_before_clone,
                input_features=list(self.input_features),
                controller_state={name: copy.deepcopy(getattr(self, name)) for name in _STATE_FIELDS},
            )
            self._exclude_collection_time(time.perf_counter() - tic)
        self.attempt_started = time.perf_counter()
        self.attempt_count += 1

    def _exclude_collection_time(self, elapsed):
        # Offline instrumentation is never enabled in timed policy comparisons.
        self.checkpoint_collection_seconds += elapsed
        self.started += elapsed
        self.last_end += elapsed

    def on_start(self, context):
        initialization = time.perf_counter() - self.attempt_started
        self.initialization_seconds += initialization
        if self.attempt_count == 1:
            self.first_initialization_seconds = initialization
        controls = super().on_start(context)
        if self.checkpoint_observer is not None:
            tic = time.perf_counter()
            self.pending_checkpoint.observation = list(self.last_observation)
            self.pending_checkpoint.cycle_initialization_seconds = initialization
            self.checkpoint_observer(self.pending_checkpoint)
            self.pending_checkpoint = None
            self._exclude_collection_time(time.perf_counter() - tic)
        return controls


def run_episode(problem, config, policy, *, evaluate_reference=False,
                return_solution=False, initial_checkpoint=None, checkpoint_observer=None,
                compute_policy_features=True):
    """Run a fresh solve or the remaining part of a checkpointed solve.

    ``wall_seconds`` is newly incurred solver time, including state restoration,
    all feature/inference work and the final audit.  ``total_elapsed_seconds``
    also includes the checkpoint's past time.  Offline checkpoint collection is
    excluded explicitly; no comparisons should enable ``checkpoint_observer``.
    """
    config.validate()
    if not compute_policy_features and (initial_checkpoint is not None or checkpoint_observer is not None):
        raise ValueError("Checkpoints require policy features; disable both checkpoint options for fixed deployment.")
    started = time.perf_counter()
    online = problem.online_view()
    rounder = FastRounder(audit_mode=config.rounding_audit)
    controller = Controller(
        online, config, policy, rounder, started,
        initial_checkpoint=initial_checkpoint, checkpoint_observer=checkpoint_observer,
        compute_policy_features=compute_policy_features,
    )
    error, history = None, []
    while controller.iterations < config.max_iterations:
        attempt_start_residual = controller.last_transformed
        remaining = config.max_iterations - controller.iterations
        if remaining < 2:
            break
        cfg = TTDGMRESConfig(
            index=1, restart=17, max_iterations=remaining, max_cycles=1,
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
            controller.begin_attempt()
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
                if controller.last_transformed < 0.99 * attempt_start_residual:
                    controller.breakdown_restarts.append({"iteration": controller.iterations,
                                                          "status": result.status,
                                                          "residual": controller.last_transformed})
                    continue
                if controller.recover("nonprogressing_breakdown"):
                    continue
            if (not result.converged and not result.breakdown and result.iterations > 0
                    and controller.iterations < config.max_iterations):
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
    wall_seconds = time.perf_counter() - controller.started
    total_elapsed_seconds = controller.elapsed_before + wall_seconds
    if total_elapsed_seconds > config.seconds_budget:
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
        "wall_seconds": wall_seconds, "total_elapsed_seconds": total_elapsed_seconds,
        "checkpoint_elapsed_seconds": controller.elapsed_before,
        "remaining_time_budget_at_start": config.seconds_budget - controller.elapsed_before,
        "iterations": controller.iterations,
        "iterations_this_run": controller.iterations - controller.initial_iterations,
        "cycles": len(controller.records), "total_cycles": controller.cycle_number,
        "eta": audit["eta"], "relative_transformed_residual": audit["relative_transformed_residual"],
        "relative_true_residual": audit["relative_true_residual"], "range_defect": audit["range_defect"],
        "forward_error": forward_error, "peak_basis_core_bytes": controller.peak_basis_bytes,
        "peak_round_rank": stats["max_rank"], "feature_seconds": controller.feature_seconds + controller.input_feature_seconds,
        "input_feature_seconds": controller.input_feature_seconds,
        "input_feature_extra_matvecs": int(compute_policy_features and initial_checkpoint is None),
        "feature_mode": "online_policy_features" if compute_policy_features else "disabled_fixed_deployment",
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
        "numerical_engine": NUMERICAL_ENGINE, "feature_dimension": FEATURE_DIM,
        "initialization_seconds": controller.initialization_seconds,
        "first_initialization_seconds": controller.first_initialization_seconds,
        "checkpoint_restoration_seconds": controller.restoration_seconds,
        "checkpoint_collection_seconds_excluded": controller.checkpoint_collection_seconds,
        "checkpoint_boundary": "before_shared_single_cycle_solver_initialization",
    }
    output = {"summary": summary, "cycles": controller.records, "transitions": controller.transitions,
              "recoveries": controller.recoveries, "round_calls": rounder.logs,
              "breakdown_restarts": controller.breakdown_restarts,
              "history": controller.audit_history, "solver_history": history}
    if return_solution:
        output["solution"] = controller.last_x
    return output


def rollout_from_checkpoint(problem, config, checkpoint, first_action, baseline_action):
    """One selected action followed by a fixed policy, with genuine continuation."""
    if not 0 <= first_action < len(ACTIONS) or not 0 <= baseline_action < len(ACTIONS):
        raise ValueError("Invalid rollout action.")
    calls = 0

    def policy(observation):
        nonlocal calls
        action = first_action if calls == 0 else baseline_action
        calls += 1
        return action

    result = run_episode(problem, config, policy, initial_checkpoint=checkpoint)
    summary = result["summary"]
    result["training_row"] = {
        "features": list(checkpoint.observation), "action": int(first_action),
        "remaining_wall": summary["wall_seconds"],
        "time_budget": config.seconds_budget,
        "remaining_time_budget": summary["remaining_time_budget_at_start"],
        "success": summary["accepted"],
        "executed_first_action": summary["executed_action_indices"][0]
                                 if summary["executed_action_indices"] else None,
        "baseline_action": int(baseline_action),
        "checkpoint_iteration": checkpoint.controller_state["iterations"],
        "checkpoint_elapsed_seconds": checkpoint.elapsed_seconds,
        "restoration_seconds": summary["checkpoint_restoration_seconds"],
        "first_initialization_seconds": summary["first_initialization_seconds"],
        "data_origin": "real_rollout", "continuation_policy": "baseline",
    }
    return result
