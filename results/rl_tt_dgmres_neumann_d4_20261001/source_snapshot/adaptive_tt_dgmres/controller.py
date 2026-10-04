from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .backend.paper_tt_dgmres import _stable_tt_norm
from .backend.paper_tt_gmres import tt_max_rank
from .features import tt_interface_tokens
from .model import tensorize_histories
from .rounding import BudgetRounder


GLOBAL_FEATURE_NAMES = [
    "log10_relative_true_residual", "log10_relative_transformed_residual",
    "log10_relative_projected_residual", "log10_checked_projected_ratio",
    "previous_log_reduction", "tensor_order_over_10", "log2_solution_rank_over_8",
    "log1p_core_bytes_over_20", "previous_restart_over_32", "log10_previous_alpha",
    "log10_target_over_12", "iteration_budget_fraction", "cycle_over_30",
    "log10_previous_seconds_over_6", "previous_local_budget_fraction",
    "previous_rank_cap_hit", "index_over_4", "log10_operator_scale_over_8",
]


@dataclass(frozen=True)
class Action:
    restart: int
    alpha: float

    def encoded(self) -> tuple[float, float]:
        return self.restart / 32.0, math.log10(self.alpha)


def candidate_actions() -> list[Action]:
    return [Action(m, alpha) for m in (4, 8, 12) for alpha in (0.1, 1.0)]


def _log(value: float) -> float:
    return math.log10(max(1e-16, min(1e16, float(value))))


class SolverController:
    """Collect online-only features and select joint actions at cycle boundaries.

    Reference solutions and their errors never enter this class. The initial
    prototype's local budget is explicitly empirical. Residual acceptance is
    performed by the numerical solver and a separate final range audit.
    """

    def __init__(self, *, operator: Any, rhs: Any, operator_scale: float,
                 rounder: BudgetRounder, base_eps: float, actions: list[Action] | None = None,
                 mode: str = "heuristic", model: Any = None, fixed_action: Action | None = None,
                 first_action: Action | None = None, prefix: list[dict] | None = None,
                 snapshots: bool = False, first_frame: dict | None = None,
                 progress_guard: bool = True):
        if mode not in {"heuristic", "neural", "fixed", "scripted", "teacher"}:
            raise ValueError("Unknown controller mode.")
        self.operator, self.rhs = operator, rhs
        self.operator_scale = operator_scale
        self.rounder, self.base_eps = rounder, base_eps
        self.actions = candidate_actions() if actions is None else actions
        if not self.actions or any(a.restart <= 1 or not 0 < a.alpha <= 1 for a in self.actions):
            raise ValueError("Invalid candidate actions for this index-one prototype.")
        self.mode, self.model = mode, model
        self.fixed_action = fixed_action or Action(12, 0.1)
        self.first_action = first_action
        self.first_frame = copy.deepcopy(first_frame)
        self.frames = copy.deepcopy(prefix or [])
        self.records: list[dict] = []
        self.saved_states: list[dict] = []
        self.capture_snapshots = snapshots
        self.previous: dict = {}
        self.previous_action = self.fixed_action
        self.feature_seconds = 0.0
        self.inference_seconds = 0.0
        self.current: dict = {}
        self.progress_guard = progress_guard
        self.guard_latched = False
        self.guard_activations = 0

    def _guard(self, context: dict, proposed: Action) -> tuple[Action, str]:
        """Empirical progress fallback, independent of the stopping criterion.

        A two-cycle contraction estimate detects a proposal sequence likely to
        exhaust the remaining iteration allowance. A latched longer restart
        avoids oscillating back to the same poorly trained proposal. This is
        a control heuristic, not a proof that restarted DGMRES converges.
        """
        if not self.progress_guard or self.mode != "neural":
            return proposed, self.mode
        if len(self.records) >= 2 and not self.guard_latched:
            recent = self.records[-2:]
            used = sum(r["iteration_end"] - r["iteration_start"] for r in recent)
            progress = sum(r["log_reduction"] for r in recent)
            rate = progress / max(used, 1)
            relative = context["metrics"]["residual_norm"] / context["rhs_norm"]
            needed = math.log(max(relative / context["tolerance"], 1.0))
            remaining = context["max_iterations"] - context["total_iterations"]
            if rate <= 0 or 0.8 * remaining * rate < needed:
                self.guard_latched = True
                self.guard_activations += 1
        if self.guard_latched:
            # Largest available Krylov space and smallest local-error fraction.
            # Final acceptance still requires the same independently measured residual.
            safest = min(self.actions, key=lambda a: (-a.restart, a.alpha))
            return safest, "measured_progress_fallback"
        return proposed, "neural"

    def _frame(self, context: dict) -> dict:
        metrics = context["metrics"]
        rel = float(metrics["residual_norm"]) / max(context["rhs_norm"], 1e-300)
        initial_t = max(context["initial_transformed_residual_norm"], 1e-300)
        transformed = float(metrics["transformed_residual_norm"])
        projected = max(float(context["projected_residual"]), 1e-300)
        x = context["x"]
        # Spectrum acquisition is on a feature-only tensor, and its full cost
        # is included in solver wall time. No dense tensor is materialized.
        residual = self.rhs - self.operator.matvec(x)
        tokens = tt_interface_tokens(residual)
        core_bytes = sum(c.numel() * c.element_size() for c in x.cores)
        previous_budget = self.previous.get("rounding", {})
        spent = float(previous_budget.get("actual_relative_error_sum", 0.0))
        limit = float(previous_budget.get("budget_total", 1.0))
        global_values = [
            _log(rel), _log(transformed / initial_t), _log(projected / initial_t),
            float(np.clip(_log(transformed / projected), -12, 12)),
            float(self.previous.get("log_reduction", 0.0)), len(x.shape) / 10.0,
            math.log2(1 + tt_max_rank(x)) / 8.0, math.log1p(core_bytes) / 20.0,
            self.previous_action.restart / 32.0, _log(self.previous_action.alpha),
            _log(context["tolerance"]) / 12.0,
            context["total_iterations"] / max(context["max_iterations"], 1),
            context["cycle"] / 30.0, _log(self.previous.get("wall_seconds", 1.0)) / 6.0,
            min(spent / max(limit, 1e-300), 2.0),
            float(bool(previous_budget.get("rank_cap_hits", 0))),
            context["index"] / 4.0, _log(self.operator_scale) / 8.0,
        ]
        if len(global_values) != 18 or not np.isfinite(global_values).all():
            raise ValueError("Nonfinite or malformed controller state.")
        return {"tokens": tokens, "global": global_values}

    def _choose(self, context: dict) -> Action:
        if self.first_action is not None and not self.records:
            return self.first_action
        if self.mode in {"fixed", "teacher"}:
            return self.fixed_action
        if self.mode == "scripted":
            sequence = [Action(4, 1.0), Action(8, 0.1), Action(12, 1.0)]
            return sequence[len(self.records) % len(sequence)]
        if self.mode == "heuristic":
            reduction = self.previous.get("log_reduction", 1.0)
            relative = context["metrics"]["residual_norm"] / context["rhs_norm"]
            restart = 12 if reduction < 0.6 else 8
            return Action(restart, 0.1 if relative < 1e-3 else 1.0)
        if self.model is None:
            raise ValueError("Neural control requires a trained checkpoint.")
        inputs = tensorize_histories([self.frames[-5:]], [a.encoded() for a in self.actions])
        started = time.perf_counter()
        with torch.no_grad():
            scores = self.model(**inputs)["log_cost"][0]
        self.inference_seconds += time.perf_counter() - started
        if not torch.isfinite(scores).all():
            raise ValueError("Controller produced nonfinite action scores.")
        return self.actions[int(torch.argmin(scores).item())]

    def on_cycle_start(self, context: dict) -> dict:
        started = time.perf_counter()
        frame = (copy.deepcopy(self.first_frame) if self.first_frame is not None and not self.records
                 else self._frame(context))
        self.feature_seconds += time.perf_counter() - started
        self.frames.append(frame)
        if self.capture_snapshots and len(self.records) in {0, 2}:
            self.saved_states.append({
                "solution": context["x"].clone(), "history": copy.deepcopy(self.frames[-5:]),
                "source_cycle": context["cycle"],
                "source_iteration": context["total_iterations"],
            })
        proposed = self._choose(context)
        action, source = self._guard(context, proposed)
        self.rounder.start_cycle(action.restart, action.alpha, self.base_eps, context["index"])
        self.current = {
            "cycle": context["cycle"], "iteration_start": context["total_iterations"],
            "restart": action.restart, "alpha": action.alpha,
            "proposed_restart": proposed.restart, "proposed_alpha": proposed.alpha,
            "decision_source": source,
            "relative_residual_start": context["metrics"]["residual_norm"] / context["rhs_norm"],
            "started": started, "feature_history": copy.deepcopy(self.frames[-5:]),
        }
        self.previous_action = action
        return {"restart": action.restart, "round_tolerance": self.base_eps,
                "orth_round_tolerance": self.base_eps, "solution_round_tolerance": self.base_eps}

    def on_cycle_end(self, context: dict) -> None:
        record = self.current.copy()
        record["wall_seconds"] = time.perf_counter() - record.pop("started")
        relative = context["metrics"]["residual_norm"] / context["rhs_norm"]
        record.update({
            "iteration_end": context["total_iterations"],
            "relative_residual_end": relative,
            "log_reduction": math.log(max(record["relative_residual_start"], 1e-300) /
                                      max(relative, 1e-300)),
            "solution_rank": tt_max_rank(context["x"]),
            "projected_residual": context["projected_residual"],
            "checked_transformed_residual": context["metrics"]["transformed_residual_norm"],
            "converged": bool(context["converged"]), "rounding": self.rounder.cycle_summary(),
        })
        self.records.append(record)
        self.previous = record
