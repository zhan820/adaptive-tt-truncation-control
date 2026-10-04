"""Audited empirical local-error budgets for the learning prototype.

The budget is a sum of local relative TT truncation errors. It is NOT a weighted
residual-gap bound, backward-error certificate, or convergence guarantee. An
independent recomputed-residual gate remains responsible for solver acceptance.
All error audits are floating-point QR norm evaluations, not interval bounds.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import torch

from .backend.paper_tt_dgmres import _stable_tt_norm
from .backend.paper_tt_gmres import tt_round_capped


class BudgetRoundError(ValueError):
    """A rejected round; ``record`` contains the corresponding audit log."""

    def __init__(self, message: str, record: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.record = record


class BudgetExhaustedError(BudgetRoundError):
    pass


class RoundingToleranceError(BudgetRoundError):
    pass


def _ranks(tensor) -> List[int]:
    return [int(core.shape[-1]) for core in tensor.cores[:-1]]


def _sync(tensor) -> None:
    if tensor.cores[0].is_cuda:
        torch.cuda.synchronize(tensor.cores[0].device)


def _physical_tt(tensor):
    """Absorb optional Tucker mode factors before the backend pure-TT norm.

    Factors in a subtraction need not be orthonormal, so a QR norm of the cores
    alone can be wrong. Absorption is local and never expands the whole tensor.
    """
    if all(factor is None for factor in tensor.Us):
        return tensor
    physical = tensor.clone()
    physical.cores = [
        core if factor is None else torch.einsum("anb,in->aib", core, factor)
        for core, factor in zip(physical.cores, physical.Us)
    ]
    physical.Us = [None] * len(physical.cores)
    return physical


class BudgetRounder:
    """Callable ``(tensor, eps, max_rank=None, phase='round') -> rounded``.

    ``start_cycle(restart, alpha, base_eps, index=1)`` resets the cycle budget.
    The heuristic expected count is m(m+1)+4m+2a+4. Each call gets at most its
    requested tolerance and the remaining budget divided by the estimated
    remaining calls. Actual errors, rather than requested tolerances, consume
    the budget. Rank-cap violations raise ``RoundingToleranceError``.

    With ``adaptive=False`` all calls use the original epsilon unchanged, which
    permits baseline parity checks; audits and real tolerance rejection remain.
    Calls before the first cycle use their requested epsilon and are audited.
    ``logs`` is cumulative; ``cycle_summary()`` and ``cumulative_stats()`` return
    JSON-compatible dictionaries. Timing includes a separate cost for audits.
    ``measured_within_*`` reports the measured error without floating slack;
    ``within_*`` reports the rejection check including explicitly logged slack.
    In particular, acceptance within slack is not a claim that the measured
    error met the requested tolerance. Near cancellation, an unattainably tight
    request is rejected rather than silently increasing this slack.
    """

    def __init__(self, adaptive: bool = True, audit: bool = True,
                 relative_noise_slack: float = 1e-12):
        if not audit:
            raise ValueError("This prototype requires audits for every TT-round.")
        if not math.isfinite(relative_noise_slack) or relative_noise_slack < 0:
            raise ValueError("relative_noise_slack must be finite and nonnegative.")
        self.adaptive = bool(adaptive)
        self.relative_noise_slack = float(relative_noise_slack)
        self.logs: List[Dict[str, Any]] = []
        self.cycle = 0
        self._cycle_start = 0
        self._expected_calls = 0
        self._budget_total: Optional[float] = None
        self._remaining_budget: Optional[float] = None
        self._cycle_calls = 0

    def start_cycle(self, restart: int, alpha: float, base_eps: float,
                    index: int = 1) -> None:
        if int(restart) != restart or int(index) != index or index < 0 or restart <= index:
            raise ValueError("Require integer restart > index >= 0.")
        if not math.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("Require 0 < alpha <= 1.")
        if not math.isfinite(base_eps) or base_eps <= 0:
            raise ValueError("base_eps must be positive and finite.")
        self.cycle += 1
        self._cycle_start = len(self.logs)
        self._cycle_calls = 0
        self._expected_calls = int(restart * (restart + 1) + 4 * restart + 2 * index + 4)
        self._budget_total = float(alpha * base_eps * self._expected_calls)
        self._remaining_budget = self._budget_total

    def __call__(self, tensor, eps: float, max_rank=None, phase: str = "round"):
        requested = float(eps)
        if not math.isfinite(requested) or requested < 0:
            raise ValueError("Round tolerance must be finite and nonnegative.")
        if not tensor.cores:
            raise ValueError("Cannot round an empty tensor train.")
        before = _ranks(tensor)
        if max_rank is None:
            caps = None
        elif isinstance(max_rank, int):
            caps = [max_rank] * len(before)
        else:
            caps = [int(rank) for rank in max_rank]
        if caps is not None and (len(caps) != len(before) or any(cap < 1 for cap in caps)):
            raise ValueError("Rank cap must be positive and match the internal cuts.")
        effective = requested
        allocation = None
        if self.adaptive and self._remaining_budget is not None:
            if self._remaining_budget <= 0:
                record = {"cycle": self.cycle, "phase": str(phase),
                          "requested_eps": requested, "status": "budget_exhausted"}
                raise BudgetExhaustedError("Empirical local truncation budget is exhausted.", record)
            remaining_calls = max(1, self._expected_calls - self._cycle_calls)
            allocation = self._remaining_budget / remaining_calls
            effective = min(requested, allocation)

        _sync(tensor)
        started = time.perf_counter()
        rounded = tt_round_capped(tensor, effective, max_rank)
        _sync(rounded)
        round_seconds = time.perf_counter() - started
        started = time.perf_counter()
        physical_input, physical_rounded = _physical_tt(tensor), _physical_tt(rounded)
        input_norm = float(_stable_tt_norm(physical_input))
        absolute_error = float(_stable_tt_norm(physical_input - physical_rounded))
        _sync(rounded)
        audit_seconds = time.perf_counter() - started
        after = _ranks(rounded)
        if input_norm > 0:
            relative_error = absolute_error / input_norm
        else:
            relative_error = 0.0 if absolute_error == 0.0 else math.inf
        machine_eps = torch.finfo(tensor.cores[0].real.dtype).eps
        noise_relative = max(
            self.relative_noise_slack,
            32 * machine_eps * len(tensor.cores) * math.sqrt(max(before + [1])),
        )
        noise_absolute = noise_relative * input_norm
        finite = math.isfinite(input_norm) and math.isfinite(absolute_error) and math.isfinite(relative_error)
        measured_within_requested = finite and absolute_error <= requested * input_norm
        measured_within_effective = finite and absolute_error <= effective * input_norm
        within_requested = finite and absolute_error <= requested * input_norm + noise_absolute
        within_effective = finite and absolute_error <= effective * input_norm + noise_absolute
        cap_hit = caps is not None and any(rank >= cap for rank, cap in zip(after, caps))
        remaining_before = self._remaining_budget
        within_budget = (not self.adaptive or remaining_before is None or
                         relative_error <= remaining_before + noise_relative)
        record = {
            "cycle": self.cycle, "phase": str(phase),
            "requested_eps": requested, "effective_eps": effective,
            "allocated_relative_error": allocation,
            "input_norm": input_norm, "actual_absolute_error": absolute_error,
            "actual_relative_error": relative_error,
            "noise_slack_absolute": noise_absolute, "noise_slack_relative": noise_relative,
            "rank_before": max(before + [1]), "rank_after": max(after + [1]),
            "ranks_before": before, "ranks_after": after,
            "rank_cap_hit": bool(cap_hit), "within_requested": bool(within_requested),
            "within_effective": bool(within_effective), "within_budget": bool(within_budget),
            "measured_within_requested": bool(measured_within_requested),
            "measured_within_effective": bool(measured_within_effective),
            "audit_acceptance": ("within_tolerance" if measured_within_effective else
                                 "within_floating_slack" if within_effective else "rejected"),
            "audit_method": "floating_qr_norm_not_verified_bound",
            "budget_remaining_before": remaining_before,
            "round_seconds": round_seconds, "audit_seconds": audit_seconds,
            "status": "accepted" if within_effective and within_budget else "rejected",
        }
        self.logs.append(record)
        if self._budget_total is not None:
            self._cycle_calls += 1
        if not within_effective:
            raise RoundingToleranceError(
                "TT-round actual relative error %.3e exceeds effective tolerance %.3e "
                "(floating audit slack %.3e, rank cap hit: %s)." %
                (relative_error, effective, noise_relative, cap_hit), record,
            )
        if not within_budget:
            raise BudgetExhaustedError("TT-round exceeded the remaining empirical error budget.", record)
        if self._remaining_budget is not None:
            self._remaining_budget = max(0.0, self._remaining_budget - relative_error)
        record["budget_remaining_after"] = self._remaining_budget
        return rounded

    @staticmethod
    def _stats(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "round_calls": len(logs),
            "round_seconds": sum(row["round_seconds"] for row in logs),
            "audit_seconds": sum(row["audit_seconds"] for row in logs),
            "actual_relative_error_sum": sum(row["actual_relative_error"] for row in logs),
            "actual_absolute_error_sum": sum(row["actual_absolute_error"] for row in logs),
            "noise_slack_relative_sum": sum(row["noise_slack_relative"] for row in logs),
            "rank_cap_hits": sum(row["rank_cap_hit"] for row in logs),
            "rejected_rounds": sum(row["status"] != "accepted" for row in logs),
            "accepted_with_floating_slack": sum(
                row["status"] == "accepted" and row["audit_acceptance"] == "within_floating_slack"
                for row in logs
            ),
            "max_rank": max([row["rank_after"] for row in logs] + [1]),
            "budget_kind": "empirical_sum_of_local_relative_errors",
            "is_residual_certificate": False,
        }

    def cycle_summary(self) -> Dict[str, Any]:
        summary = self._stats(self.logs[self._cycle_start:])
        summary.update({"cycle": self.cycle, "adaptive": self.adaptive,
                        "expected_round_calls": self._expected_calls,
                        "budget_total": self._budget_total,
                        "budget_remaining": self._remaining_budget})
        return summary

    def cumulative_stats(self) -> Dict[str, Any]:
        summary = self._stats(self.logs)
        summary["cycles"] = self.cycle
        return summary
