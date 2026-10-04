"""Fast pure-TT rounding with explicit floating-point error accounting.

The default audit reuses discarded singular values from a left-QR/right-SVD
sweep. Their squared sum is the truncation error in exact arithmetic; it does
not certify floating-point QR/SVD error or the solver's residual. ``full`` adds
an independent QR norm of the input-minus-output TT. Neither mode is an
interval bound. Solver acceptance must still use an independent true residual.
"""
from __future__ import annotations

import math
import numbers
import time
from typing import Any, Dict, List, Optional

import torch
import tntorch as tn


class RoundFailure(ValueError):
    """A refused truncation; ``record`` is also appended to the rounder log."""

    def __init__(self, message: str, record: Dict[str, Any]):
        super().__init__(message)
        self.record = record


def _ranks(tensor) -> List[int]:
    return [int(core.shape[2]) for core in tensor.cores[:-1]]


def _caps(max_rank, count: int) -> Optional[List[int]]:
    if max_rank is None:
        return None
    values = [max_rank] * count if isinstance(max_rank, numbers.Integral) else list(max_rank)
    if len(values) != count or any(
        isinstance(rank, bool) or not isinstance(rank, numbers.Integral) or rank < 1
        for rank in values
    ):
        raise ValueError("max_rank must contain one positive integer per internal cut.")
    # A scalar cap also has to be valid for a one-mode TT (no internal cuts).
    if isinstance(max_rank, numbers.Integral) and (isinstance(max_rank, bool) or max_rank < 1):
        raise ValueError("max_rank must be a positive integer.")
    return [int(rank) for rank in values]


def _new_tensor(tensor, cores):
    return tn.Tensor(cores, idxs=getattr(tensor, "idxs", None), batch=False)


def _safe_norm(values) -> float:
    """Use scaling outside the ordinary range; never confuse underflow with zero."""
    value = float(torch.linalg.vector_norm(values))
    if math.isfinite(value) and 1e-150 <= value <= 1e150:
        return value
    scale = float(torch.amax(torch.abs(values)))
    if scale == 0.0 or not math.isfinite(scale):
        return scale
    return float(torch.linalg.vector_norm(values / scale)) * scale


def _independent_norm(tensor) -> float:
    """Right-to-left QR norm, independent of the rounding's left QR sweep."""
    cores = list(tensor.cores)
    for index in range(len(cores) - 1, 0, -1):
        core = cores[index]
        _, triangular = torch.linalg.qr(core.reshape(core.shape[0], -1).T, mode="reduced")
        previous = cores[index - 1]
        cores[index - 1] = (previous.reshape(-1, previous.shape[2]) @ triangular.T).reshape(
            previous.shape[0], previous.shape[1], -1
        )
    return _safe_norm(cores[0])


class FastRounder:
    """Callable with a relative ``eps`` or an explicit absolute error budget.

    Supported inputs are ordinary, unbatched, float64 CPU tensor trains.
    ``adaptive`` is retained as an integration argument; this rounder does not
    change the caller's tolerance or impose a cumulative error budget.
    ``absolute_tolerance=budget`` explicitly selects absolute-error acceptance;
    the supplied ``eps`` remains logged but is not the active relative limit.
    The SVD tolerance then equals ``budget / ||input||``. A tensor smaller than
    that budget may be replaced by a zero TT. Logs identify this budget mode
    and separately report whether the originally supplied relative eps was met.

    ``tail`` audits the sum of local singular-value tails without constructing
    a difference TT. ``full`` additionally measures that difference by a second,
    independent QR sweep. A rank cap that fails the selected audit raises
    :class:`RoundFailure`; a cap never silently replaces the tolerance.

    Near cancellation, QR arithmetic can exceed a relative tolerance even when
    every SVD tail is tiny. An explicitly logged conditioning heuristic detects
    large norm reductions during QR propagation and requires a full difference
    audit of the rounded candidate. Canonicalization is retained when it meets
    the requested budget; otherwise an input whose ranks already fit the cap
    is copied unchanged. Requests near machine precision or a zero QR norm
    are preserved directly when the existing ranks fit. This heuristic and both audits
    are floating-point diagnostics, not rigorous numerical certificates.
    """

    def __init__(self, adaptive: bool = False, audit_mode: str = "tail"):
        if audit_mode not in ("tail", "full"):
            raise ValueError("audit_mode must be 'tail' or 'full'.")
        self.adaptive = bool(adaptive)
        self.audit_mode = audit_mode
        self.logs: List[Dict[str, Any]] = []
        self.cycle = 0
        self._cycle_start = 0

    def start_cycle(self, *args, **kwargs) -> None:
        """Start a statistics cycle; tolerance selection belongs to the policy."""
        self.cycle += 1
        self._cycle_start = len(self.logs)

    @torch.no_grad()
    def __call__(self, tensor, eps: float, max_rank=None, phase: str = "round", *,
                 absolute_tolerance: Optional[float] = None):
        requested = float(eps)
        if not math.isfinite(requested) or requested < 0:
            raise ValueError("eps must be finite and nonnegative.")
        absolute_budget = None if absolute_tolerance is None else float(absolute_tolerance)
        if absolute_budget is not None and (not math.isfinite(absolute_budget) or absolute_budget < 0):
            raise ValueError("absolute_tolerance must be finite and nonnegative.")
        if not tensor.cores or getattr(tensor, "batch", False):
            raise ValueError("Expected a nonempty, unbatched pure TT.")
        if any(factor is not None for factor in tensor.Us):
            raise ValueError("FastRounder requires pure TT cores, without Tucker factors.")
        cores = list(tensor.cores)
        for index, core in enumerate(cores):
            if core.ndim != 3 or core.dtype != torch.float64 or core.device.type != "cpu":
                raise ValueError("FastRounder requires three-dimensional float64 CPU cores.")
            if min(core.shape) < 1 or (index and core.shape[0] != cores[index - 1].shape[2]):
                raise ValueError("TT core dimensions and adjacent ranks must match.")
        if cores[0].shape[0] != 1 or cores[-1].shape[2] != 1:
            raise ValueError("TT boundary ranks must equal one.")
        before = _ranks(tensor)
        caps = _caps(max_rank, len(before))
        original_fits = caps is None or all(rank <= cap for rank, cap in zip(before, caps))
        started = time.perf_counter()
        audit_seconds = 0.0
        core_norms = [_safe_norm(core) for core in cores]
        if not all(math.isfinite(value) for value in core_norms):
            raise FloatingPointError("TT core norms must be finite.")
        record: Dict[str, Any] = {
            "cycle": self.cycle, "phase": str(phase), "requested_eps": requested,
            "effective_eps": requested if absolute_budget is None else None, "audit_mode": self.audit_mode,
            "budget_mode": "relative" if absolute_budget is None else "absolute",
            "requested_absolute_error": absolute_budget,
            "effective_relative_eps": requested if absolute_budget is None else None,
            "ranks_before": before, "rank_before": max(before + [1]),
            "rank_cap": caps, "is_residual_certificate": False,
            "is_verified_error_bound": False, "full_audit_performed": False,
            "preserved_input": False, "preserved_reason": None,
            "tail_absolute_error": 0.0, "tail_relative_error": 0.0,
            "local_tail_errors": [], "actual_absolute_error": None,
            "actual_relative_error": None, "roundoff_relative_indicator": 0.0,
            "qr_cancellation_indicator": 1.0,
            "zeroed_with_absolute_budget": False,
        }

        def decompose(operation, *arguments, **options):
            try:
                return operation(*arguments, **options)
            except torch.linalg.LinAlgError as exc:
                record.update({
                    "input_norm": None, "ranks_after": before, "rank_after": max(before + [1]),
                    "rank_cap_hit": False, "rank_cap_satisfied": original_fits,
                    "within_requested": False, "within_effective": False,
                    "measured_within_requested": None, "audited_relative_error": None,
                    "audit_method": "decomposition_failed", "audit_acceptance": "rejected",
                    "status": "rejected", "failure_detail": str(exc),
                    "round_seconds": max(0.0, time.perf_counter() - started - audit_seconds),
                    "audit_seconds": audit_seconds,
                })
                self.logs.append(record)
                raise RoundFailure("TT-round QR/SVD decomposition failed: " + str(exc), record) from exc

        def finish(result, input_norm, absolute_error, method, *, preserved_reason=None):
            elapsed = time.perf_counter() - started
            after = _ranks(result)
            if absolute_budget is not None and record["effective_relative_eps"] is None and input_norm is not None:
                effective_value = (absolute_budget / input_norm if input_norm > 0 else
                                   0.0 if absolute_budget == 0 else math.inf)
                record["effective_relative_eps"] = record["effective_eps"] = effective_value
            relative = (absolute_error / input_norm if input_norm and input_norm > 0
                        else 0.0 if absolute_error == 0.0 else math.inf)
            if method == "floating_svd_tail_estimate":
                # Avoid losing a relative cap violation when the absolute error
                # underflows for a tensor near the smallest representable value.
                relative = record["tail_relative_error"]
            cap_met = caps is None or all(rank <= cap for rank, cap in zip(after, caps))
            within_relative = math.isfinite(relative) and relative <= requested
            within_absolute = (None if absolute_budget is None else
                               math.isfinite(absolute_error) and absolute_error <= absolute_budget)
            within = cap_met and (within_relative if absolute_budget is None else within_absolute)
            record.update({
                "input_norm": input_norm, "ranks_after": after, "rank_after": max(after + [1]),
                "rank_cap_hit": caps is not None and any(rank >= cap for rank, cap in zip(after, caps)),
                "rank_cap_satisfied": cap_met, "within_requested": within,
                "within_requested_relative": within_relative,
                "within_requested_absolute": within_absolute,
                "within_effective": within, "measured_within_requested": (
                    within if record["actual_relative_error"] is not None else None),
                "audited_relative_error": relative, "audited_absolute_error": absolute_error,
                "audit_method": method,
                "audit_acceptance": "within_tolerance" if within else "rejected",
                "status": "accepted" if within else "rejected",
                "round_seconds": max(0.0, elapsed - audit_seconds),
                "audit_seconds": audit_seconds,
                "preserved_input": preserved_reason is not None,
                "preserved_reason": preserved_reason,
            })
            self.logs.append(record)
            if not within:
                metric = "relative" if absolute_budget is None else "absolute"
                value = relative if absolute_budget is None else absolute_error
                limit = requested if absolute_budget is None else absolute_budget
                raise RoundFailure(
                    "TT-round %s %s error %.3e exceeds budget %.3e "
                    "or the rank cap (cap hit: %s)." %
                    (method, metric, value, limit, record["rank_cap_hit"]), record,
                )
            return result

        def unchanged(input_norm, reason):
            if record["zeroed_with_absolute_budget"]:
                record["candidate_zeroed_with_absolute_budget"] = True
                record["zeroed_with_absolute_budget"] = False
            if record["tail_absolute_error"] or record["tail_relative_error"]:
                record["candidate_tail_absolute_error"] = record["tail_absolute_error"]
                record["candidate_tail_relative_error"] = record["tail_relative_error"]
                record["tail_absolute_error"] = record["tail_relative_error"] = 0.0
                record["local_tail_errors"] = []
                record["local_absolute_tail_errors"] = []
            record["actual_absolute_error"] = 0.0
            record["actual_relative_error"] = 0.0
            return finish(tensor.clone(), input_norm, 0.0, "identical_input_cores", preserved_reason=reason)

        if any(value == 0.0 and not bool(torch.count_nonzero(core))
               for value, core in zip(core_norms, cores)):
            zero = _new_tensor(tensor, [core.new_zeros((1, core.shape[1], 1)) for core in cores])
            record["actual_absolute_error"] = record["actual_relative_error"] = 0.0
            return finish(zero, 0.0, 0.0, "structural_zero")
        if (requested if absolute_budget is None else absolute_budget) == 0 and original_fits:
            # No decomposition is needed to return the exact same representation.
            return unchanged(None, "zero_tolerance")
        if len(cores) == 1:
            return unchanged(core_norms[0], "single_core")

        maximum_growth = 1.0
        for index in range(len(cores) - 1):
            core = cores[index]
            orthogonal, triangular = decompose(torch.linalg.qr, core.reshape(-1, core.shape[2]), mode="reduced")
            cores[index] = orthogonal.reshape(core.shape[0], core.shape[1], -1)
            following = cores[index + 1]
            propagated = triangular @ following.reshape(following.shape[0], -1)
            propagation_norm = _safe_norm(propagated)
            if not math.isfinite(propagation_norm):
                raise FloatingPointError("TT-round QR propagation produced a nonfinite norm.")
            scale = _safe_norm(triangular) * core_norms[index + 1]
            growth = scale / propagation_norm if propagation_norm > 0 else math.inf
            maximum_growth = max(maximum_growth, growth)
            cores[index + 1] = propagated.reshape(-1, following.shape[1], following.shape[2])
        input_norm = _safe_norm(cores[-1])
        effective = (requested if absolute_budget is None else absolute_budget / input_norm
                     if input_norm > 0 else 0.0 if absolute_budget == 0 else math.inf)
        record["effective_relative_eps"] = record["effective_eps"] = effective
        machine_eps = torch.finfo(torch.float64).eps
        roundoff_indicator = 8.0 * machine_eps * len(cores) * math.sqrt(max(before + [1])) * maximum_growth
        # A growth of O(sqrt(rank)) is usual and is not evidence of cancellation.
        cancellation_risk = (maximum_growth > 1e3 and roundoff_indicator > 0.1 * effective)
        precision_risk = effective <= 64.0 * machine_eps
        record["qr_cancellation_indicator"] = maximum_growth
        record["roundoff_relative_indicator"] = roundoff_indicator
        if original_fits and (precision_risk or input_norm == 0):
            reason = "floating_precision_accuracy" if precision_risk else "cancellation_relative_accuracy"
            return unchanged(input_norm, reason)

        # Keeping a small margin prevents a cap at the tolerance boundary from
        # being accepted due only to rank-selection arithmetic.
        local_relative_budget = effective * (1.0 - 32.0 * machine_eps) / math.sqrt(len(cores) - 1)
        zero_with_budget = absolute_budget is not None and input_norm <= absolute_budget
        record["zeroed_with_absolute_budget"] = zero_with_budget
        local_relative_errors = [1.0] if zero_with_budget else []
        local_absolute_errors = [input_norm] if zero_with_budget else []
        sweep = () if zero_with_budget else range(len(cores) - 1, 0, -1)
        for index in sweep:
            core = cores[index]
            left, singular, right = decompose(torch.linalg.svd, core.reshape(core.shape[0], -1), full_matrices=False)
            audit_started = time.perf_counter()
            if not bool(torch.isfinite(singular).all()):
                raise FloatingPointError("TT-round SVD produced nonfinite singular values.")
            # Normalize before squaring to avoid overflow/underflow of ||X||^2.
            scaled = singular / input_norm if input_norm > 0 else singular
            tails = torch.cat((torch.flip(torch.cumsum(torch.flip(scaled.square(), [0]), 0), [0]),
                               singular.new_zeros(1)))
            allowed = tails[1:] <= local_relative_budget * local_relative_budget
            keep = int(torch.nonzero(allowed, as_tuple=False)[0, 0]) + 1
            if caps is not None:
                keep = min(keep, caps[index - 1])
            absolute_tail = _safe_norm(singular[keep:]) if keep < len(singular) else 0.0
            tail = (absolute_tail / input_norm if input_norm > 0 else
                    0.0 if absolute_tail == 0.0 else math.inf)
            local_relative_errors.append(tail)
            local_absolute_errors.append(absolute_tail)
            audit_seconds += time.perf_counter() - audit_started
            cores[index] = right[:keep, :].reshape(keep, core.shape[1], core.shape[2])
            previous = cores[index - 1]
            transfer = left[:, :keep] * singular[:keep]
            cores[index - 1] = (previous.reshape(-1, previous.shape[2]) @ transfer).reshape(
                previous.shape[0], previous.shape[1], keep
            )
        rounded = _new_tensor(tensor, [core.new_zeros((1, core.shape[1], 1)) for core in cores]
                              if zero_with_budget else cores)
        audit_started = time.perf_counter()
        # hypot scales internally. Squared sums, or relative_error * input_norm,
        # can underflow and falsely report a nonzero absolute error as zero.
        tail_relative = math.hypot(*local_relative_errors)
        tail_absolute = math.hypot(*local_absolute_errors)
        record.update({"tail_absolute_error": tail_absolute, "tail_relative_error": tail_relative,
                       "local_tail_errors": list(reversed(local_relative_errors)),
                       "local_absolute_tail_errors": list(reversed(local_absolute_errors))})
        error = tail_absolute
        method = "floating_svd_tail_estimate"
        if self.audit_mode == "full" or cancellation_risk or precision_risk or input_norm == 0:
            error = decompose(_independent_norm, tensor - rounded)
            independent_input_norm = decompose(_independent_norm, tensor)
            record.update({"full_audit_performed": True, "actual_absolute_error": error,
                           "actual_relative_error": (error / independent_input_norm if independent_input_norm > 0
                                                     else 0.0 if error == 0.0 else math.inf),
                           "rounding_input_norm": input_norm})
            input_norm = independent_input_norm
            method = "independent_floating_qr_difference_norm"
        audit_seconds += time.perf_counter() - audit_started
        accepted_absolute_limit = requested * input_norm if absolute_budget is None else absolute_budget
        if record["full_audit_performed"] and original_fits and error > accepted_absolute_limit:
            record["candidate_actual_relative_error"] = record["actual_relative_error"]
            record["candidate_actual_absolute_error"] = record["actual_absolute_error"]
            return unchanged(input_norm, "full_audit_rejected_candidate")
        return finish(rounded, input_norm, error, method)

    @staticmethod
    def _stats(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "round_calls": len(logs),
            "round_seconds": sum(row["round_seconds"] for row in logs),
            "audit_seconds": sum(row["audit_seconds"] for row in logs),
            "max_rank": max([row["rank_after"] for row in logs] + [1]),
            "rank_cap_hits": sum(row["rank_cap_hit"] for row in logs),
            "rejected_rounds": sum(row["status"] != "accepted" for row in logs),
            "full_audit_calls": sum(row["full_audit_performed"] for row in logs),
            "preserved_input_calls": sum(row["preserved_input"] for row in logs),
            "absolute_budget_calls": sum(row["budget_mode"] == "absolute" for row in logs),
            "zeroed_with_absolute_budget_calls": sum(row["zeroed_with_absolute_budget"] for row in logs),
            "tail_relative_error_sum": sum(row["tail_relative_error"] for row in logs),
            "audit_kind": "floating_point_local_errors",
            "is_residual_certificate": False, "is_verified_error_bound": False,
        }

    def cycle_summary(self) -> Dict[str, Any]:
        summary = self._stats(self.logs[self._cycle_start:])
        summary.update({"cycle": self.cycle, "adaptive": self.adaptive, "audit_mode": self.audit_mode})
        return summary

    def cumulative_stats(self) -> Dict[str, Any]:
        summary = self._stats(self.logs)
        summary.update({"cycles": self.cycle, "audit_mode": self.audit_mode})
        return summary
