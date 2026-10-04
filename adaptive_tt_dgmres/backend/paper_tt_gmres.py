from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import tntorch as tn


torch.set_default_dtype(torch.float64)


TTTensor = tn.Tensor
DeviceLike = Union[torch.device, str]


def _prod(values: Sequence[int]) -> int:
    out = 1
    for value in values:
        out *= int(value)
    return out


def tt_clone(tensor: TTTensor) -> TTTensor:
    return tensor.clone()


def tt_device(tensor: TTTensor) -> torch.device:
    """Return the device shared by the TT cores.

    ``tntorch.Tensor`` does not expose a public ``device`` attribute.  Keeping
    this small adapter here avoids reaching into ``.cores`` throughout the
    solver and, importantly, lets tensor factories follow an existing TT.
    """
    if not tensor.cores:
        raise ValueError("A TT tensor must contain at least one core.")
    return tensor.cores[0].device


def tt_dtype(tensor: TTTensor) -> torch.dtype:
    """Return the dtype shared by the TT cores."""
    if not tensor.cores:
        raise ValueError("A TT tensor must contain at least one core.")
    return tensor.cores[0].dtype


def tt_to(
    tensor: TTTensor,
    *,
    device: DeviceLike | None = None,
    dtype: torch.dtype | None = None,
    copy: bool = False,
) -> TTTensor:
    """Move/cast a TT tensor without relying on global PyTorch defaults.

    ``tntorch.Tensor.to`` mutates the object in place.  This wrapper mirrors
    PyTorch's usual non-copying behaviour when no conversion is needed and
    clones before a real conversion, so moving a problem to CUDA does not
    unexpectedly modify a caller-owned CPU tensor.  Pass ``copy=True`` to
    force a clone even when device and dtype already match.
    """
    target_device = tt_device(tensor) if device is None else torch.device(device)
    target_dtype = tt_dtype(tensor) if dtype is None else dtype
    needs_conversion = target_device != tt_device(tensor) or target_dtype != tt_dtype(tensor)
    if not copy and not needs_conversion:
        return tensor

    result = tensor.clone()
    result.cores = [
        core.to(device=target_device, dtype=target_dtype)
        for core in result.cores
    ]
    result.Us = [
        None if factor is None else factor.to(device=target_device, dtype=target_dtype)
        for factor in result.Us
    ]
    # tntorch's ``to(dtype)`` also casts integer indexing metadata and its
    # clone currently shares that metadata with the source.  Move independent
    # copies of the indices by device only.
    result.idxs = [index.clone().to(device=target_device) for index in result.idxs]
    return result


def tt_scalar(value: Any) -> float:
    """Convert a real scalar (including a CUDA tensor) to a Python float.

    This is the one intentional device synchronization point used by the
    NumPy least-squares implementation.  Keeping it explicit makes accidental
    ``float(cuda_tensor)`` conversions easy to find and replace later.
    """
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Expected a scalar tensor.")
        value = value.detach().item()
    return float(value)


def tt_zeros(
    shape: Sequence[int],
    *,
    device: DeviceLike | None = None,
    dtype: torch.dtype | None = None,
    like: TTTensor | None = None,
) -> TTTensor:
    """Create a rank-one zero TT on an explicit device and dtype.

    Defaults remain CPU/float64 for compatibility.  ``like`` is convenient in
    iterative algorithms, where a newly-created zero must live beside the
    right-hand side; explicit ``device``/``dtype`` take precedence over it.
    """
    if like is not None:
        resolved_device = tt_device(like) if device is None else torch.device(device)
        resolved_dtype = tt_dtype(like) if dtype is None else dtype
    else:
        resolved_device = torch.device("cpu") if device is None else torch.device(device)
        resolved_dtype = torch.float64 if dtype is None else dtype

    cores = [
        torch.zeros((1, int(mode_size), 1), device=resolved_device, dtype=resolved_dtype)
        for mode_size in shape
    ]
    if not cores:
        raise ValueError("A TT tensor must have at least one mode.")
    return tn.Tensor(cores, device=resolved_device)


def tt_randn(
    shape: Sequence[int],
    ranks_tt: int | Sequence[int] = 2,
    *,
    device: DeviceLike | None = None,
    dtype: torch.dtype | None = None,
) -> TTTensor:
    """Create a random TT directly on the requested device and dtype."""
    mode_sizes = [int(mode_size) for mode_size in shape]
    if not mode_sizes:
        raise ValueError("A TT tensor must have at least one mode.")

    if isinstance(ranks_tt, int):
        internal_ranks = [int(ranks_tt)] * (len(mode_sizes) - 1)
    else:
        internal_ranks = [int(rank) for rank in ranks_tt]
        if len(internal_ranks) != len(mode_sizes) - 1:
            raise ValueError("Expected one TT rank per internal interface.")
    if any(rank < 1 for rank in internal_ranks):
        raise ValueError("TT ranks must be positive.")

    resolved_device = torch.device("cpu") if device is None else torch.device(device)
    resolved_dtype = torch.float64 if dtype is None else dtype
    ranks = [1, *internal_ranks, 1]
    cores = [
        torch.randn(
            (ranks[index], mode_size, ranks[index + 1]),
            device=resolved_device,
            dtype=resolved_dtype,
        )
        for index, mode_size in enumerate(mode_sizes)
    ]
    return tn.Tensor(cores, device=resolved_device)


def _tt_round_inplace(
    tensor: TTTensor,
    *,
    eps: float,
    rmax: int | Sequence[int] | None = None,
) -> TTTensor:
    """Call tntorch rounding with a device-safe exact-zero fallback."""
    original_device = tt_device(tensor)
    original_dtype = tt_dtype(tensor)
    try:
        tensor.round_tt(eps=eps, rmax=rmax)
        if tt_device(tensor) != original_device or tt_dtype(tensor) != original_dtype:
            return tt_zeros(tensor.shape, device=original_device, dtype=original_dtype)
        return tensor
    except RuntimeError:
        # tntorch's truncated-SVD special case for an all-zero matrix creates
        # untyped CPU tensors.  It consequently fails for CUDA (and float32)
        # exact-zero TTs.  The routine may already have installed one of those
        # CPU cores before the following GPU einsum raises, so do not contract
        # the now mixed-device tensor here.  A changed core identifies that
        # particular branch; unrelated runtime failures are re-raised.
        changed_core = any(
            core.device != original_device or core.dtype != original_dtype
            for core in tensor.cores
        )
        if changed_core:
            return tt_zeros(tensor.shape, device=original_device, dtype=original_dtype)
        raise


def tt_round(tensor: TTTensor, eps: float) -> TTTensor:
    rounded = tensor.clone()
    if eps > 0.0:
        rounded = _tt_round_inplace(rounded, eps=eps)
    return rounded


def tt_round_capped(
    tensor: TTTensor,
    eps: float,
    max_rank: int | Sequence[int] | None = None,
) -> TTTensor:
    """Round a TT once with both a tolerance and rank cap."""
    rounded = tensor.clone()
    if eps > 0.0 or max_rank is not None:
        rounded = _tt_round_inplace(rounded, eps=eps, rmax=max_rank)
    return rounded


def tt_dot_tensor(x: TTTensor, y: TTTensor) -> torch.Tensor:
    """Return the TT inner product as a scalar tensor on the cores' device."""
    if x.dim() != y.dim() or tuple(x.shape) != tuple(y.shape):
        raise ValueError("TT inner products require tensors with identical shapes.")
    if tt_device(x) != tt_device(y):
        raise ValueError("TT inner products require tensors on the same device.")
    if tt_dtype(x) != tt_dtype(y):
        raise ValueError("TT inner products require tensors with the same dtype.")

    # tntorch currently initializes its contraction environment using the
    # global default dtype.  That breaks float32 tensors and can also create an
    # unintended CPU/CUDA conversion.  This equivalent pure-TT contraction is
    # explicitly allocated beside the cores.
    environment = torch.ones(
        (x.cores[0].shape[0], y.cores[0].shape[0]),
        device=tt_device(x),
        dtype=tt_dtype(x),
    )
    for mode, (x_core, y_core) in enumerate(zip(x.cores, y.cores)):
        if x_core.dim() != 3 or y_core.dim() != 3:
            raise ValueError("tt_dot_tensor currently supports ordinary TT cores only.")
        if x.Us[mode] is not None:
            x_core = torch.einsum("aib,ni->anb", x_core, x.Us[mode])
        if y.Us[mode] is not None:
            y_core = torch.einsum("aib,ni->anb", y_core, y.Us[mode])
        environment = torch.einsum(
            "ab,aic,bid->cd",
            environment,
            x_core.conj(),
            y_core,
        )
    return environment.sum().real


def tt_dot(x: TTTensor, y: TTTensor) -> float:
    return tt_scalar(tt_dot_tensor(x, y))


def tt_norm_tensor(x: TTTensor) -> torch.Tensor:
    """Return the TT norm as a scalar tensor on the cores' device."""
    return torch.sqrt(torch.clamp(tt_dot_tensor(x, x), min=0))


def tt_norm(x: TTTensor) -> float:
    return tt_scalar(tt_norm_tensor(x))


def tt_max_rank(x: TTTensor) -> int:
    ranks = [int(value) for value in x.ranks_tt.tolist()]
    internal = ranks[1:-1]
    return max(internal) if internal else 1


def tt_rank_tuple(x: TTTensor) -> Tuple[int, ...]:
    return tuple(int(value) for value in x.ranks_tt.tolist())


def tt_storage_ratio(x: TTTensor) -> float:
    return float(x.numcoef()) / float(x.numel())


def tt_rank1(
    factors: Sequence[torch.Tensor],
    *,
    device: DeviceLike | None = None,
    dtype: torch.dtype | None = None,
) -> TTTensor:
    if not factors:
        raise ValueError("Expected at least one rank-one factor.")
    resolved_device = factors[0].device if device is None else torch.device(device)
    resolved_dtype = factors[0].dtype if dtype is None else dtype
    cores = [
        factor.to(device=resolved_device, dtype=resolved_dtype)
        .reshape(1, factor.numel(), 1)
        .clone()
        for factor in factors
    ]
    return tn.Tensor(cores, device=resolved_device)


def tt_scale(tensor: TTTensor, coefficient: Any) -> TTTensor:
    """Scale a TT without moving a tensor coefficient through the host.

    ``tntorch``'s scalar multiplication routes through NumPy, which cannot
    consume a CUDA scalar.  Multiplying one TT core is mathematically
    equivalent and keeps a 0-D coefficient on the tensor's device.
    """
    if isinstance(coefficient, torch.Tensor):
        if coefficient.numel() != 1:
            raise ValueError("Expected a scalar coefficient.")
        if coefficient.device != tt_device(tensor):
            raise ValueError("A tensor coefficient must be on the TT tensor's device.")
        if coefficient.dtype != tt_dtype(tensor):
            raise ValueError("A tensor coefficient must have the TT tensor's dtype.")
        scalar = coefficient.reshape(())
    else:
        scalar = torch.as_tensor(
            coefficient,
            device=tt_device(tensor),
            dtype=tt_dtype(tensor),
        )
        if scalar.numel() != 1:
            raise ValueError("Expected a scalar coefficient.")
        scalar = scalar.reshape(())

    result = tensor.clone()
    result.cores[0] = result.cores[0] * scalar
    return result


def tt_linear_combination(
    vectors: Sequence[TTTensor],
    coeffs: Sequence[Any],
    round_eps: float,
    max_rank: int | None = None,
) -> TTTensor:
    """Form a TT linear combination and compress it exactly once.

    Passing ``max_rank`` lets callers combine the tolerance and rank cap in a
    single TT-SVD.  The previous DGMRES path rounded once without a cap and
    immediately rounded the same tensor again with a cap.
    """
    if not vectors:
        raise ValueError("Expected at least one TT vector.")
    result = tt_zeros(vectors[0].shape, like=vectors[0])
    for coeff, vector in zip(coeffs, vectors):
        result = result + tt_scale(vector, coeff)
    if round_eps > 0.0 or max_rank is not None:
        result = _tt_round_inplace(result, eps=round_eps, rmax=max_rank)
    return result


@dataclass
class TTLinearOperator:
    shape: Tuple[int, ...]
    matvec_fn: Callable[[TTTensor], TTTensor]
    name: str = "A"
    exact_norm: float | None = None
    device: torch.device | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if self.device is not None:
            self.device = torch.device(self.device)

    def matvec(self, vector: TTTensor) -> TTTensor:
        if self.device is not None and tt_device(vector) != self.device:
            raise ValueError(
                f"Operator {self.name!r} is on {self.device}, but the vector is on "
                f"{tt_device(vector)}. Move the problem consistently with tt_to()."
            )
        if self.dtype is not None and tt_dtype(vector) != self.dtype:
            raise ValueError(
                f"Operator {self.name!r} uses {self.dtype}, but the vector uses "
                f"{tt_dtype(vector)}. Cast the problem consistently with tt_to()."
            )
        return self.matvec_fn(vector)

    def __call__(self, vector: TTTensor) -> TTTensor:
        return self.matvec(vector)


def tt_matrix_operator(
    cores: Sequence[torch.Tensor],
    name: str = "A",
    exact_norm: float | None = None,
    *,
    device: DeviceLike | None = None,
    dtype: torch.dtype | None = None,
) -> TTLinearOperator:
    if not cores:
        raise ValueError("Expected at least one TT-matrix core.")

    resolved_device = cores[0].device if device is None else torch.device(device)
    resolved_dtype = cores[0].dtype if dtype is None else dtype
    shape = tuple(int(core.shape[1]) for core in cores)
    stored_cores = [
        core.to(device=resolved_device, dtype=resolved_dtype).clone()
        for core in cores
    ]

    def matvec(vector: TTTensor) -> TTTensor:
        result_cores: List[torch.Tensor] = []
        for a_core, x_core in zip(stored_cores, vector.cores):
            contracted = torch.einsum("aijb,cjd->acibd", a_core, x_core)
            ra1, rx1, n, ra2, rx2 = contracted.shape
            result_cores.append(contracted.reshape(ra1 * rx1, n, ra2 * rx2))
        return tn.Tensor(result_cores, device=resolved_device)

    return TTLinearOperator(
        shape=shape,
        matvec_fn=matvec,
        name=name,
        exact_norm=exact_norm,
        device=resolved_device,
        dtype=resolved_dtype,
    )


def compose_operators(left: TTLinearOperator, right: TTLinearOperator, name: str | None = None) -> TTLinearOperator:
    if left.shape != right.shape:
        raise ValueError("Operator shapes do not match for composition.")
    if left.device is not None and right.device is not None and left.device != right.device:
        raise ValueError("Operator devices do not match for composition.")
    if left.dtype is not None and right.dtype is not None and left.dtype != right.dtype:
        raise ValueError("Operator dtypes do not match for composition.")

    composed_name = name or f"{left.name}*{right.name}"
    exact_norm = None
    if left.exact_norm is not None and right.exact_norm is not None:
        exact_norm = left.exact_norm * right.exact_norm

    def matvec(vector: TTTensor) -> TTTensor:
        return left.matvec(right.matvec(vector))

    return TTLinearOperator(
        shape=left.shape,
        matvec_fn=matvec,
        name=composed_name,
        exact_norm=exact_norm,
        device=left.device if left.device is not None else right.device,
        dtype=left.dtype if left.dtype is not None else right.dtype,
    )


def estimate_operator_norm(
    operator: TTLinearOperator,
    samples: int = 10,
    tt_rank: int = 2,
    seed: int = 0,
    *,
    device: DeviceLike | None = None,
    dtype: torch.dtype | None = None,
) -> float:
    torch.manual_seed(seed)
    resolved_device = (
        operator.device
        if device is None and operator.device is not None
        else torch.device("cpu") if device is None else torch.device(device)
    )
    resolved_dtype = (
        operator.dtype
        if dtype is None and operator.dtype is not None
        else torch.float64 if dtype is None else dtype
    )
    best = 0.0
    for _ in range(samples):
        vector = tt_randn(
            operator.shape,
            ranks_tt=tt_rank,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        norm = tt_norm(vector)
        if norm == 0.0:
            continue
        vector = (1.0 / norm) * vector
        best = max(best, tt_norm(operator.matvec(vector)))
    return best


def backward_error(operator: TTLinearOperator, rhs: TTTensor, x: TTTensor, operator_norm: float) -> Tuple[float, TTTensor]:
    residual = rhs - operator.matvec(x)
    residual_norm = tt_norm(residual)
    denom = operator_norm * tt_norm(x) + tt_norm(rhs)
    if denom == 0.0:
        return residual_norm, residual
    return residual_norm / denom, residual


@dataclass
class TTGMRESConfig:
    restart: int = 25
    max_iterations: int = 500
    tolerance: float = 1e-5
    round_tolerance: float = 1e-5
    orth_round_tolerance: float | None = None
    residual_round_tolerance: float | None = None
    solution_round_tolerance: float | None = None
    operator_norm: float | None = None
    operator_norm_mode: str = "sampled"
    operator_norm_samples: int = 10
    operator_norm_tt_rank: int = 2
    operator_norm_seed: int = 0
    max_cycles: int | None = None
    safe_orth_rounding: bool = False

    def resolved(self) -> "TTGMRESConfig":
        return TTGMRESConfig(
            restart=self.restart,
            max_iterations=self.max_iterations,
            tolerance=self.tolerance,
            round_tolerance=self.round_tolerance,
            orth_round_tolerance=self.round_tolerance if self.orth_round_tolerance is None else self.orth_round_tolerance,
            residual_round_tolerance=self.round_tolerance if self.residual_round_tolerance is None else self.residual_round_tolerance,
            solution_round_tolerance=self.round_tolerance if self.solution_round_tolerance is None else self.solution_round_tolerance,
            operator_norm=self.operator_norm,
            operator_norm_mode=self.operator_norm_mode,
            operator_norm_samples=self.operator_norm_samples,
            operator_norm_tt_rank=self.operator_norm_tt_rank,
            operator_norm_seed=self.operator_norm_seed,
            max_cycles=self.max_cycles,
            safe_orth_rounding=self.safe_orth_rounding,
        )


@dataclass
class TTGMRESResult:
    solution: TTTensor
    converged: bool
    iterations: int
    cycles: int
    operator_norm: float
    history: Dict[str, List[float]]
    config: TTGMRESConfig

    @property
    def final_backward_error(self) -> float:
        return self.history["backward_error"][-1] if self.history["backward_error"] else math.nan


def _resolve_operator_norm(operator: TTLinearOperator, config: TTGMRESConfig) -> float:
    if config.operator_norm is not None:
        return float(config.operator_norm)
    if config.operator_norm_mode == "exact":
        if operator.exact_norm is None:
            raise ValueError("operator_norm_mode='exact' but operator.exact_norm is not available.")
        return float(operator.exact_norm)
    if config.operator_norm_mode == "sampled":
        return estimate_operator_norm(
            operator,
            samples=config.operator_norm_samples,
            tt_rank=config.operator_norm_tt_rank,
            seed=config.operator_norm_seed,
        )
    raise ValueError(f"Unsupported operator_norm_mode: {config.operator_norm_mode}")


def _empty_history() -> Dict[str, List[float]]:
    return {
        "iteration": [],
        "cycle": [],
        "backward_error": [],
        "true_residual_norm": [],
        "least_squares_residual_norm": [],
        "solution_rank": [],
        "last_vector_rank": [],
        "last_vector_ratio": [],
        "basis_ratio": [],
    }


def _append_history(
    history: Dict[str, List[float]],
    iteration: int,
    cycle: int,
    backward_eta: float,
    true_residual_norm: float,
    least_squares_residual_norm: float,
    solution: TTTensor,
    current_basis: Sequence[TTTensor],
) -> None:
    last_vector = current_basis[-1]
    dense_size = float(last_vector.numel())

    history["iteration"].append(float(iteration))
    history["cycle"].append(float(cycle))
    history["backward_error"].append(float(backward_eta))
    history["true_residual_norm"].append(float(true_residual_norm))
    history["least_squares_residual_norm"].append(float(least_squares_residual_norm))
    history["solution_rank"].append(float(tt_max_rank(solution)))
    history["last_vector_rank"].append(float(tt_max_rank(last_vector)))
    history["last_vector_ratio"].append(float(tt_storage_ratio(last_vector)))
    basis_storage = sum(float(vector.numcoef()) for vector in current_basis)
    history["basis_ratio"].append(basis_storage / (len(current_basis) * dense_size))


def tt_gmres_paper(
    operator: TTLinearOperator,
    rhs: TTTensor,
    config: TTGMRESConfig | None = None,
    x0: TTTensor | None = None,
) -> TTGMRESResult:
    cfg = (config or TTGMRESConfig()).resolved()
    operator_norm = _resolve_operator_norm(operator, cfg)

    x = tt_zeros(rhs.shape, like=rhs) if x0 is None else tt_round(x0, cfg.solution_round_tolerance)
    history = _empty_history()
    total_iterations = 0
    cycles = 0
    converged = False

    max_cycles = cfg.max_cycles
    if max_cycles is None:
        max_cycles = max(1, math.ceil(cfg.max_iterations / cfg.restart))

    while total_iterations < cfg.max_iterations and cycles < max_cycles and not converged:
        cycles += 1
        residual = rhs - operator.matvec(x)
        residual = tt_round(residual, cfg.residual_round_tolerance)
        beta = tt_norm(residual)
        if beta == 0.0:
            converged = True
            break

        basis: List[TTTensor] = [residual * (1.0 / beta)]
        hbar = np.zeros((cfg.restart + 1, cfg.restart), dtype=float)
        x_cycle = x

        for local_k in range(cfg.restart):
            if total_iterations >= cfg.max_iterations:
                break

            w = tt_round(operator.matvec(basis[local_k]), cfg.round_tolerance)
            for i in range(local_k + 1):
                hbar[i, local_k] = tt_dot(basis[i], w)
                w = w - hbar[i, local_k] * basis[i]
                if cfg.safe_orth_rounding:
                    w = tt_round(w, cfg.orth_round_tolerance)
            w = tt_round(w, cfg.orth_round_tolerance)

            hbar[local_k + 1, local_k] = tt_norm(w)
            if hbar[local_k + 1, local_k] > 0.0:
                basis.append(w * (1.0 / hbar[local_k + 1, local_k]))

            e1 = np.zeros(local_k + 2, dtype=float)
            e1[0] = beta
            hk = hbar[: local_k + 2, : local_k + 1]
            yk, *_ = np.linalg.lstsq(hk, e1, rcond=None)

            correction = tt_linear_combination(basis[: local_k + 1], yk, cfg.solution_round_tolerance)
            x_cycle = tt_round(x + correction, cfg.solution_round_tolerance)

            eta, true_residual = backward_error(operator, rhs, x_cycle, operator_norm)
            true_residual = tt_round(true_residual, cfg.residual_round_tolerance)
            ls_residual = float(np.linalg.norm(e1 - hk @ yk))

            total_iterations += 1
            _append_history(
                history=history,
                iteration=total_iterations,
                cycle=cycles,
                backward_eta=eta,
                true_residual_norm=tt_norm(true_residual),
                least_squares_residual_norm=ls_residual,
                solution=x_cycle,
                current_basis=basis[: local_k + 1],
            )

            if eta < cfg.tolerance:
                x = x_cycle
                converged = True
                break

            if hbar[local_k + 1, local_k] == 0.0:
                x = x_cycle
                converged = True
                break

        x = x_cycle

    return TTGMRESResult(
        solution=x,
        converged=converged,
        iterations=total_iterations,
        cycles=cycles,
        operator_norm=operator_norm,
        history=history,
        config=cfg,
    )


def tt_right_preconditioned_gmres_paper(
    operator: TTLinearOperator,
    preconditioner: TTLinearOperator,
    rhs: TTTensor,
    config: TTGMRESConfig | None = None,
    x0: TTTensor | None = None,
) -> TTGMRESResult:
    cfg = (config or TTGMRESConfig()).resolved()
    x = tt_zeros(rhs.shape, like=rhs) if x0 is None else tt_round(x0, cfg.solution_round_tolerance)
    composed = compose_operators(operator, preconditioner, name=f"{operator.name}*{preconditioner.name}")

    composed_norm = _resolve_operator_norm(composed, cfg)
    history = _empty_history()
    total_iterations = 0
    cycles = 0
    converged = False

    max_cycles = cfg.max_cycles
    if max_cycles is None:
        max_cycles = max(1, math.ceil(cfg.max_iterations / cfg.restart))

    while total_iterations < cfg.max_iterations and cycles < max_cycles and not converged:
        cycles += 1
        residual = tt_round(rhs - operator.matvec(x), cfg.residual_round_tolerance)

        inner_result = tt_gmres_paper(
            operator=composed,
            rhs=residual,
            config=TTGMRESConfig(
                restart=cfg.restart,
                max_iterations=min(cfg.restart, cfg.max_iterations - total_iterations),
                tolerance=cfg.tolerance,
                round_tolerance=cfg.round_tolerance,
                orth_round_tolerance=cfg.orth_round_tolerance,
                residual_round_tolerance=cfg.residual_round_tolerance,
                solution_round_tolerance=cfg.solution_round_tolerance,
                operator_norm=composed_norm,
                operator_norm_mode="provided",
                max_cycles=1,
                safe_orth_rounding=cfg.safe_orth_rounding,
            ),
            x0=None,
        )

        correction = preconditioner.matvec(inner_result.solution)
        x = tt_round(x + correction, cfg.solution_round_tolerance)

        for idx in range(inner_result.iterations):
            total_iterations += 1
            history["iteration"].append(float(total_iterations))
            history["cycle"].append(float(cycles))
            history["backward_error"].append(inner_result.history["backward_error"][idx])
            history["true_residual_norm"].append(inner_result.history["true_residual_norm"][idx])
            history["least_squares_residual_norm"].append(inner_result.history["least_squares_residual_norm"][idx])
            history["solution_rank"].append(inner_result.history["solution_rank"][idx])
            history["last_vector_rank"].append(inner_result.history["last_vector_rank"][idx])
            history["last_vector_ratio"].append(inner_result.history["last_vector_ratio"][idx])
            history["basis_ratio"].append(inner_result.history["basis_ratio"][idx])

        eta, _ = backward_error(operator, rhs, x, _resolve_operator_norm(operator, cfg))
        if eta < cfg.tolerance or inner_result.converged:
            converged = True

    return TTGMRESResult(
        solution=x,
        converged=converged,
        iterations=total_iterations,
        cycles=cycles,
        operator_norm=_resolve_operator_norm(operator, cfg),
        history=history,
        config=cfg,
    )


def poisson_1d_laplacian(
    n: int,
    *,
    device: DeviceLike | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    resolved_device = torch.device("cpu") if device is None else torch.device(device)
    h = 1.0 / (n + 1)
    main = 2.0 * torch.ones(n, device=resolved_device, dtype=dtype)
    off = -1.0 * torch.ones(n - 1, device=resolved_device, dtype=dtype)
    matrix = torch.diag(main) + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    return matrix / (h * h)


def laplacian_operator_norm_exact(n: int, dims: int = 3) -> float:
    h = 1.0 / (n + 1)
    lam_1d = 4.0 * np.sin(n * np.pi / (2.0 * (n + 1))) ** 2 / (h * h)
    return float(dims * lam_1d)


def poisson_operator_3d(
    n: int,
    *,
    device: DeviceLike | None = None,
    dtype: torch.dtype = torch.float64,
) -> TTLinearOperator:
    resolved_device = torch.device("cpu") if device is None else torch.device(device)
    eye = torch.eye(n, device=resolved_device, dtype=dtype)
    lap = poisson_1d_laplacian(n, device=resolved_device, dtype=dtype)

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

    exact_norm = laplacian_operator_norm_exact(n)
    return tt_matrix_operator(
        [core1, core2, core3],
        name="Poisson3D",
        exact_norm=exact_norm,
        device=resolved_device,
        dtype=dtype,
    )


def poisson_rhs_3d(
    n: int,
    *,
    device: DeviceLike | None = None,
    dtype: torch.dtype = torch.float64,
) -> TTTensor:
    resolved_device = torch.device("cpu") if device is None else torch.device(device)
    grid = torch.arange(1, n + 1, device=resolved_device, dtype=dtype) / (n + 1)
    a = 1.0 - grid**2
    ones = torch.ones_like(a)

    rhs = tt_zeros((n, n, n), device=resolved_device, dtype=dtype)
    for term in (
        tt_rank1([ones, a, a]),
        tt_rank1([a, ones, a]),
        tt_rank1([a, a, ones]),
    ):
        rhs = rhs + 2.0 * term
    return tt_round(rhs, 1e-14)


def default_paper_config() -> TTGMRESConfig:
    return TTGMRESConfig(
        restart=25,
        max_iterations=50,
        tolerance=1e-5,
        round_tolerance=1e-5,
        operator_norm_mode="exact",
    )


def summary_dict(result: TTGMRESResult) -> Dict[str, float | int | bool]:
    history = result.history
    return {
        "converged": result.converged,
        "iterations": result.iterations,
        "cycles": result.cycles,
        "final_backward_error": history["backward_error"][-1] if history["backward_error"] else math.nan,
        "final_true_residual_norm": history["true_residual_norm"][-1] if history["true_residual_norm"] else math.nan,
        "max_solution_rank": max(history["solution_rank"]) if history["solution_rank"] else 0,
        "max_last_vector_rank": max(history["last_vector_rank"]) if history["last_vector_rank"] else 0,
    }


def save_single_run_plot(result: TTGMRESResult, output_path: Path | str) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history = result.history

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].semilogy(history["iteration"], history["backward_error"], marker="x")
    axes[0].set_title("Backward Error")
    axes[0].set_xlabel("iteration")
    axes[0].grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    axes[1].plot(history["iteration"], history["solution_rank"], marker="x")
    axes[1].set_title("Solution TT-rank")
    axes[1].set_xlabel("iteration")
    axes[1].grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    axes[2].plot(history["iteration"], history["last_vector_rank"], marker="x")
    axes[2].set_title("Last Krylov TT-rank")
    axes[2].set_xlabel("iteration")
    axes[2].grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig.suptitle("Single TT-GMRES Run", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")
    return output_path


def run_figure3_experiment(
    ns: Sequence[int],
    config: TTGMRESConfig,
    output_dir: Path | str = "outputs_paper_figure3",
) -> list[dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for n in ns:
        print(f"Running n={n} ...")
        operator = poisson_operator_3d(n)
        rhs = poisson_rhs_3d(n)
        result = tt_gmres_paper(operator, rhs, config)
        item = {
            "n": n,
            "config": {
                "restart": config.restart,
                "max_iterations": config.max_iterations,
                "tolerance": config.tolerance,
                "round_tolerance": config.round_tolerance,
                "operator_norm_mode": config.operator_norm_mode,
            },
            "summary": summary_dict(result),
            "history": result.history,
        }
        results.append(item)
        print(json.dumps(item["summary"], indent=2))

    json_path = output_dir / "paper_figure3_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved figure-3 style raw data to {json_path}")
    return results


def save_figure3_plot(results: Sequence[dict], output_path: Path | str) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 6, hspace=0.5, wspace=0.6)
    axes = [
        fig.add_subplot(gs[0, 1:3]),
        fig.add_subplot(gs[0, 3:5]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6]),
    ]

    titles = [
        "(a) Convergence history",
        "(b) Maximal TT-rank of the iterative solution",
        "(c) Maximal TT-rank of the last Krylov vector",
        "(d) Compression ratio for the last Krylov vector",
        "(e) Compression ratio for the entire Krylov basis",
    ]
    ylabels = [
        r"$\eta_{A,b}$",
        "maximal TT-rank",
        "maximal TT-rank",
        "ratio",
        "ratio",
    ]

    for item in results:
        label = f"n = {item['n']}"
        history = item["history"]
        x = history["iteration"]
        axes[0].semilogy(x, history["backward_error"], marker="x", markersize=3, linewidth=1, label=label)
        axes[1].plot(x, history["solution_rank"], marker="x", markersize=3, linewidth=1, label=label)
        axes[2].plot(x, history["last_vector_rank"], marker="x", markersize=3, linewidth=1, label=label)
        axes[3].plot(x, history["last_vector_ratio"], marker="x", markersize=3, linewidth=1, label=label)
        axes[4].plot(x, history["basis_ratio"], marker="x", markersize=3, linewidth=1, label=label)

    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("iteration")
        axis.set_ylabel(ylabel)
        axis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        axis.legend(fontsize=8)

    fig.suptitle(r"Figure 3 style run: 3-d Poisson problem using $\delta=\varepsilon=10^{-5}$", fontsize=13)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")
    return output_path


def main() -> None:
    mode = "single_run"
    # mode = "figure3"

    config = default_paper_config()
    config.operator_norm_mode = "exact"

    if mode == "single_run":
        # Change parameters here for a single TT-GMRES run.
        n = 15
        config.restart = 5
        config.max_iterations = 10
        config.tolerance = 1e-5
        config.round_tolerance = 1e-5

        operator = poisson_operator_3d(n)
        rhs = poisson_rhs_3d(n)
        result = tt_gmres_paper(operator, rhs, config)

        summary = summary_dict(result)
        print("Config:")
        print(config)
        print("\nSummary:")
        print(json.dumps(summary, indent=2))

        out_dir = Path("outputs_paper_demo")
        out_dir.mkdir(exist_ok=True)
        history_path = out_dir / "paper_tt_gmres_history.json"
        history_path.write_text(json.dumps(result.history, indent=2), encoding="utf-8")
        print(f"\nSaved history to {history_path}")
        plot_path = save_single_run_plot(result, out_dir / "paper_tt_gmres_plot.png")
        if "ipykernel" in sys.modules:
            try:
                from IPython.display import Image, display

                display(Image(filename=str(plot_path)))
            except Exception:
                pass
        return

    if mode == "figure3":
        # Change parameters here for a multi-case Figure 3 style run.
        ns = [63, 127, 255]
        config.restart = 25
        config.max_iterations = 50
        config.tolerance = 1e-5
        config.round_tolerance = 1e-5

        out_dir = Path("outputs_paper_figure3")
        results = run_figure3_experiment(ns=ns, config=config, output_dir=out_dir)
        plot_path = save_figure3_plot(results, out_dir / "paper_figure3_plot.png")
        if "ipykernel" in sys.modules:
            try:
                from IPython.display import Image, display

                display(Image(filename=str(plot_path)))
            except Exception:
                pass
        return

    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
