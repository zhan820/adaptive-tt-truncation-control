from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import matplotlib.pyplot as plt
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import torch

from .paper_tt_dgmres import (
    TTDGMRESConfig,
    save_single_run_plot,
    summary_dict,
    tt_dgmres_paper,
    tt_norm_stable,
)
from .paper_tt_gmres import (
    TTLinearOperator,
    TTTensor,
    tt_dot,
    tt_matrix_operator,
    tt_max_rank,
    tt_norm,
    tt_rank1,
    tt_round,
)


torch.set_default_dtype(torch.float64)


def _device_argument(value: str) -> str:
    """Validate a CPU/CUDA device string while preserving a CLI-friendly value."""
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if device.type not in {"cpu", "cuda"}:
        raise argparse.ArgumentTypeError(
            "--device must name a CPU or CUDA device (for example cpu, cuda, cuda:0)."
        )
    return str(device)


def _resolve_compute_device(device: torch.device | str) -> torch.device:
    """Resolve an index-less CUDA request to the worker's current GPU."""
    requested = torch.device(device)
    if requested.type == "cpu":
        return torch.device("cpu")
    if requested.type != "cuda":
        raise ValueError(f"Only CPU and CUDA devices are supported, got {requested}.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested} was requested, but torch.cuda.is_available() is false."
        )
    index = torch.cuda.current_device() if requested.index is None else requested.index
    if index < 0 or index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device index {index} is unavailable; found {torch.cuda.device_count()} device(s)."
        )
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_solution_slices(solution: torch.Tensor, output_dir: Path | str, prefix: str) -> Path:
    """Save central 2D slices for a tensor solution."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if solution.ndim < 2:
        raise ValueError("Solution slice plotting expects at least a 2D tensor.")

    shape = list(solution.shape)
    centers = [extent // 2 for extent in shape]

    def extract_plane(axis_a: int, axis_b: int) -> np.ndarray:
        index = []
        for axis in range(solution.ndim):
            if axis in (axis_a, axis_b):
                index.append(slice(None))
            else:
                index.append(centers[axis])
        return np.asarray(solution[tuple(index)].detach().cpu().numpy())

    if solution.ndim == 2:
        slices = [(solution.detach().cpu().numpy(), "2D solution")]
        fig, axes = plt.subplots(1, 1, figsize=(5, 4))
        axes = [axes]
    else:
        slices = [
            (extract_plane(0, 1), f"axes (0,1), others=center {centers[2:]}"),
            (extract_plane(0, 2), f"axes (0,2), others=center {[centers[1], *centers[3:]]}"),
            (extract_plane(1, 2), f"axes (1,2), others=center {[centers[0], *centers[3:]]}"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (slice_values, title) in zip(axes, slices):
        image = ax.imshow(slice_values, origin="lower", cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("index")
        ax.set_ylabel("index")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("TT-DGMRES Solution Slices", fontsize=13)
    fig.tight_layout()

    output_path = output_dir / f"{prefix}_solution_slices.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved solution slices to {output_path}")
    return output_path


def paper_neumann_stencil_1d(
    n: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """
    One-dimensional stencil from the paper's Example 5.1.

    We follow the matrix formula directly:

        T_{i,j} = (1 / h^2) * {
            2,   (i,j) = (1,2) or (N,N-1)
           -2,   j = i
            1,   2 <= i <= N-1 and j = i +- 1
            0,   otherwise
        }

    with h = 1 / (N - 1).
    """
    if n < 2:
        raise ValueError("n must be at least 2.")

    h = 1.0 / (n - 1)
    resolved_device = torch.device(device)
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


def kron_sum_tt_operator(
    local_operator: torch.Tensor,
    dims: int,
    *,
    name: str,
    operator_norm_bound: float | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> TTLinearOperator:
    """
    Build the exact TT-matrix representation of

        A = sum_{mu=1}^d I^(otimes(mu-1)) otimes T otimes I^(otimes(d-mu)).

    The resulting TT ranks are exactly 2.
    """
    if dims < 1:
        raise ValueError("dims must be positive.")

    resolved_device = torch.device(device)
    local_operator = local_operator.to(device=resolved_device, dtype=dtype)
    n = int(local_operator.shape[0])
    eye = torch.eye(n, device=resolved_device, dtype=dtype)

    if dims == 1:
        core = local_operator.reshape(1, n, n, 1).clone()
        return tt_matrix_operator(
            [core],
            name=name,
            exact_norm=operator_norm_bound,
            device=resolved_device,
            dtype=dtype,
        )

    cores: List[torch.Tensor] = []

    first = torch.zeros(
        (1, n, n, 2), device=resolved_device, dtype=dtype
    )
    first[0, :, :, 0] = eye
    first[0, :, :, 1] = local_operator
    cores.append(first)

    for _ in range(dims - 2):
        middle = torch.zeros(
            (2, n, n, 2), device=resolved_device, dtype=dtype
        )
        middle[0, :, :, 0] = eye
        middle[0, :, :, 1] = local_operator
        middle[1, :, :, 1] = eye
        cores.append(middle)

    last = torch.zeros((2, n, n, 1), device=resolved_device, dtype=dtype)
    last[0, :, :, 0] = local_operator
    last[1, :, :, 0] = eye
    cores.append(last)

    return tt_matrix_operator(
        cores,
        name=name,
        exact_norm=operator_norm_bound,
        device=resolved_device,
        dtype=dtype,
    )


def make_reference_seed(
    n: int,
    dims: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> TTTensor:
    """Construct a low-rank separated vector used to build a consistent singular test."""
    resolved_device = torch.device(device)
    grid = torch.linspace(
        0.0, 1.0, n, device=resolved_device, dtype=dtype
    )
    base_factors = [
        0.5 + torch.cos(np.pi * grid),
        1.0 + grid - grid**2,
        0.25 + torch.sin(2.0 * np.pi * grid) + grid**2,
        0.75 + torch.cos(3.0 * np.pi * grid) + 0.5 * grid,
        0.6 + torch.sin(np.pi * grid) + grid**3,
        0.9 + 0.3 * torch.cos(2.0 * np.pi * grid) + grid**2,
    ]
    factors = [base_factors[axis % len(base_factors)].clone() for axis in range(dims)]
    return tt_rank1(factors, device=resolved_device, dtype=dtype)


@dataclass
class ExampleProblem:
    operator: TTLinearOperator
    rhs: TTTensor
    reference_solution: TTTensor | None
    local_stencil: torch.Tensor
    operator_norm_bound: float
    operator_norm_lower_bound: float
    ambient_dimension: int
    problem_variant: str = "manufactured_range_solution"
    rhs_description: str = "x_exact=A*y, b=A*x_exact with a smooth rank-1 seed"
    rhs_compatibility_relative: float = 0.0
    rhs_tt_rank: int = 1


def build_neumann_poisson_problem(
    n: int,
    dims: int,
    *,
    round_tolerance: float = 1e-12,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ExampleProblem:
    """
    Build the singular linear system for Example 5.1 directly in TT format.

    To keep the singular system consistent and provide a reference solution, we use:

        y        : a rank-1 TT seed
        x_exact  = A y      in Range(A)
        rhs      = A x_exact

    This keeps the test aligned with DGMRES for index-1 singular systems.
    """
    resolved_device = torch.device(device)
    local_stencil = paper_neumann_stencil_1d(
        n, device=resolved_device, dtype=dtype
    )
    operator_norm_bound = float(dims) * float(
        np.linalg.norm(
            local_stencil.detach().to(device="cpu", dtype=torch.float64).numpy(),
            ord=2,
        )
    )
    # The endpoint Neumann stencil has spectral radius 4/h^2 exactly.  Since
    # rho(A) <= ||A||_2, this analytic value is a certified lower bound for the
    # Euclidean operator norm of the Kronecker sum.  Using it in the backward-
    # error denominator produces a conservative upper bound on the true error.
    operator_norm_lower_bound = float(4 * dims * (n - 1) ** 2)
    operator = kron_sum_tt_operator(
        local_stencil,
        dims=dims,
        name=f"PaperNeumannPoisson{dims}D",
        # d*||T||_2 is only an upper bound, not the exact Euclidean norm.
        # Do not expose it through TTLinearOperator.exact_norm.
        operator_norm_bound=None,
        device=resolved_device,
        dtype=dtype,
    )

    seed = make_reference_seed(
        n, dims, device=resolved_device, dtype=dtype
    )
    reference_solution = tt_round(operator.matvec(seed), round_tolerance)
    rhs = tt_round(operator.matvec(reference_solution), round_tolerance)
    ambient_dimension = int(n**dims)
    _, left_null = neumann_null_vectors_tt(
        n, dims, device=resolved_device, dtype=dtype
    )
    rhs_compatibility_relative = abs(tt_dot(left_null, rhs)) / max(
        tt_norm_stable(left_null) * tt_norm_stable(rhs), np.finfo(float).tiny
    )

    return ExampleProblem(
        operator=operator,
        rhs=rhs,
        reference_solution=reference_solution,
        local_stencil=local_stencil,
        operator_norm_bound=operator_norm_bound,
        operator_norm_lower_bound=operator_norm_lower_bound,
        ambient_dimension=ambient_dimension,
        problem_variant="manufactured_range_solution",
        rhs_description="x_exact=A*y, b=A*x_exact with a smooth rank-1 seed",
        rhs_compatibility_relative=float(rhs_compatibility_relative),
        rhs_tt_rank=tt_max_rank(rhs),
    )


def neumann_left_null_weights(
    n: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return the positive left-null vector of the one-dimensional stencil."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    weights = torch.ones(n, device=torch.device(device), dtype=dtype)
    weights[0] = 0.5
    weights[-1] = 0.5
    return weights


def neumann_null_vectors_tt(
    n: int,
    dims: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> tuple[TTTensor, TTTensor]:
    """Return the natural rank-one right/left null vectors of the Kronecker sum.

    The one-dimensional stencil satisfies ``T e = 0`` and ``w^T T = 0``
    for ``e = (1, ..., 1)^T`` and trapezoidal weights
    ``w = (1/2, 1, ..., 1, 1/2)^T``.  Consequently, the ``dims``-dimensional
    Kronecker sum has right null vector ``e^(otimes dims)`` and left null
    vector ``w^(otimes dims)``.  Both are represented with TT rank one.
    """
    if dims < 1:
        raise ValueError("dims must be positive.")
    resolved_device = torch.device(device)
    right_factor = torch.ones(n, device=resolved_device, dtype=dtype)
    left_factor = neumann_left_null_weights(
        n, device=resolved_device, dtype=dtype
    )
    right_null = tt_rank1(
        [right_factor.clone() for _ in range(dims)],
        device=resolved_device,
        dtype=dtype,
    )
    left_null = tt_rank1(
        [left_factor.clone() for _ in range(dims)],
        device=resolved_device,
        dtype=dtype,
    )
    return right_null, left_null


def neumann_range_audit_tt(
    solution: TTTensor,
    right_null: TTTensor,
    left_null: TTTensor,
) -> Dict[str, float]:
    """Measure violation of ``Range(A) = {x: left_null^T x = 0}`` in TT form.

    ``solution_null_coefficient`` is the coefficient of the right-null vector
    in the oblique decomposition

        x = x_range + coefficient * right_null,

    where ``left_null^T x_range = 0``.  ``solution_range_defect`` is the
    scale-free left-null contraction ``|l^T x| / (||l|| ||x||)``.  The final
    field is the relative norm of the correction needed by the oblique
    projector.  These definitions are shared with the dense audit below.
    """
    pairing = float(tt_dot(left_null, right_null))
    if not np.isfinite(pairing) or abs(pairing) <= np.finfo(float).tiny:
        raise ValueError("The left/right null-vector pairing must be nonzero.")
    contraction = float(tt_dot(left_null, solution))
    coefficient = contraction / pairing
    solution_norm = tt_norm_stable(solution)
    left_norm = tt_norm_stable(left_null)
    right_norm = tt_norm_stable(right_null)
    tiny = np.finfo(float).tiny
    range_defect = abs(contraction) / max(left_norm * solution_norm, tiny)
    correction_relative_norm = abs(coefficient) * right_norm / max(
        solution_norm, tiny
    )
    return {
        "solution_null_coefficient": float(coefficient),
        "solution_range_defect": float(range_defect),
        "solution_range_correction_relative_norm": float(
            correction_relative_norm
        ),
        "left_right_null_pairing": float(pairing),
    }


def project_neumann_range_tt(
    solution: TTTensor,
    right_null: TTTensor,
    left_null: TTTensor,
    *,
    round_tolerance: float = 0.0,
) -> TTTensor:
    """Apply ``x <- x - e (l^T x)/(l^T e)`` without dense materialisation.

    If a positive TT-rounding tolerance is requested, the rounded tensor is
    projected once more without rounding.  That last rank-one cleanup is
    intentional: TT-SVD rounding need not preserve the left-null constraint.
    """
    if round_tolerance < 0.0:
        raise ValueError("round_tolerance must be nonnegative.")
    pairing = float(tt_dot(left_null, right_null))
    if not np.isfinite(pairing) or abs(pairing) <= np.finfo(float).tiny:
        raise ValueError("The left/right null-vector pairing must be nonzero.")

    right_norm = tt_norm_stable(right_null)

    def remove_resolvable_null_component(current: TTTensor) -> TTTensor:
        coefficient = float(tt_dot(left_null, current)) / pairing
        correction_norm = abs(coefficient) * right_norm
        numerical_floor = (
            64.0
            * np.finfo(float).eps
            * max(tt_norm_stable(current), np.finfo(float).tiny)
        )
        if correction_norm <= numerical_floor:
            # Adding an all-zero (or roundoff-sized) rank-one block changes no
            # meaningful value but can increase every TT interface rank.
            return current
        return current - coefficient * right_null

    projected = remove_resolvable_null_component(solution)
    if round_tolerance > 0.0:
        projected = tt_round(projected, round_tolerance)
        projected = remove_resolvable_null_component(projected)
    return projected


def make_neumann_range_projector(
    n: int,
    dims: int,
    *,
    round_tolerance: float = 0.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Callable[[TTTensor], TTTensor]:
    """Create a cached TT projector suitable for a solver callback."""
    right_null, left_null = neumann_null_vectors_tt(
        n, dims, device=device, dtype=dtype
    )

    def projector(solution: TTTensor) -> TTTensor:
        return project_neumann_range_tt(
            solution,
            right_null,
            left_null,
            round_tolerance=round_tolerance,
        )

    return projector


def _left_null_contraction_dense(
    solution: np.ndarray,
    left_weights: np.ndarray,
    dims: int,
) -> float:
    """Contract a dense vector with a separable left-null vector."""
    n = int(left_weights.size)
    values = np.asarray(solution, dtype=float)
    expected_size = int(n**dims)
    if values.size != expected_size:
        raise ValueError(
            f"Expected a dense vector with {expected_size} entries, got {values.size}."
        )
    contraction: Any = values.reshape([n] * dims)
    for _ in range(dims):
        contraction = np.tensordot(left_weights, contraction, axes=(0, 0))
    return float(contraction)


def neumann_range_audit_dense(
    solution: np.ndarray,
    n: int,
    dims: int,
) -> Dict[str, float]:
    """Dense counterpart of :func:`neumann_range_audit_tt`.

    The implementation never allocates the ambient right/left null vectors;
    it uses their separability, which matters for the ``12^d`` experiments.
    """
    if dims < 1:
        raise ValueError("dims must be positive.")
    left_weights = neumann_left_null_weights(n).detach().cpu().numpy()
    contraction = _left_null_contraction_dense(solution, left_weights, dims)
    pairing = float(np.sum(left_weights) ** dims)
    coefficient = contraction / pairing
    solution_norm = float(np.linalg.norm(np.asarray(solution).reshape(-1)))
    left_norm = float(np.linalg.norm(left_weights) ** dims)
    right_norm = float(np.sqrt(n**dims))
    tiny = np.finfo(float).tiny
    range_defect = abs(contraction) / max(left_norm * solution_norm, tiny)
    correction_relative_norm = abs(coefficient) * right_norm / max(
        solution_norm, tiny
    )
    return {
        "solution_null_coefficient": float(coefficient),
        "solution_range_defect": float(range_defect),
        "solution_range_correction_relative_norm": float(
            correction_relative_norm
        ),
        "left_right_null_pairing": float(pairing),
    }


def project_neumann_range_dense(
    solution: np.ndarray,
    n: int,
    dims: int,
) -> np.ndarray:
    """Return the dense oblique Range(A) projection without null-vector arrays."""
    audit = neumann_range_audit_dense(solution, n, dims)
    projected = np.asarray(solution, dtype=float).copy()
    coefficient = float(audit["solution_null_coefficient"])
    correction_norm = abs(coefficient) * float(np.sqrt(projected.size))
    numerical_floor = (
        64.0
        * np.finfo(float).eps
        * max(float(np.linalg.norm(projected.reshape(-1))), np.finfo(float).tiny)
    )
    if correction_norm > numerical_floor:
        projected -= coefficient
    return projected


def make_neumann_range_projector_dense(
    n: int,
    dims: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create the dense Range(A) projector used for like-for-like histories."""

    def projector(solution: np.ndarray) -> np.ndarray:
        return project_neumann_range_dense(solution, n, dims)

    return projector


def internal_two_point_block(
    n: int,
    start: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Indicator of two adjacent interior grid nodes."""
    if n < 4:
        raise ValueError("An internal two-point block requires n >= 4.")
    if start < 1 or start > n - 3:
        raise ValueError(f"Internal block start must lie in [1, {n - 3}].")
    block = torch.zeros(n, device=torch.device(device), dtype=dtype)
    block[start : start + 2] = 1.0
    return block


def build_irregular_neumann_poisson_problem(
    n: int,
    dims: int,
    *,
    round_tolerance: float = 0.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> ExampleProblem:
    """Build a compatible Neumann--Poisson system with localized source--sink blocks.

    Unlike :func:`build_neumann_poisson_problem`, this routine does not choose an
    exact solution and back-compute ``b``.  It directly prescribes one pair of
    discontinuous tensor-product source and sink blocks.  Every one-dimensional
    block occupies two adjacent interior nodes.  Source and sink blocks have the
    same left-null mass, so each pair is compatible analytically.
    """
    if dims < 1:
        raise ValueError("dims must be positive.")
    if n != 12:
        raise ValueError(
            "The audited localized source--sink recipe is defined for n=12."
        )

    resolved_device = torch.device(device)
    local_stencil = paper_neumann_stencil_1d(
        n, device=resolved_device, dtype=dtype
    )
    operator_norm_bound = float(dims) * float(
        np.linalg.norm(
            local_stencil.detach().to(device="cpu", dtype=torch.float64).numpy(),
            ord=2,
        )
    )
    operator_norm_lower_bound = float(4 * dims * (n - 1) ** 2)
    operator = kron_sum_tt_operator(
        local_stencil,
        dims=dims,
        name=f"IrregularNeumannPoisson{dims}D",
        operator_norm_bound=None,
        device=resolved_device,
        dtype=dtype,
    )

    coefficients = [1.0]
    start_count = n - 3
    terms: List[tuple[float, TTTensor]] = []
    for pair_index, coefficient in enumerate(coefficients):
        positive_factors: List[torch.Tensor] = []
        negative_factors: List[torch.Tensor] = []
        for axis in range(dims):
            positive_start = 1 + ((2 * pair_index + 4 * axis) % start_count)
            negative_start = 1 + (
                (5 * pair_index + 7 * axis + 3) % start_count
            )
            positive_factors.append(
                internal_two_point_block(
                    n,
                    positive_start,
                    device=resolved_device,
                    dtype=dtype,
                )
            )
            negative_factors.append(
                internal_two_point_block(
                    n,
                    negative_start,
                    device=resolved_device,
                    dtype=dtype,
                )
            )
        terms.append(
            (
                float(coefficient),
                tt_rank1(
                    positive_factors,
                    device=resolved_device,
                    dtype=dtype,
                ),
            )
        )
        terms.append(
            (
                -float(coefficient),
                tt_rank1(
                    negative_factors,
                    device=resolved_device,
                    dtype=dtype,
                ),
            )
        )

    raw_rhs = terms[0][0] * terms[0][1]
    for coefficient, term in terms[1:]:
        raw_rhs = raw_rhs + coefficient * term
    if round_tolerance > 0.0:
        raw_rhs = tt_round(raw_rhs, round_tolerance)
    rhs_norm = tt_norm_stable(raw_rhs)
    if rhs_norm == 0.0:
        raise RuntimeError("The localized source--sink forcing vanished.")
    rhs = (1.0 / rhs_norm) * raw_rhs

    left_weights = neumann_left_null_weights(
        n, device=resolved_device, dtype=dtype
    )
    left_null = tt_rank1(
        [left_weights] * dims, device=resolved_device, dtype=dtype
    )
    left_norm = tt_norm_stable(left_null)
    compatibility_relative = abs(tt_dot(left_null, rhs)) / max(
        left_norm * tt_norm_stable(rhs), np.finfo(float).tiny
    )
    ambient_dimension = int(n**dims)
    return ExampleProblem(
        operator=operator,
        rhs=rhs,
        reference_solution=None,
        local_stencil=local_stencil,
        operator_norm_bound=operator_norm_bound,
        operator_norm_lower_bound=operator_norm_lower_bound,
        ambient_dimension=ambient_dimension,
        problem_variant="irregular_compatible_forcing",
        rhs_description=(
            "one balanced pair of discontinuous two-node tensor-product "
            "source/sink blocks; directly prescribed and not back-computed from "
            "a reference solution"
        ),
        rhs_compatibility_relative=float(compatibility_relative),
        rhs_tt_rank=tt_max_rank(rhs),
    )


def default_config(
    operator_norm_for_backward_error: float,
    *,
    restart: int = 30,
    max_iterations: int = 180,
    tolerance: float = 1e-10,
    round_tolerance: float = 1e-10,
) -> TTDGMRESConfig:
    return TTDGMRESConfig(
        index=1,
        restart=restart,
        max_iterations=max_iterations,
        tolerance=tolerance,
        round_tolerance=round_tolerance,
        operator_norm=operator_norm_for_backward_error,
        stopping_metric="relative_a_residual",
    )


def run_experiment(
    *,
    n: int = 12,
    dims: int = 3,
    restart: int = 30,
    max_iterations: int = 180,
    tolerance: float = 1e-10,
    round_tolerance: float = 1e-10,
    output_dir: str | Path = "outputs_tt_dgmres_neumann_poisson",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    resolved_device = _resolve_compute_device(device)
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    problem = build_neumann_poisson_problem(
        n=n,
        dims=dims,
        round_tolerance=1e-12,
        device=resolved_device,
        dtype=dtype,
    )
    config = default_config(
        problem.operator_norm_lower_bound,
        restart=restart,
        max_iterations=max_iterations,
        tolerance=tolerance,
        round_tolerance=round_tolerance,
    )

    solution_projector = make_neumann_range_projector(
        n,
        dims,
        device=resolved_device,
        dtype=dtype,
    )
    _synchronize_device(resolved_device)
    start = time.perf_counter()
    result = tt_dgmres_paper(
        operator=problem.operator,
        rhs=problem.rhs,
        config=config,
        reference_solution=problem.reference_solution,
        solution_projector=solution_projector,
    )
    _synchronize_device(resolved_device)
    elapsed_seconds = time.perf_counter() - start

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = save_single_run_plot(result, output_dir / "tt_dgmres_neumann_poisson_plot.png")
    history_path = output_dir / "tt_dgmres_neumann_poisson_history.json"
    history_path.write_text(json.dumps(result.history, indent=2), encoding="utf-8")

    # Keep this driver TT-only. Dense reconstruction is unsafe for larger n^d runs.
    solution_path = None
    reference_solution_path = None
    solution_slices_path = None

    solution_tt_path = output_dir / "tt_dgmres_neumann_poisson_solution_tt.pt"
    reference_solution_tt_path = output_dir / "tt_dgmres_neumann_poisson_reference_solution_tt.pt"
    rhs_tt_path = output_dir / "tt_dgmres_neumann_poisson_rhs_tt.pt"
    torch.save([core.detach().cpu() for core in result.solution.cores], solution_tt_path)
    torch.save([core.detach().cpu() for core in problem.reference_solution.cores], reference_solution_tt_path)
    torch.save([core.detach().cpu() for core in problem.rhs.cores], rhs_tt_path)

    ones = torch.ones(
        n,
        device=resolved_device,
        dtype=dtype,
    )
    local_null_residual = torch.max(torch.abs(problem.local_stencil @ ones)).item()
    solution_ranks = [int(rank) for rank in result.solution.ranks_tt.tolist()]
    reference_ranks = [int(rank) for rank in problem.reference_solution.ranks_tt.tolist()]
    rhs_ranks = [int(rank) for rank in problem.rhs.ranks_tt.tolist()]

    summary = summary_dict(result)
    summary.update(
        {
            "n": n,
            "dims": dims,
            "ambient_dimension": problem.ambient_dimension,
            "operator_name": problem.operator.name,
            "operator_norm_bound": problem.operator_norm_bound,
            "operator_norm_lower_bound": problem.operator_norm_lower_bound,
            "backward_error_semantics": (
                "certified upper bound using the analytic spectral-radius "
                "lower bound in the denominator"
            ),
            "local_stencil_null_residual": local_null_residual,
            "solution_shape": list(result.solution.shape),
            "reference_solution_norm": tt_norm_stable(problem.reference_solution),
            "rhs_norm": tt_norm_stable(problem.rhs),
            "elapsed_seconds": elapsed_seconds,
            "device_requested": str(torch.device(device)),
            "resolved_device": str(resolved_device),
            "pytorch_version": torch.__version__,
            "torch": torch.__version__,
            "cuda_version": torch.version.cuda,
            "torch_cuda": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(resolved_device)
                if resolved_device.type == "cuda"
                else None
            ),
            "gpu": (
                torch.cuda.get_device_name(resolved_device)
                if resolved_device.type == "cuda"
                else None
            ),
            "cuda_peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(resolved_device))
                if resolved_device.type == "cuda"
                else 0
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(resolved_device))
                if resolved_device.type == "cuda"
                else 0
            ),
            "dense_output_disabled": True,
            "dense_solution_npy_saved": False,
            "dense_reference_npy_saved": False,
            "dense_slice_plot_saved": False,
            "solution_tt_ranks": solution_ranks,
            "reference_solution_tt_ranks": reference_ranks,
            "rhs_tt_ranks": rhs_ranks,
            "solution_tt_numcoef": int(result.solution.numcoef()),
            "reference_solution_tt_numcoef": int(problem.reference_solution.numcoef()),
            "rhs_tt_numcoef": int(problem.rhs.numcoef()),
        }
    )

    metadata = {
        "example": {
            "model": "Paper Example 5.1 Neumann-Poisson",
            "matrix_formula": "A = sum_{mu=1}^d I^(otimes(mu-1)) otimes T otimes I^(otimes(d-mu))",
            "local_stencil_definition": "Paper formula (5-5) used directly",
            "consistency_construction": "x_exact = A y, rhs = A x_exact",
        },
        "config": asdict(config),
        "summary": summary,
        "saved_files": {
            "plot_path": str(plot_path),
            "history_path": str(history_path),
            "summary_path": str(output_dir / "tt_dgmres_neumann_poisson_summary.json"),
            "solution_tt_path": str(solution_tt_path),
            "reference_solution_tt_path": str(reference_solution_tt_path),
            "rhs_tt_path": str(rhs_tt_path),
            "solution_path": None,
            "reference_solution_path": None,
            "solution_slices_path": None,
        },
    }

    summary_path = output_dir / "tt_dgmres_neumann_poisson_summary.json"
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "problem": problem,
        "config": config,
        "result": result,
        "summary": summary,
        "plot_path": plot_path,
        "history_path": history_path,
        "summary_path": summary_path,
        "solution_path": solution_path,
        "reference_solution_path": reference_solution_path,
        "solution_tt_path": solution_tt_path,
        "reference_solution_tt_path": reference_solution_tt_path,
        "rhs_tt_path": rhs_tt_path,
        "solution_slices_path": solution_slices_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TT-DGMRES on the paper's Neumann-Poisson Example 5.1 using a direct TT operator."
    )
    parser.add_argument("--n", type=int, default=12, help="Grid points per spatial direction.")
    parser.add_argument("--dims", type=int, default=3, help="Number of spatial dimensions.")
    parser.add_argument("--restart", type=int, default=30, help="Restart length for TT-DGMRES.")
    parser.add_argument("--max-iterations", type=int, default=180, help="Maximum total TT-DGMRES iterations.")
    parser.add_argument("--tolerance", type=float, default=1e-10, help="Stopping tolerance.")
    parser.add_argument("--round-tolerance", type=float, default=1e-10, help="TT rounding tolerance.")
    parser.add_argument(
        "--device",
        type=_device_argument,
        default="cpu",
        help="TT compute device: cpu, cuda, or an indexed CUDA device such as cuda:0.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_tt_dgmres_neumann_poisson",
        help="Directory for plots and json outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = run_experiment(
        n=args.n,
        dims=args.dims,
        restart=args.restart,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        round_tolerance=args.round_tolerance,
        output_dir=args.output_dir,
        device=args.device,
    )

    print("Config:")
    print(artifact["config"])
    print("\nSummary:")
    print(json.dumps(artifact["summary"], indent=2))
    print(f"\nSaved history to {artifact['history_path']}")
    print(f"Saved plot to {artifact['plot_path']}")
    print(f"Saved summary to {artifact['summary_path']}")
    print("Dense solution output disabled.")
    print(f"Saved TT solution cores to {artifact['solution_tt_path']}")
    print(f"Saved TT reference solution cores to {artifact['reference_solution_tt_path']}")
    print(f"Saved TT rhs cores to {artifact['rhs_tt_path']}")


if __name__ == "__main__":
    main()
