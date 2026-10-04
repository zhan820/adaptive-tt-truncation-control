from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .backend.paper_tt_dgmres import (
    TTDGMRESConfig, _stable_tt_norm, variable_kron_sum_tt_operator,
)
from .backend.paper_tt_gmres import tt_randn, tt_round, tt_scale
from .backend.tt_dgmres_neumann_poisson_example import (
    make_neumann_range_projector, neumann_null_vectors_tt,
    neumann_range_audit_tt, paper_neumann_stencil_1d,
)


@dataclass(frozen=True)
class ProblemSpec:
    name: str
    n: int
    dims: int
    seed: int
    anisotropy: float = 1.0


@dataclass
class Problem:
    spec: ProblemSpec
    operator: Any
    rhs: Any
    reference_solution: Any
    projector: Any
    right_null: Any
    left_null: Any
    norm_lower: float
    rhs_norm: float


def build_problem(spec: ProblemSpec) -> Problem:
    """Index-one, possibly anisotropic Neumann sum with a manufactured range solution.

    No full tensor is formed. Reference solutions belong only to evaluation.
    Positive axis weights preserve the known left/right nullspaces.
    """
    if spec.n < 3 or spec.dims < 2 or spec.anisotropy <= 0:
        raise ValueError("Require n >= 3, dims >= 2 and positive anisotropy.")
    weights = torch.logspace(0, torch.log10(torch.tensor(spec.anisotropy)).item(),
                             spec.dims, dtype=torch.float64)
    local = paper_neumann_stencil_1d(spec.n, dtype=torch.float64)
    operator = variable_kron_sum_tt_operator(
        [float(w) * local for w in weights], name=spec.name, dtype=torch.float64,
    )
    with torch.random.fork_rng():
        torch.manual_seed(spec.seed)
        seed_tensor = tt_randn([spec.n] * spec.dims, ranks_tt=2, dtype=torch.float64)
    projector = make_neumann_range_projector(spec.n, spec.dims)
    reference = projector(tt_round(operator.matvec(seed_tensor), 1e-13))
    reference = tt_scale(reference, 1.0 / _stable_tt_norm(reference))
    rhs = tt_round(operator.matvec(reference), 1e-13)
    right_null, left_null = neumann_null_vectors_tt(spec.n, spec.dims)
    return Problem(spec, operator, rhs, reference, projector, right_null, left_null,
                   4.0 * (spec.n - 1) ** 2 * float(weights.sum()), _stable_tt_norm(rhs))


def make_config(problem: Problem, *, restart: int = 8, max_iterations: int = 180,
                tolerance: float = 1e-6, round_eps: float = 1e-8) -> TTDGMRESConfig:
    return TTDGMRESConfig(
        index=1, restart=restart, max_iterations=max_iterations,
        max_cycles=max_iterations, tolerance=tolerance, round_tolerance=round_eps,
        orth_round_tolerance=round_eps, solution_round_tolerance=round_eps,
        residual_round_tolerance=round_eps, transformed_round_tolerance=round_eps,
        operator_norm=problem.norm_lower, max_tt_rank=48,
        safe_orth_rounding=True, tree_solution_combination=True,
        reorthogonalize=True, stopping_metric="relative_true_residual",
    )


def final_audit(problem: Problem, solution: Any, *, include_reference: bool = True) -> dict:
    """Unrounded TT residual plus explicit range check; no learned stopping test."""
    residual = problem.rhs - problem.operator.matvec(solution)
    residual_norm = _stable_tt_norm(residual)
    xnorm = _stable_tt_norm(solution)
    transformed = problem.operator.matvec(residual)
    range_metrics = neumann_range_audit_tt(
        solution=solution, right_null=problem.right_null, left_null=problem.left_null,
    )
    values = {
        "relative_true_residual": residual_norm / problem.rhs_norm,
        "backward_error_upper": residual_norm / (problem.norm_lower * xnorm + problem.rhs_norm),
        "transformed_residual_norm": _stable_tt_norm(transformed),
        **range_metrics,
    }
    if include_reference:
        values["relative_solution_error_evaluation_only"] = _stable_tt_norm(
            solution - problem.reference_solution) / _stable_tt_norm(problem.reference_solution)
    return values


def default_splits() -> dict[str, list[ProblemSpec]]:
    # Operators, not neighboring trajectory rows, determine each partition.
    return {
        "train": [ProblemSpec("train_n5_d2_a1", 5, 2, 11, 1.0),
                  ProblemSpec("train_n6_d2_a2", 6, 2, 17, 2.0),
                  ProblemSpec("train_n5_d3_a2", 5, 3, 23, 2.0)],
        "validation": [ProblemSpec("validation_n6_d3_a1_5", 6, 3, 31, 1.5)],
        "test": [ProblemSpec("test_n7_d3_a1_25", 7, 3, 101, 1.25),
                 ProblemSpec("test_n5_d4_a1_75", 5, 4, 103, 1.75)],
    }
