"""Regression tests for adaptive control at DGMRES cycle boundaries."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from adaptive_tt_dgmres import solver
from adaptive_tt_dgmres.backend.paper_tt_dgmres import _stable_tt_norm, tt_round_capped
from adaptive_tt_dgmres.backend.tt_dgmres_neumann_poisson_example import (
    build_neumann_poisson_problem,
    default_config,
)
from adaptive_tt_dgmres.backend.tt_dgmres_seventh_fast_restart_sweep import (
    tt_dgmres_sparse_diagnostics,
)


@pytest.fixture(scope="module")
def problem():
    torch.set_num_threads(1)
    return build_neumann_poisson_problem(n=5, dims=2)


def config(problem, **changes):
    base = default_config(
        problem.operator_norm_lower_bound, restart=4, max_iterations=9,
        tolerance=1e-30, round_tolerance=1e-12,
    )
    return replace(base, **changes)


@pytest.mark.parametrize("tree,safe,distinct_eps", [
    (False, False, False), (True, True, False), (True, True, True),
])
def test_constant_policy_preserves_baseline(problem, tree, safe, distinct_eps):
    cfg = config(problem, tree_solution_combination=tree, safe_orth_rounding=safe)
    if distinct_eps:
        cfg = replace(cfg, round_tolerance=1e-8, orth_round_tolerance=1e-9,
                      solution_round_tolerance=1e-10, residual_round_tolerance=1e-12,
                      transformed_round_tolerance=1e-12)
    kwargs = dict(operator=problem.operator, rhs=problem.rhs, config=cfg,
                  diagnostic_interval=3, reference_solution=problem.reference_solution)
    baseline = tt_dgmres_sparse_diagnostics(**kwargs)
    phases = []

    def rounder(tensor, eps, max_rank, phase):
        phases.append(phase)
        return tt_round_capped(tensor, eps, max_rank)

    def constant(state):
        assert "reference_solution" not in state
        assert "relative_solution_error" not in state["metrics"]
        return {"restart": cfg.restart, "round_tolerance": cfg.round_tolerance,
                "orth_round_tolerance": cfg.resolved().orth_round_tolerance,
                "solution_round_tolerance": cfg.resolved().solution_round_tolerance}

    adaptive = solver.tt_dgmres_adaptive(
        **kwargs, cycle_controller=constant, algorithm_rounder=rounder,
    )
    assert adaptive.iterations == baseline.iterations
    assert adaptive.cycles == baseline.cycles
    assert adaptive.status == baseline.status
    error = _stable_tt_norm(adaptive.solution - baseline.solution)
    assert error <= 1e-11 * max(_stable_tt_norm(baseline.solution), 1.0)
    for metric in ("backward_error", "transformed_residual_norm",
                   "projected_transformed_residual_norm"):
        np.testing.assert_allclose(adaptive.history[metric], baseline.history[metric],
                                   rtol=1e-9, atol=1e-11)
    assert {"initial_residual", "initial_power", "cycle_residual", "cycle_power",
            "matvec", "orth", "solution"}.issubset(phases)
    assert ("combination_tree" if tree else "combination") in phases


def test_smaller_dynamic_restart_does_not_hit_old_cycle_cap(problem):
    events = []
    result = solver.tt_dgmres_adaptive(
        operator=problem.operator, rhs=problem.rhs,
        config=config(problem, restart=9, max_iterations=11), diagnostic_interval=50,
        cycle_controller=lambda state: {"restart": 2 if state["cycle"] % 2 else 3},
        cycle_observer=events.append,
    )
    assert result.iterations == 11
    assert [event["restart"] for event in events] == [2, 3, 2, 3, 2]
    assert [event["cycle_iterations"] for event in events] == [2, 3, 2, 3, 1]
    assert result.cycles == 5
    assert all(event["elapsed_seconds"] >= 0 for event in events)


def test_diagnostics_do_not_inherit_policy_tolerances(problem, monkeypatch):
    audit_eps = []
    original = solver._diagnostic_residual_metrics

    def inspect_audit(**kwargs):
        audit_eps.append((kwargs["residual_round_eps"], kwargs["transformed_round_eps"]))
        return original(**kwargs)

    monkeypatch.setattr(solver, "_diagnostic_residual_metrics", inspect_audit)
    solver.tt_dgmres_adaptive(
        operator=problem.operator, rhs=problem.rhs,
        config=config(problem, max_iterations=4), diagnostic_interval=2,
        cycle_controller=lambda state: {"restart": 2, "round_tolerance": 1e-5},
    )
    assert len(audit_eps) >= 3
    assert set(audit_eps) == {(1e-12, 1e-12)}


def test_explicit_diagnostic_tolerances_do_not_change_algorithm_rounding(problem, monkeypatch):
    audit_eps, algorithm_eps = [], []
    original = solver._diagnostic_residual_metrics

    def inspect_audit(**kwargs):
        audit_eps.append((kwargs["residual_round_eps"], kwargs["transformed_round_eps"]))
        return original(**kwargs)

    def inspect_round(tensor, eps, max_rank, phase):
        algorithm_eps.append((phase, eps))
        return tt_round_capped(tensor, eps, max_rank)

    monkeypatch.setattr(solver, "_diagnostic_residual_metrics", inspect_audit)
    solver.tt_dgmres_adaptive(
        operator=problem.operator, rhs=problem.rhs,
        config=config(problem, max_iterations=4, round_tolerance=1e-8),
        diagnostic_interval=2, diagnostic_residual_round_tolerance=1e-12,
        diagnostic_transformed_round_tolerance=1e-13,
        cycle_controller=lambda state: {"restart": 2}, algorithm_rounder=inspect_round,
    )
    assert len(audit_eps) >= 3
    assert set(audit_eps) == {(1e-12, 1e-13)}
    assert {phase for phase, _ in algorithm_eps}.issuperset(
        {"initial_residual", "initial_power", "cycle_residual", "cycle_power"})
    assert {eps for _, eps in algorithm_eps} == {1e-8}


def test_initial_solution_can_resume_a_cycle(problem):
    first = solver.tt_dgmres_adaptive(
        operator=problem.operator, rhs=problem.rhs,
        config=config(problem, max_iterations=4), diagnostic_interval=50,
    )
    resumed = solver.tt_dgmres_adaptive(
        operator=problem.operator, rhs=problem.rhs,
        config=config(problem, max_iterations=4), diagnostic_interval=50,
        initial_solution=first.solution,
    )
    continuous = solver.tt_dgmres_adaptive(
        operator=problem.operator, rhs=problem.rhs,
        config=config(problem, max_iterations=8), diagnostic_interval=50,
    )
    assert _stable_tt_norm(resumed.solution - continuous.solution) < 1e-9


@pytest.mark.parametrize("action", [
    {"restart": 1}, {"restart": 2.5}, {"round_tolerance": float("nan")},
    {"orth_round_tolerance": 0}, {"solution_round_tolerance": -1},
])
def test_invalid_policy_actions_fail_explicitly(problem, action):
    with pytest.raises(ValueError):
        solver.tt_dgmres_adaptive(
            operator=problem.operator, rhs=problem.rhs, config=config(problem),
            cycle_controller=lambda state: action,
        )
