"""Regression tests for adaptive control at DGMRES cycle boundaries."""
from dataclasses import replace

import numpy as np
import pytest
import torch

from rl_tt_dgmres_v2 import solver
from adaptive_tt_dgmres.solver import tt_dgmres_adaptive as legacy_solver
from adaptive_tt_dgmres.backend.paper_tt_dgmres import _stable_tt_norm, tt_round_capped
from adaptive_tt_dgmres.backend.paper_tt_dgmres import TTDGMRESConfig
from adaptive_tt_dgmres.backend.paper_tt_gmres import tt_matrix_operator, tt_rank1, tt_zeros
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
        reorthogonalization="always",
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


def one_site_problem(matrix, rhs, **changes):
    matrix = np.asarray(matrix, dtype=float)
    core = torch.as_tensor(matrix, dtype=torch.float64)
    operator_norm = float(np.linalg.norm(matrix, 2))
    operator = tt_matrix_operator(
        [core.reshape(1, *matrix.shape, 1)], exact_norm=operator_norm,
    )
    vector = tt_rank1([torch.as_tensor(rhs, dtype=torch.float64)])
    cfg = TTDGMRESConfig(
        index=0, restart=matrix.shape[0], max_iterations=matrix.shape[0],
        tolerance=1e-12, round_tolerance=1e-14, operator_norm=operator_norm,
        stopping_metric="relative_a_residual", safe_orth_rounding=True,
    )
    return operator, vector, replace(cfg, **changes)


@pytest.mark.parametrize("safe", [False, True])
def test_dgks_skips_orthogonal_images_and_repeats_dependent_image(safe):
    # Orthogonal cyclic images require one pass until the last Krylov step.
    matrix = np.roll(np.eye(4), 1, axis=0)
    operator, rhs, cfg = one_site_problem(matrix, [1, 0, 0, 0], safe_orth_rounding=safe)
    events, cycles = [], []
    selective = solver.tt_dgmres_adaptive(
        operator=operator, rhs=rhs, config=cfg, iteration_observer=events.append,
        cycle_observer=cycles.append,
    )
    always = solver.tt_dgmres_adaptive(
        operator=operator, rhs=rhs, config=cfg, reorthogonalization="always",
    )
    np.testing.assert_allclose(selective.solution.torch().numpy(), [0, 0, 0, 1], atol=1e-13)
    assert selective.converged and always.converged
    assert [event["orthogonalization_passes"] for event in events] == [1, 1, 1, 2]
    assert [event["reorthogonalized"] for event in events] == [False, False, False, True]
    assert selective.history["algorithm_inner_products"][-1] == 14
    assert always.history["algorithm_inner_products"][-1] == 20
    assert selective.history["reorthogonalization_selected"][-1] == 1
    assert selective.history["orthogonalization_passes"][-1] == 5
    assert cycles[-1]["cycle_reorthogonalization_selected"] == 1
    assert cycles[-1]["cycle_orthogonalization_passes"] == 5
    assert all(event["orthogonalization_seconds"] >= 0 for event in events)
    assert cycles[-1]["algorithm_round_calls"] == sum(cycles[-1]["round_calls_by_phase"].values())


@pytest.mark.parametrize("mode", ["selective", "always", "never"])
def test_singular_drazin_solution_matches_dense_reference(mode):
    # This rank-three operator has an exact one-dimensional invariant Krylov
    # space, exercising the inherited certified lucky-breakdown branch.
    matrix = np.diag([0.0, 2.0, 2.0, 2.0])
    dense_rhs = np.array([3.0, 1.0, -1.0, 2.0])
    operator, rhs, cfg = one_site_problem(matrix, dense_rhs, index=1)
    result = solver.tt_dgmres_adaptive(
        operator=operator, rhs=rhs, config=cfg, reorthogonalization=mode,
        diagnostic_interval=1,
    )
    actual = result.solution.torch().numpy()
    np.testing.assert_allclose(actual, [0.0, 0.5, -0.5, 1.0], atol=2e-11)
    assert result.converged
    assert np.linalg.norm(matrix @ (dense_rhs - matrix @ actual)) < 1e-10
    assert result.history["diagnostic_metric_valid_for_stopping"][-1] == 1
    if mode == "never":
        assert result.history["reorthogonalization_selected"][-1] == 0


@pytest.mark.parametrize("mode", ["selective", "always", "never"])
def test_nonsymmetric_dense_solution_matches_direct_solve(mode):
    matrix = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, -0.5], [0.25, 0.0, 4.0]])
    dense_rhs = np.array([1.0, -2.0, 0.5])
    operator, rhs, cfg = one_site_problem(matrix, dense_rhs)
    result = solver.tt_dgmres_adaptive(
        operator=operator, rhs=rhs, config=cfg, reorthogonalization=mode,
    )
    np.testing.assert_allclose(result.solution.torch().numpy(),
                               np.linalg.solve(matrix, dense_rhs), atol=2e-11)
    assert result.converged


@pytest.mark.parametrize("mode", ["selective", "always", "never"])
def test_inherited_k_greater_than_index_breakdown_is_not_false_convergence(mode):
    # At k=3 > index=1 the archived DGMRES backend uses k-index correction
    # vectors even though the three-dimensional invariant space is exhausted.
    # Preserve this conservative candidate and independent failure status. A
    # caller can resume a further cycle from it; the acceleration must not
    # silently introduce a different projected algorithm or claim success.
    operator, rhs, cfg = one_site_problem(np.diag([0.0, 1.0, 2.0, 4.0]),
                                         [3.0, 1.0, -1.0, 2.0], index=1)
    baseline = legacy_solver(operator=operator, rhs=rhs, config=cfg, diagnostic_interval=1)
    result = solver.tt_dgmres_adaptive(
        operator=operator, rhs=rhs, config=cfg, reorthogonalization=mode,
        diagnostic_interval=1,
    )
    assert result.breakdown and baseline.breakdown
    assert not result.converged and not baseline.converged
    assert result.iterations == baseline.iterations == 3
    np.testing.assert_allclose(result.solution.torch().numpy(),
                               baseline.solution.torch().numpy(), atol=2e-11)
    assert result.history["relative_transformed_residual_norm"][-1] > cfg.tolerance


def test_legacy_reorthogonalize_flag_still_disables_second_pass():
    operator, rhs, cfg = one_site_problem(np.eye(2), [1.0, 0.0], reorthogonalize=False)
    events = []
    result = solver.tt_dgmres_adaptive(
        operator=operator, rhs=rhs, config=cfg, reorthogonalization="always",
        iteration_observer=events.append,
    )
    assert result.converged
    assert events[0]["orthogonalization_passes"] == 1


@pytest.mark.parametrize("threshold,expected", [(0.4, False), (0.8, True)])
def test_dgks_threshold_compares_first_pass_and_image_norms(threshold, expected):
    matrix = np.array([[np.sqrt(0.75), -0.5], [0.5, np.sqrt(0.75)]])
    operator, rhs, cfg = one_site_problem(matrix, [1.0, 0.0], max_iterations=1)
    events = []
    solver.tt_dgmres_adaptive(
        operator=operator, rhs=rhs, config=cfg, reorth_threshold=threshold,
        iteration_observer=events.append,
    )
    # ||A v|| = 1 and ||(I-vv*) A v|| = 0.5 at the first Arnoldi step.
    assert events[0]["reorthogonalized"] is expected


@pytest.mark.parametrize("kwargs", [
    {"reorthogonalization": "sometimes"}, {"reorth_threshold": 0},
    {"reorth_threshold": 1}, {"reorth_threshold": float("nan")},
    {"reorth_threshold": True},
])
def test_invalid_reorthogonalization_controls_are_rejected(problem, kwargs):
    with pytest.raises(ValueError):
        solver.tt_dgmres_adaptive(
            operator=problem.operator, rhs=problem.rhs, config=config(problem), **kwargs,
        )


def test_callback_failure_preserves_last_completed_cycle(problem):
    checkpoints = []

    def controller(state):
        checkpoints.append(state["x"].clone())
        if state["cycle"] == 2:
            raise RuntimeError("injected controller failure")
        return {}

    with pytest.raises(RuntimeError, match="injected controller failure"):
        solver.tt_dgmres_adaptive(
            operator=problem.operator, rhs=problem.rhs,
            config=config(problem, max_iterations=8), diagnostic_interval=50,
            cycle_controller=controller,
        )
    first_cycle = solver.tt_dgmres_adaptive(
        operator=problem.operator, rhs=problem.rhs,
        config=config(problem, max_iterations=4), diagnostic_interval=50,
    )
    assert len(checkpoints) == 2
    assert _stable_tt_norm(checkpoints[-1] - first_cycle.solution) < 1e-11
    resumed = solver.tt_dgmres_adaptive(
        operator=problem.operator, rhs=problem.rhs,
        config=config(problem, max_iterations=4), diagnostic_interval=50,
        initial_solution=checkpoints[-1],
    )
    continuous = solver.tt_dgmres_adaptive(
        operator=problem.operator, rhs=problem.rhs,
        config=config(problem, max_iterations=8), diagnostic_interval=50,
    )
    assert _stable_tt_norm(resumed.solution - continuous.solution) < 1e-9


def test_tree_scaled_leaves_avoid_redundant_rounds(problem):
    phases = []

    def rounder(tensor, eps, max_rank, phase):
        phases.append(phase)
        return tt_round_capped(tensor, eps, max_rank)

    result = solver.tt_dgmres_adaptive(
        operator=problem.operator, rhs=problem.rhs,
        config=config(problem, max_iterations=4, max_tt_rank=5, tree_solution_combination=True),
        algorithm_rounder=rounder,
    )
    assert result.history["combination_leaf_rounds_skipped"][-1] > 0
    assert "combination_leaf" not in phases
    assert "combination_tree" in phases
    assert result.history["algorithm_round_calls"][-1] == len(phases)


def test_projected_zero_does_not_bypass_independent_residual_audit():
    operator, rhs, cfg = one_site_problem(np.diag([1.0, 2.0]), [1.0, 1.0])

    def rounder(tensor, eps, max_rank, phase):
        if phase == "matvec":
            return tt_zeros(tensor.shape, like=tensor)
        return tt_round_capped(tensor, eps, max_rank)

    result = solver.tt_dgmres_adaptive(
        operator=operator, rhs=rhs, config=cfg, algorithm_rounder=rounder,
    )
    assert not result.converged
    assert result.history["relative_transformed_residual_norm"][-1] > 0.9


def test_projected_check_factor_adds_audits_without_relaxing_stopping():
    matrix = np.array([[np.sqrt(0.75), -0.5], [0.5, np.sqrt(0.75)]])
    operator, rhs, cfg = one_site_problem(matrix, [1.0, 0.0], tolerance=0.1)
    original_events, extra_events = [], []
    original = solver.tt_dgmres_adaptive(
        operator=operator, rhs=rhs, config=cfg, diagnostic_interval=50,
        candidate_observer=original_events.append,
    )
    extra = solver.tt_dgmres_adaptive(
        operator=operator, rhs=rhs, config=cfg, diagnostic_interval=50,
        candidate_observer=extra_events.append, projected_check_factor=10,
    )
    assert [event["total_iterations"] for event in original_events] == [2]
    assert [event["total_iterations"] for event in extra_events] == [1, 2]
    assert original.iterations == extra.iterations == 2
    assert original.status == extra.status
    np.testing.assert_allclose(original.solution.torch().numpy(), extra.solution.torch().numpy())
    first = extra_events[0]
    assert first["metric_valid_for_stopping"]
    assert first["metrics"]["transformed_residual_norm"] > cfg.tolerance
    assert "reference_solution" not in first
    assert "relative_solution_error" not in first["metrics"]


def test_candidate_observer_can_interrupt_with_independently_audited_candidate():
    operator, rhs, cfg = one_site_problem(np.eye(2), [1.0, 0.0])
    candidates = []

    def stop_on_candidate(context):
        candidates.append(context)
        raise RuntimeError("caller accepted independently checked candidate")

    with pytest.raises(RuntimeError, match="caller accepted"):
        solver.tt_dgmres_adaptive(operator=operator, rhs=rhs, config=cfg,
                                   candidate_observer=stop_on_candidate)
    assert candidates[0]["metrics"]["transformed_residual_norm"] < cfg.tolerance
    np.testing.assert_allclose(candidates[0]["x"].torch().numpy(), [1.0, 0.0])


@pytest.mark.parametrize("factor", [0, -1, float("nan"), True])
def test_invalid_projected_check_factor_rejected(problem, factor):
    with pytest.raises(ValueError):
        solver.tt_dgmres_adaptive(operator=problem.operator, rhs=problem.rhs,
                                   config=config(problem), projected_check_factor=factor)


@pytest.mark.parametrize("mode", ["selective", "always", "never"])
def test_arnoldi_budget_preserves_dense_solution_and_records_local_budgets(mode):
    from rl_tt_dgmres_v2.rounding import FastRounder

    matrix = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, -0.5], [0.25, 0.0, 4.0]])
    dense_rhs = np.array([1.0, -2.0, 0.5])
    operator, rhs, cfg = one_site_problem(matrix, dense_rhs)
    rounder = FastRounder()
    events = []
    result = solver.tt_dgmres_adaptive(
        operator=operator, rhs=rhs, config=cfg, reorthogonalization=mode,
        algorithm_rounder=rounder, orth_error_mode="arnoldi_budget",
        iteration_observer=events.append,
    )
    np.testing.assert_allclose(result.solution.torch().numpy(),
                               np.linalg.solve(matrix, dense_rhs), atol=2e-11)
    assert result.converged
    orth_records = [row for row in rounder.logs if row["phase"] == "orth"]
    assert orth_records and all(row["budget_mode"] == "absolute" for row in orth_records)
    assert all(row["within_requested_absolute"] for row in orth_records)
    for event in events:
        assert event["orth_local_absolute_budget"] == pytest.approx(
            event["orth_column_absolute_budget"] / (2 * (event["local_k"] + 1)))
        assert event["orth_audited_absolute_error_sum"] <= event["orth_column_absolute_budget"]
    assert any(event["inexact_breakdown"] for event in events)


def test_budget_only_breakdown_cannot_claim_an_exact_invariant_space(monkeypatch):
    matrix = np.array([[np.sqrt(0.75), -0.5], [0.5, np.sqrt(0.75)]])
    operator, rhs, cfg = one_site_problem(matrix, [1.0, 0.0], index=1,
                                         round_tolerance=0.1, orth_round_tolerance=0.5)

    def forbidden_certification(*args, **kwargs):
        raise AssertionError("an inexact noise threshold is not invariant-space evidence")

    monkeypatch.setattr(solver, "_arnoldi_invariance_is_certified", forbidden_certification)
    result = solver.tt_dgmres_adaptive(operator=operator, rhs=rhs, config=cfg,
                                       orth_error_mode="arnoldi_budget")
    assert result.status == "inexact_arnoldi_breakdown"
    assert result.breakdown and not result.converged
    assert result.iterations == 1
    assert result.history["relative_transformed_residual_norm"][-1] > 0.9


def test_exact_certified_lucky_branch_survives_arnoldi_budget_mode():
    operator, rhs, cfg = one_site_problem(np.diag([0.0, 2.0]), [1.0, 1.0], index=1)
    result = solver.tt_dgmres_adaptive(operator=operator, rhs=rhs, config=cfg,
                                       orth_error_mode="arnoldi_budget")
    assert result.converged
    np.testing.assert_allclose(result.solution.torch().numpy(), [0.0, 0.5], atol=1e-12)


def test_direct_breakdown_candidate_matches_true_one_dimensional_minimizer():
    matrix = np.diag([0.0, 1.0, 1.01])
    dense_rhs = np.array([0.2, 1.0, 0.1])
    operator, rhs, cfg = one_site_problem(matrix, dense_rhs, index=1, round_tolerance=0.1)
    events, candidates = [], []
    result = solver.tt_dgmres_adaptive(
        operator=operator, rhs=rhs, config=cfg, orth_error_mode="arnoldi_budget",
        direct_breakdown_correction=True, iteration_observer=events.append,
        candidate_observer=candidates.append,
    )
    direction = matrix @ dense_rhs
    direction /= np.linalg.norm(direction)
    z = matrix @ matrix @ direction
    expected_y = (z @ (matrix @ dense_rhs)) / (z @ z)
    np.testing.assert_allclose(result.solution.torch().numpy(), expected_y * direction, atol=1e-12)
    assert result.status == "inexact_breakdown_with_direct_correction"
    assert not result.converged
    assert events[0]["direct_correction_used"]
    assert events[0]["direct_breakdown_extra_matvecs"] == 4
    assert candidates[0]["candidate_kind"] == "direct_index_one_residual_minimizer"
    assert result.history["direct_breakdown_corrections"][-1] == 1
    assert result.history["direct_breakdown_matvecs"][-1] == 4
    actual_transformed = matrix @ (dense_rhs - matrix @ result.solution.torch().numpy())
    assert abs(z @ actual_transformed) < 1e-12


def test_direct_breakdown_minimizer_uses_unrounded_true_residual():
    matrix = np.array([[2.0, 0.4], [0.0, 3.0]])
    dense_rhs = np.array([1.0, -2.0])
    operator, rhs, _ = one_site_problem(matrix, dense_rhs)
    x0 = np.array([0.25, -0.1])
    direction = np.array([0.6, 0.8])
    record = solver._direct_index_one_correction(
        operator, rhs, tt_rank1([torch.as_tensor(x0)]),
        tt_rank1([torch.as_tensor(direction)]),
    )
    z = matrix @ matrix @ direction
    g = matrix @ (dense_rhs - matrix @ x0)
    assert record["coefficient"] == pytest.approx((z @ g) / (z @ z), abs=1e-13)
    assert record["projected_residual"] == pytest.approx(
        np.linalg.norm(g - record["coefficient"] * z), abs=1e-13)
    assert record["matvecs"] == 4


def test_direct_correction_rejects_zero_operator_image():
    operator, rhs, _ = one_site_problem(np.diag([0.0, 1.0]), [1.0, 1.0])
    record = solver._direct_index_one_correction(
        operator, rhs, tt_zeros(rhs.shape, like=rhs),
        tt_rank1([torch.tensor([1.0, 0.0])]),
    )
    assert record["status"] == "zero_or_nonfinite_image"
    assert record["coefficient"] is None
    assert record["matvecs"] == 2
