"""Deterministic controller accounting tests; no training or timing benchmarks."""
from types import SimpleNamespace

import pytest
import torch

from adaptive_tt_dgmres.backend.paper_tt_dgmres import tt_norm_stable
from adaptive_tt_dgmres.backend.paper_tt_gmres import tt_scale
from rl_tt_dgmres_v2 import runner
from rl_tt_dgmres_v2.problems import ProblemSpec, audit_solution, build_problem


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def small_problem():
    return build_problem(ProblemSpec("accounting", 3, 2, 13, rhs_rank=1))


def context(x, transformed, iteration):
    return {
        "x": x, "total_iterations": iteration,
        "metrics": {"transformed_residual_norm": transformed,
                    "transformed_residual_certified": True},
        "projected_residual": transformed,
    }


def step(kwargs, iteration):
    kwargs["iteration_observer"]({
        "iteration": iteration, "basis_tt_core_bytes": 128 * iteration,
        "orthogonalization_passes": 2, "reorthogonalized": True,
    })


def install_clock_and_audit(monkeypatch, audit_cost):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock.now)

    def independent(problem, x, include_reference=False):
        clock.now += audit_cost
        assert include_reference is False
        return audit_solution(problem, x, include_reference=False)

    monkeypatch.setattr(runner, "audit_solution", independent)
    return clock


def test_retry_restores_checked_checkpoint_and_charges_rejected_work(monkeypatch, small_problem):
    problem = small_problem
    clock = install_clock_and_audit(monkeypatch, audit_cost=2.0)
    attempts = []
    checkpoint = tt_scale(problem.reference_solution, 0.25)

    def fake_solver(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            kwargs["cycle_controller"](context(kwargs["initial_solution"], problem.ab_norm, 0))
            clock.now += 2.0
            step(kwargs, 2)
            kwargs["cycle_observer"](context(checkpoint, 0.75 * problem.ab_norm, 2))
            kwargs["cycle_controller"](context(checkpoint, 0.75 * problem.ab_norm, 2))
            clock.now += 3.0
            step(kwargs, 5)
            kwargs["cycle_observer"](context(
                tt_scale(problem.reference_solution, 0.8), 3.0 * problem.ab_norm, 5,
            ))
            pytest.fail("Residual growth must reject this cycle")
        assert tt_norm_stable(kwargs["initial_solution"] - checkpoint) < 1e-14
        assert kwargs["config"].max_iterations == 15
        assert kwargs["config"].max_tt_rank == 192
        controls = kwargs["cycle_controller"](context(checkpoint, 0.75 * problem.ab_norm, 0))
        assert controls == {
            "restart": runner.RECOVERY_ACTION.update_dim + 1,
            "round_tolerance": runner.RECOVERY_ACTION.matvec_eps,
            "orth_round_tolerance": runner.RECOVERY_ACTION.orth_eps,
            "solution_round_tolerance": runner.RECOVERY_ACTION.solution_eps,
        }
        clock.now += 4.0
        step(kwargs, 3)
        kwargs["cycle_observer"](context(problem.reference_solution, 0.0, 3))
        pytest.fail("The checked target must terminate the episode")

    monkeypatch.setattr(runner, "tt_dgmres_adaptive", fake_solver)
    result = runner.run_episode(
        problem, runner.EpisodeConfig(max_iterations=20), lambda obs: 2,
        evaluate_reference=False,
    )
    summary = result["summary"]
    assert summary["accepted"]
    assert summary["iterations"] == 8
    assert summary["recoveries"] == 1
    assert summary["wall_seconds"] == pytest.approx(11.0)
    assert summary["final_audit_seconds"] == pytest.approx(2.0)
    assert summary["reward_sum"] == pytest.approx(-11.0 / 30.0)
    assert [cycle["completed"] for cycle in result["cycles"]] == [True, False, True]
    assert result["cycles"][1]["rejection"] == "residual_growth"
    assert sum(cycle["seconds"] for cycle in result["cycles"]) == pytest.approx(11.0)
    assert [transition["done"] for transition in result["transitions"]] == [False, False, True]
    assert summary["executed_action_indices"][-1] == -1
    assert result["recoveries"][0]["fallback_action"] == -1


def test_final_independent_audit_can_exhaust_budget_and_incurs_failure_penalty(monkeypatch, small_problem):
    problem = small_problem
    clock = install_clock_and_audit(monkeypatch, audit_cost=4.0)

    def fake_solver(**kwargs):
        kwargs["cycle_controller"](context(kwargs["initial_solution"], problem.ab_norm, 0))
        clock.now += 2.0
        step(kwargs, 2)
        kwargs["cycle_observer"](context(problem.reference_solution, 0.0, 2))

    monkeypatch.setattr(runner, "tt_dgmres_adaptive", fake_solver)
    config = runner.EpisodeConfig(seconds_budget=5.0, failure_penalty=7.0)
    result = runner.run_episode(problem, config, lambda obs: 3, evaluate_reference=False)
    assert not result["summary"]["accepted"]
    assert result["summary"]["status"] == "time_budget"
    assert result["summary"]["eta"] < config.tolerance
    assert result["summary"]["wall_seconds"] == pytest.approx(6.0)
    assert result["summary"]["final_audit_seconds"] == pytest.approx(4.0)
    assert result["summary"]["reward_sum"] == pytest.approx(-6.0 / 5.0 - 7.0)
    assert result["cycles"][0]["seconds"] == pytest.approx(6.0)
    assert result["transitions"][-1]["done"]


def test_recovery_limit_charges_all_attempts_and_keeps_initial_checkpoint(monkeypatch, small_problem):
    problem = small_problem
    clock = install_clock_and_audit(monkeypatch, audit_cost=0.5)
    attempts = []

    def fake_solver(**kwargs):
        attempts.append(kwargs)
        assert tt_norm_stable(kwargs["initial_solution"]) == 0.0
        kwargs["cycle_controller"](context(kwargs["initial_solution"], problem.ab_norm, 0))
        clock.now += 1.5
        step(kwargs, 1)
        raise runner.RoundFailure("synthetic rank-cap rejection", {})

    monkeypatch.setattr(runner, "tt_dgmres_adaptive", fake_solver)
    config = runner.EpisodeConfig(max_recoveries=1, failure_penalty=9.0)
    result = runner.run_episode(problem, config, lambda obs: 1, evaluate_reference=False)
    assert len(attempts) == 2
    assert not result["summary"]["accepted"]
    assert result["summary"]["iterations"] == 2
    assert result["summary"]["recoveries"] == 1
    assert result["summary"]["wall_seconds"] == pytest.approx(3.5)
    assert result["summary"]["reward_sum"] == pytest.approx(-3.5 / 30.0 - 9.0)
    assert sum(row["seconds"] for row in result["cycles"]) == pytest.approx(3.5)
    assert all(not row["completed"] for row in result["cycles"])


@pytest.mark.parametrize("action_index", range(len(runner.ACTIONS)))
def test_joint_profile_reaches_solver_and_policy_receives_only_numeric_state(monkeypatch, small_problem, action_index):
    problem = small_problem
    clock = install_clock_and_audit(monkeypatch, audit_cost=0.25)
    observations = []

    def policy(observation):
        observations.append(observation)
        assert len(observation) == len(runner.FEATURE_NAMES)
        assert all(isinstance(value, float) for value in observation)
        return action_index

    def fake_solver(**kwargs):
        assert "reference_solution" not in kwargs
        assert kwargs["orth_error_mode"] == "arnoldi_budget"
        assert callable(kwargs["candidate_observer"])
        controls = kwargs["cycle_controller"](context(kwargs["initial_solution"], problem.ab_norm, 0))
        expected = runner.ACTIONS[action_index]
        assert controls == {
            "restart": expected.update_dim + 1,
            "round_tolerance": expected.matvec_eps,
            "orth_round_tolerance": expected.orth_eps,
            "solution_round_tolerance": expected.solution_eps,
        }
        clock.now += 0.5
        step(kwargs, 2)
        kwargs["cycle_observer"](context(problem.reference_solution, 0.0, 2))

    class ReferenceGuard:
        spec = problem.spec

        def online_view(self):
            return problem.online_view()

        @property
        def reference_solution(self):
            raise AssertionError("Training must never read the reference solution")

        def __getattr__(self, name):
            return getattr(problem, name)

    monkeypatch.setattr(runner, "tt_dgmres_adaptive", fake_solver)
    result = runner.run_episode(
        ReferenceGuard(), runner.EpisodeConfig(safety_filter=False), policy,
        evaluate_reference=False,
    )
    assert len(observations) == 1
    assert result["summary"]["accepted"]
    assert result["summary"]["forward_error"] is None
    assert result["summary"]["executed_action_indices"] == [action_index]


@pytest.mark.parametrize("orth_error_mode", ["relative", "arnoldi_budget"])
def test_candidate_check_can_stop_early_without_changing_target(monkeypatch, small_problem, orth_error_mode):
    problem = small_problem
    clock = install_clock_and_audit(monkeypatch, audit_cost=0.25)

    def fake_solver(**kwargs):
        assert kwargs["orth_error_mode"] == orth_error_mode
        kwargs["cycle_controller"](context(kwargs["initial_solution"], problem.ab_norm, 0))
        # An independently checked candidate above the actual eta target must
        # not stop just because the projected-residual trigger requested it.
        partial = context(tt_scale(problem.reference_solution, 0.5), 0.5 * problem.ab_norm, 1)
        partial["metric_valid_for_stopping"] = True
        kwargs["candidate_observer"](partial)
        clock.now += 0.75
        step(kwargs, 2)
        candidate = context(problem.reference_solution, 0.0, 2)
        candidate["metric_valid_for_stopping"] = True
        kwargs["candidate_observer"](candidate)
        pytest.fail("The independently checked target candidate should terminate early")

    monkeypatch.setattr(runner, "tt_dgmres_adaptive", fake_solver)
    config = runner.EpisodeConfig(orth_error_mode=orth_error_mode)
    result = runner.run_episode(problem, config, lambda obs: 3, evaluate_reference=False)
    assert result["summary"]["accepted"]
    assert result["summary"]["eta"] <= config.tolerance
    assert result["summary"]["wall_seconds"] == pytest.approx(1.0)
    assert result["summary"]["iterations"] == 2
    assert result["summary"]["reward_sum"] == pytest.approx(-1.0 / config.seconds_budget)


def test_invalid_orthogonalization_error_mode_is_rejected():
    with pytest.raises(ValueError):
        runner.EpisodeConfig(orth_error_mode="unknown").validate()


@pytest.mark.parametrize("improves", [True, False])
def test_inexact_breakdown_restarts_with_progress_or_uses_finite_recovery(monkeypatch, small_problem, improves):
    problem = small_problem
    clock = install_clock_and_audit(monkeypatch, audit_cost=0.25)
    calls = []

    def fake_solver(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            kwargs["cycle_controller"](context(kwargs["initial_solution"], problem.ab_norm, 0))
            clock.now += 1.0
            step(kwargs, 2)
            partial = tt_scale(problem.reference_solution, 0.5 if improves else 0.0)
            transformed = problem.ab_norm * (0.5 if improves else 1.0)
            kwargs["cycle_observer"](context(partial, transformed, 2))
            return SimpleNamespace(
                solution=partial, iterations=2, status="inexact_arnoldi_breakdown",
                converged=False, breakdown=True, history={},
            )
        assert len(calls) == 2
        assert kwargs["config"].max_iterations == 18
        assert kwargs["config"].max_tt_rank == (96 if improves else 192)
        kwargs["cycle_controller"](context(
            kwargs["initial_solution"], problem.ab_norm * (0.5 if improves else 1.0), 0,
        ))
        clock.now += 1.0
        step(kwargs, 2)
        kwargs["cycle_observer"](context(problem.reference_solution, 0.0, 2))

    monkeypatch.setattr(runner, "tt_dgmres_adaptive", fake_solver)
    result = runner.run_episode(
        problem, runner.EpisodeConfig(max_iterations=20), lambda obs: 2,
        evaluate_reference=False,
    )
    assert result["summary"]["accepted"]
    assert result["summary"]["iterations"] == 4
    assert result["summary"]["recoveries"] == int(not improves)
    assert result["summary"]["breakdown_restarts"] == int(improves)
    assert result["summary"]["executed_action_indices"][-1] == (2 if improves else -1)
    assert result["summary"]["wall_seconds"] == pytest.approx(2.25)
    assert result["summary"]["reward_sum"] == pytest.approx(-2.25 / 30.0)
