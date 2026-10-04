"""Shared-engine, online-feature and exact outer-state checkpoint checks."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adaptive_tt_dgmres.backend.paper_tt_dgmres import _stable_tt_norm
from adaptive_tt_dgmres.backend.paper_tt_gmres import tt_scale
from rl_tt_dgmres_v2.problems import ProblemSpec, audit_solution, build_problem
from rl_tt_dgmres_v3 import runner


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def problem():
    return build_problem(ProblemSpec("runner_v3", 3, 2, 881, rhs_rank=1))


def context(x, transformed, iterations):
    return {"x": x, "total_iterations": iterations,
            "metrics": {"transformed_residual_norm": transformed,
                        "transformed_residual_certified": True},
            "projected_residual": transformed}


def install_clock(monkeypatch, audit_cost=0.25):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock.now)

    def independent(problem, solution, include_reference=False):
        assert include_reference is False
        clock.now += audit_cost
        return audit_solution(problem, solution, include_reference=False)

    monkeypatch.setattr(runner, "audit_solution", independent)
    return clock


def fake_cycle_solver(problem, clock, actions=None):
    def solve(**kwargs):
        assert kwargs["config"].max_cycles == 1
        previous = kwargs["initial_solution"]
        fraction = float(torch.sum(previous.torch() * problem.reference_solution.torch()) /
                         torch.sum(problem.reference_solution.torch() ** 2))
        clock.now += 0.3
        controls = kwargs["cycle_controller"](context(previous, problem.ab_norm * (1 - fraction), 0))
        if actions is not None:
            actions.append(controls)
        clock.now += 0.7
        kwargs["iteration_observer"]({
            "iteration": 2, "basis_tt_core_bytes": 128,
            "orthogonalization_passes": 1, "reorthogonalized": False,
        })
        next_fraction = min(1.0, fraction + 0.5)
        solution = tt_scale(problem.reference_solution, next_fraction)
        kwargs["cycle_observer"](context(solution, problem.ab_norm * (1 - next_fraction), 2))
        return SimpleNamespace(solution=solution, iterations=2, status="cycle_limit",
                               converged=False, breakdown=False, history={})

    return solve


def test_features_are_online_numeric_and_distinguish_input_structure(problem):
    online = problem.online_view()
    richer = build_problem(ProblemSpec("richer", 3, 2, 992, rhs_rank=2))
    left = runner._input_features(online)
    right = runner._input_features(richer.online_view())
    assert len(left) == 12
    assert runner.FEATURE_DIM == 32
    assert np.isfinite(left).all()
    assert all(isinstance(value, float) for value in left)
    assert left != right
    assert not any("seed" in name or "family" in name or "reference" in name
                   for name in runner.FEATURE_NAMES)


def test_cycle_checkpoints_restore_history_and_only_charge_remaining_work(monkeypatch, problem):
    clock = install_clock(monkeypatch)
    monkeypatch.setattr(runner, "tt_dgmres_adaptive", fake_cycle_solver(problem, clock))
    checkpoints = []
    cfg = runner.EpisodeConfig(max_iterations=12, seconds_budget=10)
    full = runner.run_episode(problem, cfg, lambda _: 6,
                              checkpoint_observer=checkpoints.append, return_solution=True)
    assert full["summary"]["accepted"]
    assert full["summary"]["wall_seconds"] == pytest.approx(2.25)
    assert len(checkpoints) == 2
    middle = checkpoints[1]
    assert middle.elapsed_seconds == pytest.approx(1.0)
    assert middle.controller_state["iterations"] == 2
    assert middle.controller_state["cycle_number"] == 1
    assert middle.features is middle.observation
    assert len(middle.features) == 32
    assert middle.cycle_initialization_seconds == pytest.approx(0.3)
    clock.now = 500.0
    restored = runner.run_episode(problem, cfg, lambda _: 6,
                                  initial_checkpoint=middle, return_solution=True)
    summary = restored["summary"]
    assert summary["accepted"]
    assert summary["wall_seconds"] == pytest.approx(1.25)
    assert summary["total_elapsed_seconds"] == pytest.approx(2.25)
    assert summary["remaining_time_budget_at_start"] == pytest.approx(9.0)
    assert summary["iterations"] == 4
    assert summary["iterations_this_run"] == 2
    assert summary["total_cycles"] == 2
    assert summary["cycles"] == 1
    assert summary["input_feature_extra_matvecs"] == 0
    assert summary["reward_sum"] == pytest.approx(-1.25 / cfg.seconds_budget)
    assert _stable_tt_norm(full["solution"] - restored["solution"]) < 1e-13


def test_observer_overhead_is_explicitly_excluded(monkeypatch, problem):
    clock = install_clock(monkeypatch)
    monkeypatch.setattr(runner, "tt_dgmres_adaptive", fake_cycle_solver(problem, clock))
    checkpoints = []

    def observer(checkpoint):
        checkpoints.append(checkpoint)
        clock.now += 80.0

    result = runner.run_episode(problem, runner.EpisodeConfig(seconds_budget=3.0),
                                lambda _: 6, checkpoint_observer=observer)
    assert result["summary"]["accepted"]
    assert result["summary"]["wall_seconds"] == pytest.approx(2.25)
    assert result["summary"]["checkpoint_collection_seconds_excluded"] == pytest.approx(160)
    assert checkpoints[1].elapsed_seconds == pytest.approx(1.0)
    assert sum(row["seconds"] for row in result["cycles"]) == pytest.approx(2.25)


def test_restore_does_not_reset_time_budget(monkeypatch, problem):
    clock = install_clock(monkeypatch)
    monkeypatch.setattr(runner, "tt_dgmres_adaptive", fake_cycle_solver(problem, clock))
    checkpoints = []
    cfg = runner.EpisodeConfig(seconds_budget=2.1)
    result = runner.run_episode(problem, cfg, lambda _: 6, checkpoint_observer=checkpoints.append)
    assert not result["summary"]["accepted"]
    assert result["summary"]["status"] == "time_budget"
    restored = runner.run_episode(problem, cfg, lambda _: 6, initial_checkpoint=checkpoints[1])
    assert restored["summary"]["eta"] <= cfg.tolerance
    assert not restored["summary"]["accepted"]
    assert restored["summary"]["status"] == "time_budget"
    assert restored["summary"]["total_elapsed_seconds"] == pytest.approx(2.25)


def test_checkpoint_config_must_be_identical(monkeypatch, problem):
    clock = install_clock(monkeypatch)
    monkeypatch.setattr(runner, "tt_dgmres_adaptive", fake_cycle_solver(problem, clock))
    checkpoints = []
    cfg = runner.EpisodeConfig(seconds_budget=10)
    runner.run_episode(problem, cfg, lambda _: 6, checkpoint_observer=checkpoints.append)
    with pytest.raises(ValueError, match="configuration"):
        runner.run_episode(problem, replace(cfg, tolerance=1e-5), lambda _: 6,
                           initial_checkpoint=checkpoints[0])


def test_restored_protection_state_and_absolute_iteration_budget_are_preserved(monkeypatch, problem):
    clock = install_clock(monkeypatch)
    monkeypatch.setattr(runner, "tt_dgmres_adaptive", fake_cycle_solver(problem, clock))
    checkpoints = []
    cfg = runner.EpisodeConfig(seconds_budget=10, max_iterations=12)
    original = runner.run_episode(problem, cfg, lambda _: 6, checkpoint_observer=checkpoints.append)
    for checkpoint, cycle in zip(checkpoints, original["cycles"]):
        assert checkpoint.features == cycle["state"]
    checkpoint = checkpoints[1]
    checkpoint.controller_state.update(
        rank_cap=192, force_safe=1, stagnation=2,
        recoveries=[{"reason": "synthetic_past_recovery"}],
    )
    observed = []

    def resume(**kwargs):
        assert kwargs["config"].max_iterations == 10
        assert kwargs["config"].max_tt_rank == 192
        assert kwargs["config"].residual_round_tolerance == 1e-12
        kwargs["cycle_controller"](context(kwargs["initial_solution"], problem.ab_norm * 0.5, 0))
        clock.now += 1
        kwargs["iteration_observer"]({"iteration": 2, "basis_tt_core_bytes": 12})
        kwargs["cycle_observer"](context(problem.reference_solution, 0, 2))

    monkeypatch.setattr(runner, "tt_dgmres_adaptive", resume)
    result = runner.run_episode(problem, cfg, lambda features: observed.append(features) or 2,
                                initial_checkpoint=checkpoint)
    assert result["summary"]["accepted"]
    assert result["summary"]["recoveries"] == 1
    assert result["summary"]["iterations"] == 4
    assert result["summary"]["executed_action_indices"] == [-1]
    assert observed[0][7] == pytest.approx(2 / 12)
    assert observed[0][17] == pytest.approx(2 / 4)
    assert observed[0][18] == pytest.approx(1 / 3)
    assert observed[0][19] == pytest.approx(192 / 256)


def test_rollout_first_action_then_baseline_and_training_row(monkeypatch, problem):
    clock = install_clock(monkeypatch)
    actions = []
    monkeypatch.setattr(runner, "tt_dgmres_adaptive", fake_cycle_solver(problem, clock, actions))
    checkpoints = []
    cfg = runner.EpisodeConfig(seconds_budget=10, safety_filter=False)
    runner.run_episode(problem, cfg, lambda _: 6, checkpoint_observer=checkpoints.append)
    actions.clear()
    result = runner.rollout_from_checkpoint(problem, cfg, checkpoints[0], 1, 6)
    assert result["summary"]["action_indices"] == [1, 6]
    assert actions[0]["restart"] == runner.ACTIONS[1].update_dim + 1
    assert actions[1]["restart"] == runner.ACTIONS[6].update_dim + 1
    row = result["training_row"]
    assert row["features"] == checkpoints[0].features
    assert row["action"] == 1
    assert row["executed_first_action"] == 1
    assert row["remaining_wall"] == pytest.approx(2.25)
    assert row["time_budget"] == 10
    assert row["success"]
    middle = runner.rollout_from_checkpoint(problem, cfg, checkpoints[1], 1, 6)
    assert middle["training_row"]["time_budget"] == 10
    assert middle["training_row"]["remaining_time_budget"] == pytest.approx(9.0)


def test_feature_disabled_deployment_skips_feature_work_and_disallows_checkpoints(monkeypatch, problem):
    def forbidden(*args, **kwargs):
        raise AssertionError("Fixed deployment must not compute policy features")

    monkeypatch.setattr(runner, "_input_features", forbidden)
    monkeypatch.setattr(runner.V2Controller, "observation", forbidden)
    observed = []
    result = runner.run_episode(problem, runner.EpisodeConfig(seconds_budget=10),
                                lambda values: observed.append(values) or 6,
                                compute_policy_features=False)
    assert result["summary"]["accepted"]
    assert observed and all(values == [0.0] * runner.FEATURE_DIM for values in observed)
    assert result["summary"]["feature_seconds"] == 0
    assert result["summary"]["input_feature_extra_matvecs"] == 0
    assert result["summary"]["feature_mode"] == "disabled_fixed_deployment"
    with pytest.raises(ValueError, match="Checkpoints require"):
        runner.run_episode(problem, runner.EpisodeConfig(), lambda _: 6,
                           compute_policy_features=False, checkpoint_observer=lambda _: None)
    with pytest.raises(ValueError, match="Checkpoints require"):
        runner.run_episode(problem, runner.EpisodeConfig(), lambda _: 6,
                           compute_policy_features=False, initial_checkpoint=object())


@pytest.mark.parametrize("action", [0, 4, 6, 8])
def test_fixed_deployment_and_feature_matched_fixed_have_same_numerics(action):
    problem = build_problem(ProblemSpec("fixed_deployment", 4, 3, 1029, rhs_rank=2))
    cfg = runner.EpisodeConfig(seconds_budget=10, max_iterations=100)
    observed = runner.run_episode(problem, cfg, lambda _: action,
                                  return_solution=True, compute_policy_features=True)
    deployment = runner.run_episode(problem, cfg, lambda _: action,
                                    return_solution=True, compute_policy_features=False)
    for key in ("accepted", "iterations", "eta", "action_indices", "executed_action_indices"):
        assert observed["summary"][key] == deployment["summary"][key]
    assert torch.equal(observed["solution"].torch(), deployment["solution"].torch())
    assert observed["summary"]["input_feature_extra_matvecs"] == 1
    assert deployment["summary"]["input_feature_extra_matvecs"] == 0


def test_reference_is_not_read_by_default(monkeypatch, problem):
    clock = install_clock(monkeypatch)
    monkeypatch.setattr(runner, "tt_dgmres_adaptive", fake_cycle_solver(problem, clock))

    class Guard:
        spec = problem.spec

        def online_view(self):
            return problem.online_view()

        @property
        def reference_solution(self):
            raise AssertionError("No reference access in default training/inference")

        def __getattr__(self, name):
            return getattr(problem, name)

    result = runner.run_episode(Guard(), runner.EpisodeConfig(), lambda _: 6)
    assert result["summary"]["forward_error"] is None
    assert "solution" not in result


def test_real_checkpoint_roundtrip_preserves_numerical_result(tmp_path):
    problem = build_problem(ProblemSpec("roundtrip", 5, 3, 996, rhs_rank=2))
    cfg = runner.EpisodeConfig(seconds_budget=10, max_iterations=100)
    checkpoints = []
    full = runner.run_episode(problem, cfg, lambda _: 0,
                              checkpoint_observer=checkpoints.append, return_solution=True)
    assert len(checkpoints) >= 2
    location = tmp_path / "checkpoint.pt"
    torch.save(checkpoints[1], location)
    checkpoint = torch.load(location, weights_only=False)
    resumed = runner.run_episode(problem, cfg, lambda _: 0,
                                 initial_checkpoint=checkpoint, return_solution=True)
    assert full["summary"]["accepted"] == resumed["summary"]["accepted"]
    assert full["summary"]["iterations"] == resumed["summary"]["iterations"]
    assert full["summary"]["eta"] == pytest.approx(resumed["summary"]["eta"], rel=1e-6, abs=1e-13)
    assert _stable_tt_norm(full["solution"] - resumed["solution"]) < 1e-12
    before = checkpoint.x.torch().clone()
    with torch.inference_mode():
        resumed["solution"].cores[0].zero_()
    assert torch.equal(checkpoint.x.torch(), before)
