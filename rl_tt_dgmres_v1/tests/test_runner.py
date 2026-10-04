from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adaptive_tt_dgmres.backend.paper_tt_gmres import tt_max_rank, tt_rank1
from rl_tt_dgmres_v1 import runner as runner_module
from rl_tt_dgmres_v1.problems import ProblemSpec, build_problem
from rl_tt_dgmres_v1.runner import (
    ACTIONS, FEATURE_NAMES, ConservativeRounder, EpisodeConfig, run_episode, _log,
)


@pytest.fixture(scope="module")
def problem():
    torch.set_num_threads(1)
    return build_problem(ProblemSpec("runner_test", 4, 2, 401, contamination=0.1))


def test_scope_and_transition_accounting(problem):
    config = EpisodeConfig(update_dim=4, max_iterations=15, tolerance=1e-20, seconds_budget=30)
    detail = run_episode(problem, config, lambda obs: 3, evaluate_reference=False)
    assert detail["transitions"]
    assert all(len(t["state"]) == len(FEATURE_NAMES) for t in detail["transitions"])
    assert all(np.isfinite(t["state"]).all() for t in detail["transitions"])
    assert sum(t["done"] for t in detail["transitions"]) == 1
    assert detail["transitions"][-1]["done"]
    for row in detail["round_calls"]:
        assert row["requested_eps"] == (ACTIONS[3] if row["phase"] == "matvec"
                                       else config.fixed_round_tolerance)
    summary = detail["summary"]
    expected = -summary["wall_seconds"] / config.seconds_budget
    if not summary["accepted"]:
        expected -= config.failure_penalty
    assert summary["reward_sum"] == pytest.approx(expected, abs=1e-12)
    cycles = detail["cycles"]
    for before, after in zip(cycles, cycles[1:]):
        assert before["iteration_end"] == after["iteration_start"]
        assert before["next_state"] == after["state"]
    assert all(row["iteration_end"] - row["iteration_start"] <= config.update_dim + 1 for row in cycles)


def test_incompatible_problem_can_pass_transformed_acceptance(problem):
    config = EpisodeConfig(update_dim=10, max_iterations=100, tolerance=1e-6, seconds_budget=30)
    detail = run_episode(problem, config, lambda obs: 0)
    summary = detail["summary"]
    assert summary["accepted"], summary
    assert summary["eta"] <= config.tolerance
    assert summary["range_defect"] < config.range_tolerance
    assert summary["relative_true_residual"] > 0.09


def test_reference_is_not_read_during_training(problem):
    blinded = replace(problem, reference_solution=object())
    detail = run_episode(blinded, EpisodeConfig(update_dim=4, max_iterations=6),
                         lambda obs: 0, evaluate_reference=False)
    assert detail["summary"]["forward_error"] is None


def test_time_limit_is_not_success(problem):
    detail = run_episode(problem, EpisodeConfig(seconds_budget=1e-9), lambda obs: 0)
    assert not detail["summary"]["accepted"]
    assert detail["summary"]["status"] == "time_budget"


@pytest.mark.parametrize("requested_eps", [1e-10, 1e-2])
def test_near_zero_orth_preserves_exact_cores_and_actual_cycle(requested_eps):
    tensor = tt_rank1([
        torch.tensor([1e-12, -2e-12], dtype=torch.float64),
        torch.tensor([0.6, 0.8], dtype=torch.float64),
        torch.tensor([1.0, -1.0], dtype=torch.float64),
    ])
    original_cores = [core.clone() for core in tensor.cores]
    original_ranks = tensor.ranks_tt.tolist()
    rounder = ConservativeRounder(adaptive=False)
    rounder.cycle = 7

    rounded = rounder(tensor, requested_eps, max_rank=1, phase="orth")

    assert rounded is not tensor
    assert rounded.ranks_tt.tolist() == original_ranks
    assert tt_max_rank(rounded) <= 1
    for source, saved, copied in zip(tensor.cores, original_cores, rounded.cores):
        assert torch.equal(source, saved), "The preservation check must not mutate its input."
        assert torch.equal(copied, saved), "Every retained core must be an exact copy."
        assert copied.data_ptr() != source.data_ptr()
    assert len(rounder.logs) == 1
    record = rounder.logs[0]
    assert record["cycle"] == 7
    assert record["preserved_near_cancellation"]
    assert 0.0 < record["input_norm"] < 1e-10
    assert record["requested_eps"] == requested_eps
    assert record["effective_eps"] == 0.0
    assert record["actual_absolute_error"] == record["actual_relative_error"] == 0.0
    assert record["noise_slack_absolute"] == record["noise_slack_relative"] == 0.0
    assert record["ranks_before"] == record["ranks_after"]
    assert record["audit_method"] == "identity_core_copy_no_truncation"
    assert record["status"] == "accepted"


@pytest.mark.parametrize("audit_end,accepted", [(9.75, True), (10.25, False)])
def test_final_audit_time_counts_toward_deadline_and_terminal_reward(
        problem, monkeypatch, audit_end, accepted):
    # Keep numerical solving and real-time scheduling out of this deadline test.
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(runner_module, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    config = EpisodeConfig(seconds_budget=10.0)
    events = []

    def completed_solver(**kwargs):
        context = {
            "x": runner_module.tt_zeros(problem.rhs.shape, like=problem.rhs),
            "cycle": 1,
            "total_iterations": 0,
            "metrics": {
                "transformed_residual_norm": problem.ab_norm,
                "transformed_residual_certified": True,
            },
            "projected_residual": problem.ab_norm,
        }
        kwargs["cycle_controller"](context)
        clock.now = 9.5
        context.update({
            "x": problem.reference_solution,
            "total_iterations": 2,
            "metrics": {
                "transformed_residual_norm": 0.0,
                "transformed_residual_certified": True,
            },
            "projected_residual": 0.0,
        })
        events.append("cycle_completed_before_deadline")
        # The controller raises EpisodeStop after the converged cycle; the real
        # run_episode must still perform its final independent acceptance audit.
        kwargs["cycle_observer"](context)
        raise AssertionError("The converged cycle must stop the solver.")

    def timed_final_audit(audited_problem, solution, *, include_reference):
        assert audited_problem is problem
        assert solution is problem.reference_solution
        assert not include_reference
        assert clock.now == 9.5
        events.append("final_audit")
        clock.now = audit_end
        return {
            "finite": True, "eta": 0.0, "range_defect": 0.0,
            "relative_transformed_residual": 0.0, "relative_true_residual": 0.1,
        }

    monkeypatch.setattr(runner_module, "tt_dgmres_adaptive", completed_solver)
    monkeypatch.setattr(runner_module, "audit_solution", timed_final_audit)
    detail = run_episode(problem, config, lambda obs: 0, evaluate_reference=False)

    assert events == ["cycle_completed_before_deadline", "final_audit"]
    summary = detail["summary"]
    assert summary["accepted"] is accepted
    assert summary["status"] == ("eta_target" if accepted else "time_budget")
    assert summary["eta"] == 0.0
    assert summary["wall_seconds"] == pytest.approx(audit_end)
    assert summary["final_audit_seconds"] == pytest.approx(audit_end - 9.5)
    assert len(detail["transitions"]) == len(detail["cycles"]) == 1
    assert detail["cycles"][0]["completed"]
    assert detail["cycles"][0]["seconds"] == pytest.approx(audit_end)
    assert detail["transitions"][0]["done"]
    expected_reward = -audit_end / config.seconds_budget
    if not accepted:
        expected_reward -= config.failure_penalty
    assert detail["transitions"][0]["reward"] == pytest.approx(expected_reward)
    assert summary["reward_sum"] == pytest.approx(expected_reward)


def test_nonfinite_is_not_silently_clipped():
    with pytest.raises(FloatingPointError):
        _log(float("nan"))


@pytest.mark.parametrize("bad", [{"scope": "all"}, {"update_dim": 0}, {"tolerance": float("nan")}])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        EpisodeConfig(**bad).validate()
