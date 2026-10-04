import math

import torch

from adaptive_tt_dgmres.experiment import solve
from adaptive_tt_dgmres.problems import ProblemSpec, build_problem, final_audit


def test_joint_actions_reach_independent_residual_and_range_targets():
    torch.set_num_threads(1)
    problem = build_problem(ProblemSpec("integration", 5, 2, 11))
    summary, detail, controller, result = solve(problem, mode="scripted", max_iterations=100)
    assert summary["accepted"]
    assert summary["audit"]["relative_true_residual"] < 1e-6
    assert summary["audit"]["solution_range_defect"] < 1e-9
    actions = {(row["restart"], row["alpha"]) for row in controller.records}
    assert len(actions) >= 2
    assert len({row["effective_eps"] for row in detail["round_calls"]}) >= 2
    assert all(row["within_effective"] for row in detail["round_calls"])
    assert result.iterations <= 100


def test_residual_alone_does_not_hide_generalized_nullspace_pollution():
    torch.set_num_threads(1)
    problem = build_problem(ProblemSpec("nullspace_audit", 5, 2, 11))
    contaminated = problem.reference_solution + 100.0 * problem.right_null
    audit = final_audit(problem, contaminated)
    assert audit["relative_true_residual"] < 1e-10
    assert audit["solution_range_defect"] > 0.1
    assert audit["relative_solution_error_evaluation_only"] > 100


def test_reference_error_is_evaluation_only():
    torch.set_num_threads(1)
    problem = build_problem(ProblemSpec("no_reference_leak", 4, 2, 9))
    first, _, controller_a, result_a = solve(problem, mode="scripted", max_iterations=80)
    problem.reference_solution = 10.0 * problem.reference_solution
    second, _, controller_b, result_b = solve(problem, mode="scripted", max_iterations=80)
    assert first["accepted"] and second["accepted"]
    assert first["iterations"] == second["iterations"]
    assert first["audit"]["relative_true_residual"] == second["audit"]["relative_true_residual"]
    assert first["audit"]["relative_solution_error_evaluation_only"] != second["audit"]["relative_solution_error_evaluation_only"]
    assert [(r["restart"], r["alpha"]) for r in controller_a.records] == [
        (r["restart"], r["alpha"]) for r in controller_b.records]


def test_progress_guard_recovers_a_repeated_short_restart_proposal():
    class AlwaysShort:
        def __call__(self, **inputs):
            return {"log_cost": torch.arange(inputs["actions"].shape[-2]).reshape(1, -1).float()}

    torch.set_num_threads(1)
    problem = build_problem(ProblemSpec("short_restart_recovery", 7, 3, 101, 1.25))
    raw, _, _, _ = solve(problem, mode="neural", model=AlwaysShort(), progress_guard=False)
    guarded, _, controller, _ = solve(problem, mode="neural", model=AlwaysShort())
    assert not raw["accepted"] and raw["status"] == "iteration_limit"
    assert guarded["accepted"] and guarded["progress_guard_activations"] == 1
    assert guarded["fallback_cycles"] > 0
    assert all(c["proposed_restart"] == 4 for c in controller.records)
    assert any(c["restart"] == 12 for c in controller.records)
