"""Independent small dense audits; these are floating-point checks, not proofs."""
import numpy as np
import pytest
import torch
import tntorch as tn

from rl_tt_dgmres_v2.problems import FAMILIES, ProblemSpec, build_problem
from rl_tt_dgmres_v2.rounding import FastRounder
from rl_tt_dgmres_v2 import runner


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def dense_matrix_from_edges(spec):
    """Build a separate dense Kronecker sum from the serialized edge list."""
    n = spec.n
    weighted = []
    for axis in range(2):
        matrix = np.zeros((n, n))
        edge_count = n if spec.family == "symmetric_weighted_cycle" else n - 1
        edges = spec.edge_weights[axis] if spec.edge_weights else (1.0,) * edge_count
        for edge, weight in enumerate(edges):
            direction = np.zeros(n)
            direction[edge] = 1.0
            direction[(edge + 1) % n] = -1.0
            matrix += weight * np.outer(direction, direction)
        weighted.append(spec.axis_weights[axis] * matrix)
    norm = sum(float(np.linalg.eigvalsh(matrix)[-1]) for matrix in weighted)
    return (np.kron(weighted[0], np.eye(n)) + np.kron(np.eye(n), weighted[1])) / norm


@pytest.mark.parametrize("case", range(15))
def test_small_random_episode_metrics_match_independent_dense_audit(case, monkeypatch):
    rng = np.random.default_rng(7900 + case)
    n = 3 + case % 2
    family = FAMILIES[case % len(FAMILIES)]
    edge_count = n if family == "symmetric_weighted_cycle" else n - 1
    edges = () if family == "symmetric_neumann_path" else tuple(
        tuple(float(value) for value in np.exp(rng.uniform(-0.6, 0.6, edge_count)))
        for _ in range(2)
    )
    spec = ProblemSpec(
        name="dense_audit_%02d" % case, n=n, dims=2, seed=7900 + case,
        rhs_rank=1 + case % 2, rhs_frequency=1 + case % (n - 1),
        contamination=0.0 if case % 2 == 0 else 0.1, family=family,
        axis_weights=tuple(float(value) for value in np.exp(rng.uniform(-0.5, 0.5, 2))),
        edge_weights=edges,
    )
    problem = build_problem(spec)
    matrix = dense_matrix_from_edges(spec)
    rhs = problem.rhs.torch().numpy().reshape(-1)
    reference = problem.reference_solution.torch().numpy().reshape(-1)
    q0 = np.ones(n * n) / n
    np.testing.assert_allclose(rhs - q0 * (q0 @ rhs), matrix @ reference, atol=2e-13)
    captured = []
    original_audit = runner.audit_solution

    def capture_final_solution(problem, solution, include_reference=False):
        captured.append(solution.torch().numpy().reshape(-1))
        return original_audit(problem, solution, include_reference=include_reference)

    monkeypatch.setattr(runner, "audit_solution", capture_final_solution)
    cfg = runner.EpisodeConfig(max_iterations=80, seconds_budget=4.0)
    result = runner.run_episode(problem, cfg, lambda observation: case % len(runner.ACTIONS))
    summary = result["summary"]
    assert len(captured) == 1
    actual = captured[-1]
    residual = rhs - matrix @ actual
    transformed = matrix @ residual
    dense_eta = np.linalg.norm(transformed) / (np.linalg.norm(actual) + np.linalg.norm(matrix @ rhs))
    dense_range = abs(q0 @ actual) / max(np.linalg.norm(actual), 1.0)
    assert summary["eta"] == pytest.approx(dense_eta, rel=2e-7, abs=3e-14)
    assert summary["relative_true_residual"] == pytest.approx(
        np.linalg.norm(residual) / np.linalg.norm(rhs), rel=2e-7, abs=3e-14)
    assert summary["range_defect"] == pytest.approx(dense_range, abs=3e-14)
    dense_success = dense_eta <= cfg.tolerance and dense_range <= cfg.range_tolerance
    assert summary["accepted"] == (dense_success and summary["status"] != "time_budget")
    if summary["accepted"]:
        positive_min = np.linalg.eigvalsh(matrix)[1]
        forward_error_bound = np.linalg.norm(transformed) / positive_min ** 2 + abs(q0 @ actual)
        assert np.linalg.norm(actual - reference) <= forward_error_bound + 1e-11
    assert summary["iterations"] <= cfg.max_iterations
    assert all(not row["is_residual_certificate"] and not row["is_verified_error_bound"]
               for row in result["round_calls"])
    print("case=%d action=%d accepted=%s eta=%.3e iterations=%d" %
          (case, case % len(runner.ACTIONS), summary["accepted"], dense_eta, summary["iterations"]))


def test_nonzero_underflowing_core_is_not_misclassified_as_structural_zero():
    # A finite, small norm is representable after the cores are contracted.
    # Squaring the entries of the first core underflows during an unscaled norm.
    tensor = tn.Tensor([
        torch.full((1, 2, 1), 1e-200, dtype=torch.float64),
        torch.full((1, 2, 1), 1e150, dtype=torch.float64),
    ])
    rounder = FastRounder()
    actual = rounder(tensor, 1e-10)
    np.testing.assert_allclose(actual.torch().numpy() / 1e-50, np.ones((2, 2)), rtol=1e-10)
    assert rounder.logs[-1]["audit_method"] != "structural_zero"
