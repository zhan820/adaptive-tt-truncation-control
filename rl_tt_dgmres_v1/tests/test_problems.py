"""Small dense oracles and leakage checks for the new RL benchmark family."""
from dataclasses import replace
import itertools
import math

import numpy as np
import pytest
import torch

from adaptive_tt_dgmres.backend.paper_tt_dgmres import tt_norm_stable
from adaptive_tt_dgmres.backend.paper_tt_gmres import tt_dot, tt_scale
from rl_tt_dgmres_v1.problems import (
    ProblemSpec, audit_solution, build_problem, make_splits, operator_group_key,
)


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def dense_operator(problem):
    # Intentionally restricted to small cases by tests; production never does this.
    n, dims = problem.spec.n, problem.spec.dims
    lap = np.diag([1.0] + [2.0] * (n - 2) + [1.0])
    lap += np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)
    lap /= 4.0 * math.cos(math.pi / (2.0 * n)) ** 2
    result = np.zeros((n**dims, n**dims))
    for axis, weight in enumerate(problem.normalized_axis_weights):
        term = np.ones((1, 1))
        for current in range(dims):
            term = np.kron(term, weight * lap if current == axis else np.eye(n))
        result += term
    return result


@pytest.mark.parametrize("n,dims,rank", [(2, 1, 1), (3, 2, 2), (4, 3, 3)])
def test_dense_oracle_norm_spectrum_range_and_rhs(n, dims, rank):
    problem = build_problem(ProblemSpec("oracle", n, dims, 42, 2.5, rank))
    dense = dense_operator(problem)
    eigenvalues = np.linalg.eigvalsh(dense)
    assert np.count_nonzero(np.abs(eigenvalues) < 1e-12) == 1
    assert eigenvalues[-1] == pytest.approx(1.0, abs=2e-14)
    assert eigenvalues[1] == pytest.approx(problem.lambda_min_positive, rel=2e-13)
    x = problem.reference_solution.torch().numpy().reshape(-1)
    b = problem.rhs.torch().numpy().reshape(-1)
    q0 = problem.q0.torch().numpy().reshape(-1)
    assert np.linalg.norm(x) == pytest.approx(1.0, abs=1e-13)
    np.testing.assert_allclose(q0, np.ones(n**dims) / math.sqrt(n**dims), atol=1e-15)
    np.testing.assert_allclose(dense @ q0, 0.0, atol=2e-16)
    assert abs(q0 @ x) < 2e-14
    np.testing.assert_allclose(
        problem.operator.matvec(problem.reference_solution).torch().numpy().reshape(-1),
        dense @ x, atol=2e-14,
    )
    np.testing.assert_allclose(b, dense @ x + 0.1 * np.linalg.norm(dense @ x) * q0, atol=2e-14)
    assert problem.ab_norm == pytest.approx(np.linalg.norm(dense @ b), rel=1e-13)
    assert max(problem.reference_solution.ranks_tt.tolist()) <= rank


def test_incompatible_drazin_solution_has_nonzero_ordinary_residual():
    problem = build_problem(ProblemSpec("incompatible", 5, 3, 13, contamination=0.1))
    metrics = audit_solution(problem, problem.reference_solution, include_reference=True)
    assert metrics["finite"]
    assert metrics["eta"] < 2e-14
    assert metrics["relative_transformed_residual"] < 1e-13
    assert metrics["relative_true_residual"] == pytest.approx(0.1 / math.sqrt(1.01), rel=1e-12)
    assert metrics["range_defect"] < 1e-13
    assert metrics["forward_error"] < 2e-14
    assert "forward_error" not in audit_solution(problem, problem.reference_solution)


def test_compatible_and_nullspace_contaminated_solution_are_distinguished():
    problem = build_problem(ProblemSpec("compatible", 4, 3, 22, contamination=0.0))
    assert audit_solution(problem, problem.reference_solution)["relative_true_residual"] < 1e-13
    polluted = problem.reference_solution + tt_scale(problem.q0, 0.3)
    metrics = audit_solution(problem, polluted, include_reference=True)
    assert metrics["eta"] < 2e-14
    assert metrics["range_defect"] == pytest.approx(0.3 / math.sqrt(1.09), rel=1e-12)
    assert metrics["forward_error"] == pytest.approx(0.3, rel=1e-12)
    projected = problem.projector(polluted)
    assert abs(tt_dot(problem.q0, projected)) < 2e-14
    assert tt_norm_stable(projected - problem.reference_solution) < 2e-14


def test_audit_matches_dense_oracle_away_from_solution():
    problem = build_problem(ProblemSpec("audit", 4, 2, 79, anisotropy=4.0))
    x = tt_scale(problem.reference_solution, 0.7) + tt_scale(problem.q0, 0.2)
    metrics = audit_solution(problem, x, include_reference=True)
    dense = dense_operator(problem)
    xv = x.torch().numpy().reshape(-1)
    b = problem.rhs.torch().numpy().reshape(-1)
    r = b - dense @ xv
    ar_norm = np.linalg.norm(dense @ r)
    assert metrics["eta"] == pytest.approx(ar_norm / (np.linalg.norm(xv) + np.linalg.norm(dense @ b)), rel=1e-12)
    assert metrics["relative_true_residual"] == pytest.approx(np.linalg.norm(r) / np.linalg.norm(b), rel=1e-12)
    assert metrics["relative_transformed_residual"] == pytest.approx(ar_norm / np.linalg.norm(dense @ b), rel=1e-12)


def test_nonfinite_solution_fails_cleanly():
    problem = build_problem(ProblemSpec("nonfinite", 3, 2, 1))
    x = problem.reference_solution.clone()
    x.cores[0][0, 0, 0] = math.nan
    metrics = audit_solution(problem, x, include_reference=True)
    assert not metrics["finite"]
    assert math.isinf(metrics["eta"])


def test_operator_groups_depend_on_actual_operator_not_names_or_rhs():
    original = ProblemSpec("a", 4, 3, 12, axis_weights=(1.0, 2.0, 3.0))
    same_operator = replace(original, name="different", seed=99, contamination=0.0,
                            rhs_rank=3, rhs_frequency=3, axis_weights=(2.0, 4.0, 6.0))
    assert operator_group_key(original) == operator_group_key(same_operator)
    assert operator_group_key(original) != operator_group_key(replace(original, axis_weights=(1.0, 3.0, 2.0)))
    first, second = build_problem(original), build_problem(same_operator)
    assert first.operator_fingerprint == second.operator_fingerprint
    assert first.construction_metadata()["operator_group_key"] == operator_group_key(original)


def test_splits_are_deterministic_disjoint_and_include_both_contaminations():
    splits = make_splits()
    assert splits == make_splits()
    assert {name: len(specs) for name, specs in splits.items()} == {
        "train": 48, "validation": 12, "test": 12, "ood": 6,
    }
    groups = {name: {operator_group_key(spec) for spec in specs} for name, specs in splits.items()}
    for a, b in itertools.combinations(groups, 2):
        assert groups[a].isdisjoint(groups[b])
    for specs in splits.values():
        assert {spec.contamination for spec in specs} == {0.0, 0.1}
        assert all(len(spec.axis_weights) == spec.dims for spec in specs)
    small = make_splits(train_count=2, validation_count=2, test_count=2, ood_count=2)
    for name in splits:
        assert small[name] == splits[name][:2]
    # Check actual matrix-core fingerprints too, not just metadata group labels.
    fingerprints = [build_problem(spec).operator_fingerprint for specs in small.values() for spec in specs]
    assert len(set(fingerprints)) == len(fingerprints)


def test_build_is_deterministic_without_changing_global_rng_and_online_view_is_narrow():
    spec = ProblemSpec("repeat", 5, 3, 29, rhs_rank=3, rhs_frequency=3)
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    first = build_problem(spec)
    second = build_problem(spec)
    assert torch.equal(before, torch.random.get_rng_state())
    for left, right in zip(first.reference_solution.cores, second.reference_solution.cores):
        assert torch.equal(left, right)
    online = first.online_view()
    assert not {"reference_solution", "spec", "seed", "forward_error"}.intersection(online)
    assert online["operator"] is first.operator
    with pytest.raises(TypeError):
        online["operator_norm"] = 2.0


def test_build_and_audit_do_not_expand_large_tensor(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A production routine expanded the full TT tensor")
    import tntorch as tn
    monkeypatch.setattr(tn.Tensor, "torch", forbidden)
    monkeypatch.setattr(tn.Tensor, "round_tt", forbidden)
    problem = build_problem(ProblemSpec("large", 8, 8, 14, rhs_rank=3, rhs_frequency=4))
    metrics = audit_solution(problem, problem.reference_solution, include_reference=True)
    assert metrics["finite"] and metrics["eta"] < 1e-12


@pytest.mark.parametrize("changes", [
    {"n": 1}, {"dims": 0}, {"anisotropy": 0.0}, {"contamination": -0.1},
    {"rhs_rank": 0}, {"rhs_frequency": 4}, {"axis_weights": (1.0,)},
    {"axis_weights": (1.0, -1.0)}, {"family": "unknown"},
])
def test_invalid_specs_are_rejected(changes):
    with pytest.raises(ValueError):
        build_problem(replace(ProblemSpec("invalid", 4, 2, 2), **changes))
