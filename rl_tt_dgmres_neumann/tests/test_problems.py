"""Small, independent NumPy oracles for the paper's CP construction (no solves)."""
from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest
import torch

from adaptive_tt_dgmres.backend.paper_tt_gmres import tt_dot, tt_scale
from rl_tt_dgmres_v2.problems import ProblemSpec as GraphSpec
from rl_tt_dgmres_v2.problems import operator_group_key as graph_operator_key
from rl_tt_dgmres_neumann import problems as module
from rl_tt_dgmres_neumann.problems import (
    BUILDER_SOURCE, ProblemSpec, audit_solution, build_problem, instance_key,
    operator_group_key, spec_from_dict,
)


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def dense_oracle(spec):
    """Physical NumPy tensors reproduce the original builder's RNG order.

    No production CP/TT constructor, matvec, projector, norm or rounding is
    called. Tiny source preparation errors are covered by explicit tolerances.
    The matrix is built from graph incidence, independently of production.
    """
    n, dims = spec.n, spec.dims
    incidence = np.zeros((n - 1, n))
    for edge in range(n - 1):
        incidence[edge, edge] = 1
        incidence[edge, edge + 1] = -1
    local = (n - 1) ** 2 * incidence.T @ incidence
    full = np.zeros((n ** dims, n ** dims))
    for axis in range(dims):
        term = np.ones((1, 1))
        for current in range(dims):
            term = np.kron(term, local if current == axis else np.eye(n))
        full += term
    full /= np.linalg.eigvalsh(full)[-1]
    max_mode = n - 1 if spec.max_mode is None else spec.max_mode
    phi = np.empty((n, max_mode + 1))
    phi[:, 0] = 1 / np.sqrt(n)
    for q in range(1, max_mode + 1):
        phi[:, q] = np.sqrt(2 / n) * np.cos(np.pi * q * (np.arange(n) + .5) / n)
    rng = np.random.default_rng(spec.seed)
    reference = np.zeros(n ** dims)
    for j in range(spec.cp_rank):
        rank_one = np.ones(1)
        for _ in range(dims):
            coeff = rng.standard_normal(max_mode + 1) / (1 + np.arange(max_mode + 1)) ** spec.spectral_decay
            factor = phi @ coeff
            rank_one = np.kron(rank_one, factor / np.linalg.norm(factor))
        reference += .5 ** j * rank_one
    reference -= reference.mean()
    reference /= np.linalg.norm(reference)
    q0 = np.ones(n ** dims) / np.sqrt(n ** dims)
    compatible = full @ reference
    rhs = compatible + spec.null_ratio * np.linalg.norm(compatible) * q0
    return full, reference, rhs, q0


@pytest.mark.parametrize("n,dims", [(3, 2), (3, 3), (4, 2), (4, 3)])
@pytest.mark.parametrize("seed", [20260906, 2037])
def test_matches_original_numpy_cp_rng_and_dense_drazin_problem(n, dims, seed):
    spec = ProblemSpec("paper", n, dims, seed)
    problem = build_problem(spec)
    matrix, expected_x, expected_b, q0 = dense_oracle(spec)
    x = problem.reference_solution.torch().numpy().ravel()
    b = problem.rhs.torch().numpy().ravel()
    np.testing.assert_allclose(x, expected_x, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(b, expected_b, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(problem.q0.torch().numpy().ravel(), q0, atol=2e-16)
    np.testing.assert_allclose(problem.operator.matvec(problem.reference_solution).torch().numpy().ravel(), matrix @ x, atol=2e-14)
    np.testing.assert_allclose(np.linalg.pinv(matrix) @ b, x, atol=2e-12)
    spectrum = np.linalg.eigvalsh(matrix)
    assert np.count_nonzero(abs(spectrum) < 1e-12) == 1
    assert spectrum[-1] == pytest.approx(1., abs=2e-14)
    assert spectrum[1] == pytest.approx(problem.lambda_min_positive, rel=2e-12)
    assert np.linalg.norm(x) == pytest.approx(1., abs=2e-14)
    assert abs(q0 @ x) < 2e-13
    assert q0 @ b == pytest.approx(.1 * np.linalg.norm(matrix @ x), rel=2e-12)
    assert problem.ab_norm == pytest.approx(np.linalg.norm(matrix @ b), rel=2e-12)
    audit = audit_solution(problem, problem.reference_solution, include_reference=True)
    assert audit["finite"] and audit["eta"] < 2e-13
    assert audit["range_defect"] < 2e-13
    assert audit["relative_true_residual"] == pytest.approx(.1 / math.sqrt(1.01), rel=2e-11)
    # TT subtraction of the identical represented tensor still incurs roundoff.
    assert audit["forward_error"] < 2e-14


@pytest.mark.parametrize("null_ratio,max_mode,cp_rank,decay", [(0., 1, 1, 1.), (.2, 2, 2, 3.), (.1, 3, 4, 0.)])
def test_explicit_numerical_variants(null_ratio, max_mode, cp_rank, decay):
    spec = ProblemSpec("variant", 4, 3, 109, null_ratio=null_ratio, max_mode=max_mode,
                       cp_rank=cp_rank, spectral_decay=decay)
    problem = build_problem(spec)
    _, reference, rhs, _ = dense_oracle(spec)
    np.testing.assert_allclose(problem.reference_solution.torch().numpy().ravel(), reference, atol=3e-13)
    np.testing.assert_allclose(problem.rhs.torch().numpy().ravel(), rhs, atol=3e-13)


def test_same_seed_is_deterministic_different_seed_changes_only_rhs_identity():
    spec = ProblemSpec("one", 4, 3, 20260906)
    first, repeat = build_problem(spec), build_problem(replace(spec, name="renamed"))
    changed = build_problem(replace(spec, seed=20260907))
    for left, right in zip(first.rhs.cores, repeat.rhs.cores):
        assert torch.equal(left, right)
    assert first.instance_key == repeat.instance_key != changed.instance_key
    assert first.operator_group_key == repeat.operator_group_key == changed.operator_group_key
    assert first.operator_fingerprint == changed.operator_fingerprint
    assert not np.allclose(first.rhs.torch().numpy(), changed.rhs.torch().numpy())


def test_keys_use_numeric_recipe_and_real_operator_not_labels():
    spec = ProblemSpec("base", 4, 3, 17)
    assert instance_key(spec) == instance_key(replace(spec, name="other", max_mode=3))
    variants = [replace(spec, **change) for change in (
        {"seed": 18}, {"cp_rank": 4}, {"max_mode": 2}, {"null_ratio": .2},
        {"spectral_decay": 1.}, {"preparation_tolerance": 1e-12},
    )]
    assert len({instance_key(spec), *(instance_key(v) for v in variants)}) == len(variants) + 1
    assert all(operator_group_key(v) == operator_group_key(spec) for v in variants)
    equivalent = GraphSpec("old_label", 4, 3, 99, axis_weights=(1., 1., 1.))
    assert operator_group_key(spec) == graph_operator_key(equivalent)
    assert operator_group_key(spec) != operator_group_key(replace(spec, n=5))
    assert operator_group_key(spec) != operator_group_key(replace(spec, dims=2))


def test_three_preparation_rounds_have_no_rank_cap_and_ab_is_unrounded(monkeypatch):
    seen = []
    original = module.tt_round_capped

    def recorded(tensor, eps, max_rank=None):
        seen.append((eps, max_rank))
        return original(tensor, eps, max_rank)

    monkeypatch.setattr(module, "tt_round_capped", recorded)
    spec = ProblemSpec("rounds", 4, 3, 5)
    problem = build_problem(spec)
    assert seen == [(1e-13, None)] * 3
    metadata = problem.construction_metadata()
    assert metadata["preparation"]["rank_cap"] is None
    assert metadata["preparation"]["ab_is_unrounded"] is True
    assert metadata["builder_source"] == BUILDER_SOURCE
    assert metadata["construction_parameters"]["max_mode"] == 3
    assert metadata["instance_key"] == instance_key(spec)
    json.dumps(metadata, allow_nan=False)


def test_reference_and_seed_are_excluded_from_online_inputs_and_projection_works():
    problem = build_problem(ProblemSpec("online", 4, 3, 42))
    online = problem.online_view()
    assert not {"seed", "reference_solution", "xstar", "spec", "builder_source", "instance_key"}.intersection(online)
    with pytest.raises(TypeError):
        online["index"] = 2
    polluted = problem.reference_solution + tt_scale(problem.q0, .3)
    assert abs(tt_dot(problem.q0, problem.projector(polluted))) < 3e-14
    assert audit_solution(problem, polluted)["range_defect"] > .2


def test_construction_never_expands_full_tensor_and_preserves_global_rng(monkeypatch):
    import tntorch as tn

    def forbidden(*args, **kwargs):
        raise AssertionError("Problem construction expanded a full tensor")

    monkeypatch.setattr(tn.Tensor, "torch", forbidden)
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()
    problem = build_problem(ProblemSpec("TT_only", 3, 3, 80))
    assert torch.equal(torch_before, torch.random.get_rng_state())
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert audit_solution(problem, problem.reference_solution)["eta"] < 1e-12


def test_serialization_and_zero_preparation_tolerance():
    spec = ProblemSpec("serial", 3, 2, 3, preparation_tolerance=0.)
    assert spec_from_dict(json.loads(json.dumps(asdict(spec)))) == spec
    assert audit_solution(build_problem(spec), build_problem(spec).reference_solution)["eta"] < 1e-12
    with pytest.raises(ValueError, match="Unknown"):
        spec_from_dict({**asdict(spec), "rhs_rank": 3})


@pytest.mark.parametrize("change", [
    {"name": ""}, {"n": 2}, {"n": True}, {"dims": 1}, {"seed": -1},
    {"seed": True}, {"cp_rank": 0}, {"max_mode": 0}, {"max_mode": 4},
    {"null_ratio": -.1}, {"null_ratio": float("inf")}, {"null_ratio": True},
    {"spectral_decay": -1.}, {"preparation_tolerance": float("nan")},
    {"preparation_tolerance": 1.}, {"family": "symmetric_neumann_path"},
])
def test_invalid_spec_rejected(change):
    with pytest.raises(ValueError):
        build_problem(replace(ProblemSpec("bad", 4, 2, 1), **change))
