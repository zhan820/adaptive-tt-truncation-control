"""Small dense oracles for the expanded graph families and split isolation."""
from dataclasses import asdict, replace
import hashlib
import itertools
import json
import math

import numpy as np
import pytest
import torch

from adaptive_tt_dgmres.backend.paper_tt_gmres import tt_dot, tt_scale
from rl_tt_dgmres_v1.problems import ProblemSpec as V1ProblemSpec
from rl_tt_dgmres_v1.problems import build_problem as build_v1
from rl_tt_dgmres_v2.problems import (
    FAMILIES, ProblemSpec, audit_solution, build_problem, make_splits,
    operator_group_key, spec_from_dict,
)


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def dense_oracle(problem):
    n, dims = problem.spec.n, problem.spec.dims
    raw_locals = []
    for edges, weight in zip(problem.local_edge_weights, problem.axis_weights):
        # Incidence construction is independent of production's indexed sums.
        incidence = np.zeros((len(edges), n))
        for edge in range(len(edges)):
            incidence[edge, edge] = 1.0
            incidence[edge, (edge + 1) % n] = -1.0
        raw_locals.append(weight * (n - 1) ** 2 * incidence.T @ np.diag(edges) @ incidence)
    full = np.zeros((n**dims, n**dims))
    for axis, local in enumerate(raw_locals):
        term = np.ones((1, 1))
        for current in range(dims):
            term = np.kron(term, local if axis == current else np.eye(n))
        full += term
    return full / np.linalg.eigvalsh(full)[-1]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("n,dims,rank", [(3, 2, 2), (4, 3, 3)])
def test_dense_spectrum_norm_nullspace_and_drazin(family, n, dims, rank):
    edge_count = n if family.endswith("cycle") else n - 1
    edges = () if family == FAMILIES[0] else tuple(
        tuple(0.6 + (axis + edge + 1) / (2 * n) for edge in range(edge_count))
        for axis in range(dims)
    )
    problem = build_problem(ProblemSpec(
        "dense", n, dims, 73, anisotropy=3.0, rhs_rank=rank,
        family=family, edge_weights=edges,
    ))
    dense = dense_oracle(problem)
    spectrum = np.linalg.eigvalsh(dense)
    assert np.count_nonzero(abs(spectrum) < 1e-12) == 1
    assert spectrum[-1] == pytest.approx(1.0, abs=2e-14)
    assert spectrum[1] == pytest.approx(problem.lambda_min_positive, rel=1e-12)
    q0 = problem.q0.torch().numpy().ravel()
    x = problem.reference_solution.torch().numpy().ravel()
    b = problem.rhs.torch().numpy().ravel()
    np.testing.assert_allclose(dense @ q0, 0.0, atol=3e-16)
    assert abs(q0 @ x) < 1e-13
    assert np.linalg.norm(x) == pytest.approx(1.0, abs=2e-14)
    np.testing.assert_allclose(problem.operator.matvec(problem.reference_solution).torch().numpy().ravel(), dense @ x, atol=2e-14)
    np.testing.assert_allclose(b, dense @ x + 0.1 * np.linalg.norm(dense @ x) * q0, atol=2e-14)
    np.testing.assert_allclose(np.linalg.pinv(dense) @ b, x, atol=2e-12)
    audit = audit_solution(problem, problem.reference_solution, include_reference=True)
    assert audit["finite"] and audit["eta"] < 1e-13
    assert audit["relative_true_residual"] == pytest.approx(0.1 / math.sqrt(1.01), rel=1e-11)
    assert audit["range_defect"] < 1e-13
    assert audit["forward_error"] < 1e-13


def test_v1_specs_and_uniform_weighted_path_are_compatible():
    old_spec = V1ProblemSpec("compatible", 4, 3, 17, axis_weights=(1.0, 2.0, 3.0))
    old, new = build_v1(old_spec), build_problem(old_spec)
    np.testing.assert_allclose(old.rhs.torch().numpy(), new.rhs.torch().numpy(), atol=2e-15)
    assert old.lambda_min_positive == pytest.approx(new.lambda_min_positive)
    weighted = ProblemSpec(**asdict(old_spec), edge_weights=((1.0,) * 3,) * 3)
    weighted = replace(weighted, family="symmetric_weighted_neumann_path")
    assert operator_group_key(old_spec) == operator_group_key(weighted)


def test_group_identity_uses_normalized_operator_and_fingerprint_uses_matrix_bytes():
    original = ProblemSpec(
        "a", 4, 2, 1, family="symmetric_weighted_cycle", axis_weights=(1.0, 2.0),
        edge_weights=((1.0, 2.0, 3.0, 4.0), (4.0, 3.0, 2.0, 1.0)),
    )
    same = replace(original, name="other", seed=999, rhs_rank=4, rhs_frequency=3,
                   contamination=0.0, axis_weights=(2.0, 4.0), curriculum_tier="other")
    scaled_edges = replace(original, edge_weights=tuple(tuple(2 * value for value in axis) for axis in original.edge_weights))
    assert operator_group_key(original) == operator_group_key(same) == operator_group_key(scaled_edges)
    assert operator_group_key(original) != operator_group_key(replace(original, axis_weights=(2.0, 1.0)))
    problem = build_problem(original)
    assert problem.operator_fingerprint == build_problem(same).operator_fingerprint
    metadata = problem.construction_metadata()
    assert len(metadata["operator_fingerprint"]) == 64
    assert metadata["operator_group_key"] == operator_group_key(original)
    assert metadata["edge_weights"] == [list(axis) for axis in original.edge_weights]
    # Matvec's exact matrices are reconstructed without using key labels.
    fingerprint = hashlib.sha256(b"4:2:float64")
    for weight, edges in zip(problem.axis_weights, problem.local_edge_weights):
        local = torch.zeros((4, 4), dtype=torch.float64)
        for edge, value in enumerate(edges):
            right = (edge + 1) % 4
            local[edge, edge] += value
            local[right, right] += value
            local[edge, right] -= value
            local[right, edge] -= value
        local *= 9
        local *= weight / problem.unnormalized_operator_norm
        fingerprint.update(local.numpy().tobytes())
    assert fingerprint.hexdigest() == metadata["operator_fingerprint"]


def test_split_coverage_prefix_stability_and_operator_disjointness():
    splits = make_splits()
    assert splits == make_splits()
    small = make_splits(train_count=3, validation_count=3, test_count=3, ood_count=3)
    groups = {}
    for name, specs in splits.items():
        assert small[name] == specs[:3]
        groups[name] = {operator_group_key(spec) for spec in specs}
        assert len(groups[name]) == len(specs)
        assert {spec.family for spec in specs} == set(FAMILIES)
        assert {spec.contamination for spec in specs} == {0.0, 0.1}
        if name == "ood":
            assert {spec.dims for spec in specs} == {7, 8}
            assert max(spec.n for spec in specs) <= 6
            assert max(spec.rhs_rank for spec in specs) <= 3
        else:
            assert {spec.n for spec in specs} == set(range(4, 13))
            assert {spec.dims for spec in specs} == set(range(2, 7))
            assert {spec.rhs_rank for spec in specs} == set(range(1, 6))
    for left, right in itertools.combinations(groups, 2):
        assert groups[left].isdisjoint(groups[right])
    fingerprints = [build_problem(spec).operator_fingerprint for cases in small.values() for spec in cases]
    assert len(fingerprints) == len(set(fingerprints))


def test_json_roundtrip_and_reference_free_online_view():
    spec = make_splits(train_count=3, validation_count=0, test_count=0, ood_count=0)["train"][0]
    restored = spec_from_dict(json.loads(json.dumps(asdict(spec))))
    assert restored == spec
    problem = build_problem(restored)
    online = problem.online_view()
    assert not {"reference_solution", "seed", "spec", "forward_error", "edge_weights", "curriculum_tier"}.intersection(online)
    with pytest.raises(TypeError):
        online["index"] = 2
    with pytest.raises(ValueError, match="Unknown"):
        spec_from_dict({**asdict(spec), "extra_field": 1})


def test_audit_detects_solution_nullspace_pollution():
    spec = ProblemSpec("range", 4, 3, 29, family="symmetric_weighted_cycle")
    problem = build_problem(spec)
    polluted = problem.reference_solution + tt_scale(problem.q0, 0.3)
    audit = audit_solution(problem, polluted, include_reference=True)
    assert audit["eta"] < 1e-13
    assert audit["range_defect"] == pytest.approx(0.3 / math.sqrt(1.09), rel=1e-12)
    assert audit["forward_error"] == pytest.approx(0.3, rel=1e-12)
    assert abs(tt_dot(problem.q0, problem.projector(polluted))) < 2e-14


def test_high_dimension_construction_is_tt_only_and_preserves_rng(monkeypatch):
    import tntorch as tn

    def forbidden(*args, **kwargs):
        raise AssertionError("Production construction/audit expanded or rounded a TT tensor")

    monkeypatch.setattr(tn.Tensor, "torch", forbidden)
    monkeypatch.setattr(tn.Tensor, "round_tt", forbidden)
    before = torch.random.get_rng_state().clone()
    problem = build_problem(ProblemSpec("ood", 4, 8, 42, rhs_rank=2, family="symmetric_weighted_cycle"))
    audit = audit_solution(problem, problem.reference_solution, include_reference=True)
    assert torch.equal(before, torch.random.get_rng_state())
    assert audit["finite"] and audit["eta"] < 1e-12


@pytest.mark.parametrize("change", [
    {"n": 1}, {"dims": 0}, {"rhs_rank": 0}, {"rhs_frequency": 4},
    {"family": "unknown"}, {"anisotropy": 0}, {"contamination": -0.1},
    {"edge_weights": ((1.0,),)}, {"edge_weights": ((1.0, -1.0, 1.0),) * 2},
    {"axis_weights": (1.0, float("nan"))},
])
def test_invalid_construction_is_rejected(change):
    with pytest.raises(ValueError):
        build_problem(replace(ProblemSpec("bad", 4, 2, 1), **change))
