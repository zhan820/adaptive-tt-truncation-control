"""Small CPU correctness checks for the isolated second-version TT rounder."""
import math

import pytest
import torch
import tntorch as tn

from rl_tt_dgmres_v2.rounding import FastRounder, RoundFailure


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(23)
    yield
    torch.set_num_threads(previous)


def random_tt(shape=(4, 5, 4, 3), rank=4):
    ranks = [1] + [rank] * (len(shape) - 1) + [1]
    return tn.Tensor([torch.randn(ranks[i], size, ranks[i + 1], dtype=torch.float64)
                      for i, size in enumerate(shape)])


@pytest.mark.parametrize("audit_mode", ["tail", "full"])
@pytest.mark.parametrize("eps", [1e-10, 1e-4, 0.15, 0.5])
def test_dense_error_matches_tail_and_independent_audit(audit_mode, eps):
    tensor = random_tt()
    original = [core.clone() for core in tensor.cores]
    dense = tensor.torch()
    rounder = FastRounder(audit_mode=audit_mode)
    rounded = rounder(tensor, eps, phase="unit_test")
    relative_error = float((dense - rounded.torch()).norm() / dense.norm())
    record = rounder.logs[-1]
    assert relative_error <= eps + 5e-14
    assert relative_error == pytest.approx(record["tail_relative_error"], abs=5e-14)
    assert not record["preserved_input"]
    assert not record["is_verified_error_bound"]
    if audit_mode == "full":
        assert record["actual_relative_error"] == pytest.approx(relative_error, abs=5e-14)
        assert record["full_audit_performed"]
    else:
        assert record["actual_relative_error"] is None
        assert not record["full_audit_performed"]
    assert all(torch.equal(before, after) for before, after in zip(original, tensor.cores))


@pytest.mark.parametrize("audit_mode", ["tail", "full"])
def test_rank_cap_cannot_replace_tolerance(audit_mode):
    tensor = tn.Tensor([torch.eye(4, dtype=torch.float64).reshape(1, 4, 4),
                        torch.eye(4, dtype=torch.float64).reshape(4, 4, 1)])
    rounder = FastRounder(audit_mode=audit_mode)
    with pytest.raises(RoundFailure) as failure:
        rounder(tensor, 1e-10, max_rank=1)
    record = failure.value.record
    assert record is rounder.logs[-1]
    assert record["rank_cap_hit"]
    assert record["rank_cap_satisfied"]
    assert record["tail_relative_error"] == pytest.approx(math.sqrt(3) / 2)
    assert not record["within_requested"]
    assert rounder.cumulative_stats()["rejected_rounds"] == 1


def test_valid_caps_compress_and_zero_is_exact():
    tensor = tn.Tensor([torch.eye(3, dtype=torch.float64).reshape(1, 3, 3),
                        torch.diag(torch.tensor([1.0, 0.01, 0.0001], dtype=torch.float64)).reshape(3, 3, 1)])
    rounder = FastRounder()
    rounded = rounder(tensor, 0.02, max_rank=[1])
    assert rounded.cores[0].shape[2] == 1
    assert float((tensor.torch() - rounded.torch()).norm() / tensor.torch().norm()) < 0.02
    zero = random_tt()
    zero.cores[1].zero_()
    rounded_zero = rounder(zero, 0.0, max_rank=1)
    assert float(rounded_zero.torch().norm()) == 0.0
    assert all(core.shape[0] == core.shape[2] == 1 for core in rounded_zero.cores)
    assert rounder.logs[-1]["actual_relative_error"] == 0.0


def test_cancellation_preserves_input_when_ranks_fit():
    large = random_tt(shape=(5, 5, 5), rank=2)
    small = random_tt(shape=(5, 5, 5), rank=2)
    residual = (large + small * 1e-12) - large
    rounder = FastRounder()
    rounded = rounder(residual, 1e-10, max_rank=8)
    assert rounder.logs[-1]["preserved_input"]
    assert rounder.logs[-1]["preserved_reason"] == "full_audit_rejected_candidate"
    assert rounder.logs[-1]["full_audit_performed"]
    assert rounder.logs[-1]["actual_relative_error"] == 0.0
    assert all(torch.equal(a, b) for a, b in zip(residual.cores, rounded.cores))
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(residual.cores, rounded.cores))


def test_cancellation_attempts_canonicalization_before_preserving():
    large = random_tt(shape=(5, 5, 5), rank=2)
    small = random_tt(shape=(5, 5, 5), rank=2)
    residual = (large + small * 1e-4) - large
    rounder = FastRounder()
    result = rounder(residual, 1e-10)
    record = rounder.logs[-1]
    assert record["qr_cancellation_indicator"] > 1e3
    assert record["full_audit_performed"]
    assert not record["preserved_input"]
    assert record["actual_relative_error"] <= 1e-10
    assert record["rank_after"] < record["rank_before"]
    assert float((residual.torch() - result.torch()).norm() / residual.torch().norm()) <= 1e-10


def test_forced_cancellation_compression_gets_independent_audit():
    large = random_tt(shape=(5, 5, 5), rank=2)
    small = random_tt(shape=(5, 5, 5), rank=2)
    residual = (large + small * 1e-12) - large
    rounder = FastRounder()
    with pytest.raises(RoundFailure) as failure:
        rounder(residual, 1e-10, max_rank=2)
    assert failure.value.record["full_audit_performed"]
    assert failure.value.record["actual_relative_error"] > 1e-10
    assert failure.value.record["status"] == "rejected"


def test_zero_tolerance_single_mode_and_statistics():
    tensor = random_tt()
    rounder = FastRounder()
    rounder.start_cycle(8, 1.0, 1e-10)
    exact = rounder(tensor, 0.0)
    assert all(torch.equal(a, b) for a, b in zip(tensor.cores, exact.cores))
    single = tn.Tensor([torch.randn(1, 8, 1, dtype=torch.float64)])
    assert torch.equal(rounder(single, 1e-10).cores[0], single.cores[0])
    stats = rounder.cumulative_stats()
    assert stats["round_calls"] == 2
    assert stats["max_rank"] == 4
    assert stats["round_seconds"] >= 0 and stats["audit_seconds"] >= 0
    assert stats["cycles"] == 1
    rounder.start_cycle()
    assert rounder.cycle_summary()["round_calls"] == 0


def test_zero_tolerance_with_forced_gauge_compression_is_not_silently_accepted():
    rounder = FastRounder()
    with pytest.raises(RoundFailure) as failure:
        rounder(random_tt(shape=(2, 2), rank=4), 0.0, max_rank=2)
    assert failure.value.record["full_audit_performed"]
    assert failure.value.record["tail_relative_error"] == 0.0
    assert failure.value.record["actual_relative_error"] > 0.0


def test_near_machine_precision_request_preserves_existing_representation():
    tensor = random_tt()
    rounder = FastRounder()
    rounded = rounder(tensor, 1e-18)
    assert all(torch.equal(a, b) for a, b in zip(tensor.cores, rounded.cores))
    assert rounder.logs[-1]["preserved_reason"] == "floating_precision_accuracy"


@pytest.mark.parametrize("left_scale,right_scale", [(1e-200, 1e150), (1e150, 1e-200), (1e-300, 1e200)])
@pytest.mark.parametrize("audit_mode", ["tail", "full"])
def test_extreme_gauge_scaling_is_not_misclassified_as_zero(left_scale, right_scale, audit_mode):
    tensor = tn.Tensor([torch.full((1, 2, 1), left_scale, dtype=torch.float64),
                        torch.full((1, 2, 1), right_scale, dtype=torch.float64)])
    rounder = FastRounder(audit_mode=audit_mode)
    rounded = rounder(tensor, 1e-10)
    expected_entry = left_scale * right_scale
    assert torch.allclose(rounded.torch() / expected_entry, torch.ones((2, 2), dtype=torch.float64),
                          atol=1e-12, rtol=1e-12)
    assert rounder.logs[-1]["audit_method"] != "structural_zero"
    assert rounder.logs[-1]["input_norm"] == pytest.approx(2 * expected_entry, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("cap", [0, -1, 1.5, [2], [2, 2, 0], [2, 2, 1.5]])
def test_invalid_rank_caps_are_rejected(cap):
    with pytest.raises((TypeError, ValueError)):
        FastRounder()(random_tt(), 1e-6, max_rank=cap)


def test_tucker_float32_and_nonfinite_inputs_rejected():
    tensor = random_tt()
    tensor.Us[0] = torch.eye(4, dtype=torch.float64)
    with pytest.raises(ValueError, match="pure TT"):
        FastRounder()(tensor, 1e-6)
    tensor.Us[0] = None
    tensor.cores[0] = tensor.cores[0].float()
    with pytest.raises(ValueError, match="float64"):
        FastRounder()(tensor, 1e-6)
    tensor.cores[0] = tensor.cores[0].double()
    tensor.cores[0][0, 0, 0] = math.nan
    with pytest.raises(FloatingPointError, match="finite"):
        FastRounder()(tensor, 1e-6)


def test_tail_mode_does_not_call_independent_norm(monkeypatch):
    import rl_tt_dgmres_v2.rounding as module

    def forbidden(_):
        raise AssertionError("normal tail mode must not form and QR-audit a difference TT")

    monkeypatch.setattr(module, "_independent_norm", forbidden)
    FastRounder()(random_tt(), 1e-6)


def test_decomposition_failure_is_logged_and_recoverable(monkeypatch):
    tensor = random_tt()

    def failed(*args, **kwargs):
        raise torch.linalg.LinAlgError("unit-test decomposition failure")

    monkeypatch.setattr(torch.linalg, "svd", failed)
    rounder = FastRounder()
    with pytest.raises(RoundFailure, match="decomposition failed") as failure:
        rounder(tensor, 1e-6)
    assert failure.value.record is rounder.logs[-1]
    assert failure.value.record["status"] == "rejected"
    assert rounder.cumulative_stats()["rejected_rounds"] == 1


@pytest.mark.parametrize("audit_mode", ["tail", "full"])
def test_absolute_budget_can_compress_small_norm_without_claiming_relative_accuracy(audit_mode):
    values = torch.tensor([1e-10, 2e-12, 1e-13], dtype=torch.float64)
    tensor = tn.Tensor([torch.eye(3, dtype=torch.float64).reshape(1, 3, 3),
                        torch.diag(values).reshape(3, 3, 1)])
    rounder = FastRounder(audit_mode=audit_mode)
    result = rounder(tensor, 1e-10, max_rank=1, absolute_tolerance=3e-12)
    record = rounder.logs[-1]
    error = float((tensor.torch() - result.torch()).norm())
    assert error <= 3e-12
    assert record["ranks_after"] == [1]
    assert record["budget_mode"] == "absolute"
    assert record["requested_absolute_error"] == 3e-12
    assert record["effective_relative_eps"] == pytest.approx(3e-12 / float(values.norm()))
    assert record["within_requested"] and record["within_requested_absolute"]
    assert not record["within_requested_relative"]
    assert not record["is_verified_error_bound"]


@pytest.mark.parametrize("audit_mode", ["tail", "full"])
def test_absolute_budget_removes_near_cancelled_ghost_without_preserving_raw_ranks(audit_mode):
    large = random_tt(shape=(5, 5, 5), rank=2)
    small = random_tt(shape=(5, 5, 5), rank=2)
    residual = (large + small * 1e-12) - large
    budget = 1e-8
    rounder = FastRounder(audit_mode=audit_mode)
    result = rounder(residual, 1e-10, max_rank=2, absolute_tolerance=budget)
    record = rounder.logs[-1]
    assert float((residual.torch() - result.torch()).norm()) < budget
    assert float(result.torch().norm()) == 0.0
    assert record["zeroed_with_absolute_budget"]
    assert not record["preserved_input"]
    assert record["ranks_after"] == [1, 1]
    assert not record["within_requested_relative"]
    assert record["within_requested_absolute"]
    assert rounder.cumulative_stats()["absolute_budget_calls"] == 1


def test_absolute_cap_violation_does_not_disappear_when_squared_error_underflows():
    values = torch.tensor([1e-190, 1e-192], dtype=torch.float64)
    tensor = tn.Tensor([torch.eye(2, dtype=torch.float64).reshape(1, 2, 2),
                        torch.diag(values).reshape(2, 2, 1)])
    rounder = FastRounder()
    with pytest.raises(RoundFailure) as failure:
        rounder(tensor, 1e-10, max_rank=1, absolute_tolerance=1e-193)
    assert failure.value.record["tail_absolute_error"] == pytest.approx(1e-192, rel=1e-12, abs=0.0)
    assert not failure.value.record["within_requested_absolute"]


def test_zero_absolute_budget_and_invalid_absolute_requests():
    tensor = random_tt()
    rounder = FastRounder()
    result = rounder(tensor, 0.1, absolute_tolerance=0.0)
    assert all(torch.equal(a, b) for a, b in zip(result.cores, tensor.cores))
    assert rounder.logs[-1]["budget_mode"] == "absolute"
    for value in (-1.0, math.inf, math.nan):
        with pytest.raises(ValueError, match="absolute_tolerance"):
            rounder(tensor, 0.1, absolute_tolerance=value)


def test_rejected_full_zero_candidate_is_not_logged_as_returned_zero(monkeypatch):
    import rl_tt_dgmres_v2.rounding as module

    tensor = tn.Tensor([torch.ones((1, 2, 1), dtype=torch.float64),
                        torch.ones((1, 2, 1), dtype=torch.float64)])
    norms = iter([4.0, 2.0])
    monkeypatch.setattr(module, "_independent_norm", lambda _: next(norms))
    rounder = FastRounder(audit_mode="full")
    result = rounder(tensor, 1e-10, absolute_tolerance=3.0)
    record = rounder.logs[-1]
    assert all(torch.equal(a, b) for a, b in zip(result.cores, tensor.cores))
    assert record["candidate_zeroed_with_absolute_budget"]
    assert not record["zeroed_with_absolute_budget"]
    assert record["tail_absolute_error"] == record["actual_absolute_error"] == 0.0
    assert record["candidate_actual_absolute_error"] == 4.0
