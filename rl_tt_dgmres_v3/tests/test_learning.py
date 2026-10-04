"""Synthetic unit fixtures test mechanics, not measured solver performance."""

import copy
import numpy as np
import pytest
import torch

from rl_tt_dgmres_v3.learning import GateConfig, LearningConfig, RolloutValueLearner


def rows(policy="baseline", count=8):
    result = []
    for state in range(count):
        feature = [float(state) / max(1, count - 1), 3.0]
        for action in range(9):
            result.append({
                "features": feature, "action": action, "remaining_wall": 1.0 + action / 4.0,
                "time_budget": 10.0, "success": action != 8,
                "data_origin": "real_rollout", "continuation_policy": policy,
                "state_id": f"test-fixture-{state}", "problem_id": "unit-test-only",
                "split": "train",
            })
    return result


def learner(epochs=3):
    return RolloutValueLearner(config=LearningConfig(epochs=epochs, ensemble_size=2, hidden_dim=8, batch_size=128))


def fake_prediction(cost=None, spread=None, risk=None, risk_spread=None, zmax=0.0):
    return {
        "cost_mean": [1.0] * 9 if cost is None else list(cost),
        "cost_std": [0.0] * 9 if spread is None else list(spread),
        "failure_mean": [0.0] * 9 if risk is None else list(risk),
        "failure_std": [0.0] * 9 if risk_spread is None else list(risk_spread),
        "action_counts": [10] * 9, "max_standardized_feature": zmax,
        "target_policy": "baseline", "uncertainty_is_certified": False,
    }


def test_dynamic_input_and_training_only_scaling():
    model = learner()
    summary = model.fit(rows(), source_data=["real-rollout-fixture"])
    assert model.input_dim == 2
    assert summary["bootstrap_groups"] == 8
    assert summary["failure_count"] == 8
    assert summary["successful_action_pairs"] == 8 * 28
    np.testing.assert_allclose(model.feature_mean, [0.5, 3.0])
    assert model.feature_scale[1] == 1.0
    original_mean, original_scale = model.feature_mean.copy(), model.feature_scale.copy()
    prediction = model.predict_costs([9.0, 3.0])
    assert len(prediction["cost_mean"]) == 9
    assert min(prediction["cost_mean"]) >= 0.0
    assert all(0 <= p <= 1 for p in prediction["failure_mean"])
    np.testing.assert_array_equal(model.feature_mean, original_mean)
    np.testing.assert_array_equal(model.feature_scale, original_scale)
    assert prediction["uncertainty_is_certified"] is False
    np.testing.assert_allclose(
        prediction["cost_mean"],
        np.asarray(prediction["time_mean"]) + model.config.failure_penalty * np.asarray(prediction["failure_mean"]),
    )


def test_observed_cost_includes_failure_and_does_not_clip_budget_overruns():
    data = rows(count=1)
    data[8]["remaining_wall"] = 30.0
    model = learner()
    _, _, costs, failures, _, _ = model._rows(data, "baseline")
    assert costs[8] == 8.0
    assert failures[8] == 1.0
    assert costs[0] == 0.1


def test_explicit_target_is_supported_but_penalty_omission_rejected():
    data = rows(count=1)
    data[0].update(cost_target=6.0, failure_target=1.0)
    model = learner()
    assert model._rows(data, "baseline")[2][0] == 6.0
    data[0]["cost_target"] = 0.1
    with pytest.raises(ValueError, match="failure penalty"):
        model.fit(data)


@pytest.mark.parametrize("field,value", [
    ("data_origin", "model_prediction"), ("continuation_policy", "different_policy"),
    ("split", "validation"), ("split", "test"), ("action", 9), ("action", -1),
    ("action", True), ("features", [float("nan"), 0]), ("features", []),
    ("time_budget", 0), ("remaining_wall", -1), ("success", 1),
])
def test_bad_rows_are_rejected(field, value):
    data = rows(count=1)
    data[0][field] = value
    with pytest.raises(ValueError):
        learner().fit(data)


def test_no_implicit_model_targets_or_empty_dataset():
    with pytest.raises(ValueError):
        learner().fit([])
    data = rows(count=1)
    del data[0]["data_origin"]
    with pytest.raises(ValueError, match="real_rollout"):
        learner().fit(data)


def test_inconsistent_feature_width_is_rejected():
    data = rows(count=1)
    data[0]["features"] = [1.0]
    with pytest.raises(ValueError, match="dimensions"):
        learner().fit(data)


def test_iteration_uses_new_real_policy_targets_and_frozen_warm_scaler():
    model = learner()
    model.fit(rows(), source_data=["round0.json"])
    previous_mean = model.feature_mean.copy()
    data = rows(policy="improved_1")
    for row in data:
        row["features"][0] += 5.0
        row["remaining_wall"] *= 0.8
    summary = model.fit(data, target_policy="improved_1", source_data=["round1.json"], warm_start=True)
    assert model.fit_calls == 2
    assert summary["warm_start"] is True
    assert summary["scaler_refitted"] is False
    assert model.metadata["target_policy"] == "improved_1"
    assert model.metadata["fit_history"][0]["target_policy"] == "baseline"
    assert model.metadata["fit_history"][1]["source_data"] == ["round1.json"]
    np.testing.assert_array_equal(previous_mean, model.feature_mean)


def test_retraining_without_warm_start_refits_only_supplied_training_rows():
    model = learner()
    model.fit(rows())
    data = rows(policy="p1")
    for row in data:
        row["features"] = [100.0, 7.0]
    model.fit(data, target_policy="p1", warm_start=False)
    np.testing.assert_array_equal(model.feature_mean, [100.0, 7.0])


def test_fit_and_predict_do_not_consume_global_rng_or_change_threads():
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = copy.deepcopy(np.random.get_state())
    threads = torch.get_num_threads()
    model = learner()
    model.fit(rows())
    model.predict_costs([0.2, 3.0])
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    after = np.random.get_state()
    assert after[0] == numpy_state[0] and after[2:] == numpy_state[2:]
    np.testing.assert_array_equal(after[1], numpy_state[1])
    assert torch.get_num_threads() == threads


def test_prediction_is_deterministic_and_works_inside_inference_mode():
    model = learner()
    model.fit(rows())
    expected = model.predict_costs([0.2, 3.0])
    with torch.inference_mode():
        observed = model.predict_costs(torch.tensor([0.2, 3.0], dtype=torch.float64))
    assert expected == observed
    with torch.inference_mode(), pytest.raises(RuntimeError, match="outside"):
        model.fit(rows())


def test_gate_accepts_credible_gain_and_counts_query_overhead(monkeypatch):
    model = learner()
    model.action_counts[:] = 10
    prediction = fake_prediction(cost=[1.0, 0.8] + [1.0] * 7)
    monkeypatch.setattr(model, "predict_costs", lambda _: prediction)
    selected, decision = model.policy([0.0], baseline=0)
    assert selected == 1 and decision["deviated"]
    gate = GateConfig(decision_overhead_fraction=0.3)
    selected, decision = model.policy([0.0], baseline=0, gate=gate)
    assert selected == 0
    assert decision["guarantees_wallclock_improvement"] is False


@pytest.mark.parametrize("case", ["cost_spread", "risk", "risk_spread", "counts", "base_counts", "ood"])
def test_gate_rejects_uncertain_or_risky_advantage(monkeypatch, case):
    model = learner()
    model.action_counts[:] = 10
    prediction = fake_prediction(cost=[1.0, 0.7] + [1.0] * 7)
    if case == "cost_spread":
        prediction["cost_std"][1] = 0.4
    elif case == "risk":
        prediction["failure_mean"][1] = 0.3
    elif case == "risk_spread":
        prediction["failure_std"][1] = 0.1
    elif case == "counts":
        model.action_counts[1] = 0
    elif case == "base_counts":
        model.action_counts[0] = 0
    else:
        prediction["max_standardized_feature"] = 100.0
    monkeypatch.setattr(model, "predict_costs", lambda _: prediction)
    selected, decision = model.policy([0.0], baseline=0)
    assert selected == 0
    assert decision["rejections"]["1"]


def test_checkpoint_roundtrip_predictions_and_metadata(tmp_path):
    model = learner()
    model.fit(rows(), source_data=["actual_data.json"])
    path = tmp_path / "controller.pt"
    model.save(path, metadata={"selected_on": "validation"})
    restored = RolloutValueLearner.load(path)
    assert restored.predict_costs([0.4, 3.0]) == model.predict_costs([0.4, 3.0])
    assert restored.metadata["source_data"] == ["actual_data.json"]
    assert restored.metadata["selected_on"] == "validation"
    assert restored.fit_calls == 1
    with pytest.raises(ValueError, match="protected"):
        model.save(path, metadata={"target_policy": "incorrect"})


@pytest.mark.parametrize("change", ["format", "scale", "counts", "weights", "history"])
def test_checkpoint_corruption_is_rejected(tmp_path, change):
    model = learner(epochs=1)
    model.fit(rows(count=2))
    path = tmp_path / "controller.pt"
    model.save(path)
    checkpoint = torch.load(path, weights_only=True)
    if change == "format":
        checkpoint["format_version"] = -1
    elif change == "scale":
        checkpoint["feature_scale"][0] = 0
    elif change == "counts":
        checkpoint["action_counts"][0] = -1
    elif change == "weights":
        next(iter(checkpoint["models"][0].values())).fill_(float("nan"))
    else:
        checkpoint["metadata"]["fit_history"] = []
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        RolloutValueLearner.load(path)


def test_unfitted_and_bad_configuration_are_rejected(tmp_path):
    model = learner()
    with pytest.raises(RuntimeError):
        model.predict_costs([1.0])
    with pytest.raises(RuntimeError):
        model.save(tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        LearningConfig(ensemble_size=0)
    with pytest.raises(ValueError):
        GateConfig(max_failure_probability=1.1)
    with pytest.raises(ValueError):
        GateConfig(min_gain=-1)
