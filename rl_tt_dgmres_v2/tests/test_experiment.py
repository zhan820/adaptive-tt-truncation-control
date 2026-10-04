"""Aggregation and split routing tests using small deterministic mocks."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from rl_tt_dgmres_v2 import experiment
from rl_tt_dgmres_v2.problems import ProblemSpec


def row(name, mode, seconds, accepted=True, split="test", repeat=0):
    return {
        "problem": {"name": name}, "mode": mode, "split": split,
        "repeat": repeat, "wall_seconds": seconds, "accepted": accepted,
        "iterations": 12, "status": "eta_target" if accepted else "time_budget",
        "recoveries": 0, "shield_interventions": 0, "executed_action_indices": [3],
    }


def test_paired_speedups_take_per_instance_repeat_medians_before_geometric_mean():
    rows = []
    samples = {
        "test_a": {"base": [1.0, 100.0, 101.0], "candidate": [10.0, 11.0, 1000.0]},
        "test_b": {"base": [4.0, 5.0, 6.0], "candidate": [1.0, 1.0, 1.0]},
    }
    for name, modes in samples.items():
        for mode, times in modes.items():
            rows.extend(row(name, mode, seconds, repeat=repeat) for repeat, seconds in enumerate(times))
    result = experiment.paired_speedups(rows, "base", "candidate", "test")
    assert result["common_successful_instances"] == 2
    assert [pair["baseline_seconds"] for pair in result["pairs"]] == [100.0, 5.0]
    assert [pair["candidate_seconds"] for pair in result["pairs"]] == [11.0, 1.0]
    assert result["geometric_mean_speedup"] == pytest.approx(math.sqrt((100.0 / 11.0) * 5.0))
    # Pooling all repeats first would produce a ratio of 1, a different statistic.
    assert result["geometric_mean_speedup"] != pytest.approx(1.0)


def test_failed_repeats_excluded_from_pairs_but_retained_in_aggregate():
    rows = [
        row("test_good", "base", 8.0), row("test_good", "candidate", 2.0),
        row("test_failed", "base", 5.0),
        row("test_failed", "candidate", 1.0, repeat=0),
        row("test_failed", "candidate", 30.0, accepted=False, repeat=1),
        row("test_missing", "base", 2.0),
        row("ood_separate", "base", 100.0, split="ood"),
        row("ood_separate", "candidate", 1.0, split="ood"),
    ]
    pairs = experiment.paired_speedups(rows, "base", "candidate", "test")
    assert [pair["problem"] for pair in pairs["pairs"]] == ["test_good"]
    assert pairs["geometric_mean_speedup"] == pytest.approx(4.0)
    candidate_rows = [entry for entry in rows if entry["split"] == "test" and entry["mode"] == "candidate"]
    summary = experiment.aggregate(candidate_rows)
    assert summary["runs"] == 3
    assert summary["successes"] == 2
    assert summary["instances"] == 2
    assert summary["successful_instances_all_repeats"] == 1
    assert summary["status_counts"] == {"eta_target": 2, "time_budget": 1}


def test_empty_or_no_common_successful_pairs_remain_explicit():
    result = experiment.paired_speedups([], "base", "candidate")
    assert result["common_successful_instances"] == 0
    assert result["geometric_mean_speedup"] is None
    aggregate = experiment.aggregate([])
    assert aggregate["runs"] == 0 and aggregate["instances"] == 0
    assert aggregate["median_success_seconds"] is None


@pytest.mark.parametrize("faster_test_seed", [11, 22])
def test_two_seed_selection_precedes_heldout_and_uses_only_validation_scores(
    monkeypatch, tmp_path, faster_test_seed,
):
    # Keep real main routing and selection, but replace numerical work/files
    # with observable mocks. Changing the heldout winner must not select it.
    specs = {
        split: [ProblemSpec(f"{split}_only", 4, 2, index + 1)]
        for index, split in enumerate(("train", "validation", "test", "ood"))
    }
    events, saved_json, copied = [], {}, []

    def fake_build(spec):
        return SimpleNamespace(spec=spec, construction_metadata=lambda: {"name": spec.name})

    def fake_write(path, payload):
        saved_json[str(path)] = deepcopy(payload)

    def fake_evaluate(problems, config, policy, output):
        assert [problem.spec.name for problem in problems] == ["validation_only"]
        events.append("validation_grid")
        action = policy([0.0] * len(experiment.FEATURE_NAMES))
        return [row("validation_only", "validation", 1.0 + action, split="validation")]

    def fake_train(problems, validation, config, output, args, seed, fixed_index):
        assert [problem.spec.name for problem in problems] == ["train_only"]
        assert [problem.spec.name for problem in validation] == ["validation_only"]
        assert not any(event.startswith("heldout") for event in events)
        events.append(f"train_{seed}")
        action = 1 if seed == 11 else 2
        agent = SimpleNamespace(select_action=lambda obs, epsilon, action=action: action)
        # Seed 22 is validation-best regardless of its eventual test time.
        return agent, {"seed": seed, "best_score": (1, -10.0 if seed == 11 else -5.0)}

    def fake_copy(source, destination):
        assert "train_11" in events and "train_22" in events
        assert not any(event.startswith("heldout") for event in events)
        events.append("select_from_validation")
        copied.append((str(source), str(destination)))

    def fake_run(problem, config, policy, evaluate_reference=True):
        name = problem.spec.name
        if name == "train_only":
            assert not evaluate_reference  # Kernel warmup is a training problem.
            events.append("warmup_train")
        else:
            assert name in {"test_only", "ood_only"}
            assert "select_from_validation" in events
            assert evaluate_reference  # Forward error is evaluation-only.
            events.append(f"heldout_{name}")
        action = policy([0.0] * len(experiment.FEATURE_NAMES))
        seed = 11 if action == 1 else 22 if action == 2 else None
        seconds = 0.1 if seed == faster_test_seed else 10.0
        return {"summary": row(name, "pending", seconds), "transitions": []}

    monkeypatch.setattr(experiment, "make_splits", lambda *args: specs)
    monkeypatch.setattr(experiment, "build_problem", fake_build)
    monkeypatch.setattr(experiment, "write_json", fake_write)
    monkeypatch.setattr(experiment, "source_hashes", lambda: {"source": "unchanged"})
    monkeypatch.setattr(experiment, "save_snapshot", lambda *args: None)
    monkeypatch.setattr(experiment, "evaluate_policy", fake_evaluate)
    monkeypatch.setattr(experiment, "train_one", fake_train)
    monkeypatch.setattr(experiment.shutil, "copy2", fake_copy)
    monkeypatch.setattr(experiment, "run_episode", fake_run)
    monkeypatch.setattr(experiment, "make_report", lambda *args: None)
    monkeypatch.setattr(experiment.torch, "set_num_threads", lambda count: None)
    output = tmp_path / "mock_experiment"
    experiment.main([
        "--output", str(output), "--training-seeds", "11", "22",
        "--train-count", "1", "--validation-count", "1", "--test-count", "1",
        "--ood-count", "1", "--episodes", "2", "--eval-every", "1",
        "--repeats", "1", "--skip-legacy",
    ])
    manifest = saved_json[str(output.resolve() / "manifest.json")]
    assert manifest["stage"] == "complete"
    assert manifest["chosen_training_seed"] == 22
    assert len(copied) == 1 and "seed_22" in copied[0][0]
    heldout_rows = saved_json[str(output.resolve() / "evaluation_summary.json")]
    assert {entry["mode"] for entry in heldout_rows} == {"v2_fixed", "v2_heuristic", "neural_11", "neural_22"}
    assert {entry["split"] for entry in heldout_rows} == {"test", "ood"}
