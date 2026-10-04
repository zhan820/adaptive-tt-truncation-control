from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from rl_tt_dgmres_neumann import experiment as ex


def arguments(output, **changes):
    values = dict(output=str(output), seed=202610021, train_count=3,
                  validation_count=3, test_count=3, train_dims="4,5,6", paper_dims="4,5,6,10",
                  seconds_budget=60.0, paper_seconds=60.0, max_iterations=300,
                  tolerance=1e-7, max_tt_rank=96, recovery_max_rank=256,
                  diagnostic_max_rank=512, max_states=2, repeats=3,
                  threads=1, iteration=0, epochs=2, limit=0)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    sources = []
    for name in ("builder.json", "setup.tex"):
        path = tmp_path / name
        path.write_text("unit-test provenance only", encoding="utf-8")
        sources.append(path)
    monkeypatch.setattr(ex, "PAPER_SOURCES", tuple(sources))
    monkeypatch.setattr(ex, "BUILDER_SOURCE", {"path": str(sources[0]), "sha256": ex.sha256(sources[0])})
    monkeypatch.setattr(ex, "hashes", lambda: {})
    args = arguments(tmp_path / "temporary_protocol")
    ex.prepare(args)
    return args


def test_splits_are_rhs_disjoint_but_deliberately_share_operators():
    splits = ex.make_splits(123, 18, 6, 6, (4, 5, 6))
    assert splits == ex.make_splits(123, 18, 6, 6, (4, 5, 6))
    ex._validate_splits(splits, (4, 5, 6))
    seeds = [s.seed for role in ("train", "validation", "test") for s in splits[role]]
    assert len(set(seeds)) == 30
    assert ex.PAPER_SEED not in seeds
    assert not set(seeds).intersection(ex.DEVELOPMENT_SEEDS)
    operators = [{ex.operator_group_key(s) for s in splits[role]}
                 for role in ("train", "validation", "test")]
    assert operators[0] == operators[1] == operators[2]
    assert all(sum(s.dims == d for s in splits["train"]) == 6 for d in (4, 5, 6))
    assert {(s.dims, s.seed) for s in splits["paper"]} == {
        (d, ex.PAPER_SEED) for d in (4, 5, 6, 10)}


def test_paper_dimensions_are_explicit_and_do_not_expand_a_scoped_run():
    splits = ex.make_splits(123, 6, 2, 4, (4,), paper_dimensions=(4,))
    ex._validate_splits(splits, (4,), (4,))
    assert len(splits["paper"]) == 1
    assert splits["paper"][0].dims == 4
    assert all(s.dims == 4 for cases in splits.values() for s in cases)
    with pytest.raises(ValueError):
        ex._validate_splits(splits, (4,), (4, 5, 6, 10))


@pytest.mark.parametrize("dimensions", [(), (4, 4), (7,), (4, 10, 12)])
def test_invalid_paper_scope_is_rejected(dimensions):
    with pytest.raises(ValueError):
        ex.make_splits(1, 3, 3, 3, (4, 5, 6), dimensions)


def test_rng_resamples_all_reserved_development_and_paper_seeds(monkeypatch):
    sequence = iter([*ex.DEVELOPMENT_SEEDS, ex.PAPER_SEED, 101, 102, 103])
    class FakeRandom:
        def __init__(self, seed):
            pass
        def randrange(self, *args):
            return next(sequence)
        def shuffle(self, values):
            pass
    monkeypatch.setattr(ex.random, "Random", FakeRandom)
    splits = ex.make_splits(4, 1, 1, 1, (4,), (4,))
    assert [splits[role][0].seed for role in ("train", "validation", "test")] == [101, 102, 103]


@pytest.mark.parametrize("role", ["train", "validation", "test"])
@pytest.mark.parametrize("seed", ex.DEVELOPMENT_SEEDS)
def test_development_seed_is_rejected_in_each_rhs_split(role, seed):
    splits = ex.make_splits(4, 1, 1, 1, (4,), (4,))
    splits[role][0] = replace(splits[role][0], seed=seed)
    with pytest.raises(ValueError, match="development-case leakage"):
        ex._validate_splits(splits, (4,), (4,))


@pytest.mark.parametrize("dimensions,counts", [((4, 4), (3, 3, 3)),
    ((4, 5, 6), (2, 3, 3)), ((4, 5, 6), (3, 2, 3)), ((), (3, 3, 3))])
def test_invalid_split_protocol_rejected(dimensions, counts):
    with pytest.raises(ValueError):
        ex.make_splits(1, *counts, dimensions)


@pytest.mark.parametrize("kind", ["paper_seed", "cross_split", "family"])
def test_split_validation_rejects_leakage_and_family_changes(kind):
    splits = ex.make_splits(4, 3, 3, 3, (4, 5, 6))
    old = splits["test"][0]
    if kind == "paper_seed":
        new = replace(old, seed=ex.PAPER_SEED)
    elif kind == "cross_split":
        new = replace(old, seed=splits["train"][0].seed)
    else:
        new = replace(old, cp_rank=4)
    splits["test"][0] = new
    with pytest.raises(ValueError):
        ex._validate_splits(splits, (4, 5, 6))


def test_preparation_preserves_protocol_and_paper_provenance(prepared):
    out, manifest, config, specs = ex.load_run(prepared)
    assert config.tolerance == 1e-7
    assert (config.max_tt_rank, config.recovery_max_tt_rank, config.diagnostic_max_rank) == (96, 256, 512)
    assert manifest["threads"] == 1
    assert manifest["split_contract"]["operator_disjoint"] is False
    assert manifest["paper_protocol"]["tolerance_by_dimension"]["10"] == 1e-5
    assert len(manifest["paper_provenance"]) == 2
    assert all(ex.sha256(out / item["snapshot"]) == item["sha256"]
               for item in manifest["paper_provenance"])
    assert len(specs["paper"]) == 4
    assert manifest["split_contract"]["excluded_development_seeds"] == list(ex.DEVELOPMENT_SEEDS)


def test_scoped_prepare_persists_paper_dimensions_and_tolerance_map(prepared):
    from pathlib import Path
    args = arguments(Path(prepared.output).with_name("scoped_protocol"),
                     train_dims="4", paper_dims="4", train_count=6,
                     validation_count=2, test_count=4)
    ex.prepare(args)
    out, manifest, _, specs = ex.load_run(args)
    assert manifest["paper_protocol"]["dimensions"] == [4]
    assert manifest["paper_protocol"]["tolerance_by_dimension"] == {"4": 1e-7}
    assert len(specs["paper"]) == 1
    assert specs["paper"][0].dims == 4
    assert ex.read_json(out / "prepared_contract.json")["paper_protocol"]["dimensions"] == [4]


@pytest.mark.parametrize("field", ["paper_dimensions", "development_seeds"])
def test_resume_rejects_scope_or_exclusion_tampering(prepared, field):
    out, manifest, _, _ = ex.load_run(prepared)
    if field == "paper_dimensions":
        manifest["paper_protocol"]["dimensions"] = [4]
    else:
        manifest["split_contract"]["excluded_development_seeds"] = []
    ex.write_json(out / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match="Prepared protocol changed"):
        ex.load_run(prepared)


@pytest.mark.parametrize("tamper", ["splits", "config", "paper_snapshot"])
def test_resume_rejects_protocol_changes(prepared, tamper):
    out, manifest, _, _ = ex.load_run(prepared)
    if tamper == "splits":
        values = ex.read_json(out / "splits.json")
        values["test"][0]["seed"] += 1
        ex.write_json(out / "splits.json", values)
    elif tamper == "config":
        manifest["config"]["tolerance"] = 1e-4
        ex.write_json(out / "manifest.json", manifest)
    else:
        (out / manifest["paper_provenance"][0]["snapshot"]).write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError):
        ex.load_run(prepared)


@pytest.mark.parametrize("stage,paper_started", [("frozen", False), ("evaluating", False),
    ("complete", False), ("fixed_selected", True)])
def test_freeze_or_paper_start_immediately_blocks_further_learning(prepared, stage, paper_started):
    out, manifest, _, _ = ex.load_run(prepared)
    manifest.update(stage=stage, paper_started=paper_started)
    ex.write_json(out / "manifest.json", manifest)
    with pytest.raises(RuntimeError):
        ex.load_run(prepared, training=True)


def test_fixed_grid_selects_dimension_specific_baseline_without_test_use(prepared, monkeypatch):
    from rl_tt_dgmres_v3 import runner
    calls = []
    monkeypatch.setattr(ex, "warm_up", lambda *args: None)
    monkeypatch.setattr(ex, "build_problem", lambda spec: SimpleNamespace(spec=spec))
    def fake_run(problem, config, policy, **kwargs):
        action = policy([0.0] * 32)
        calls.append((problem.spec.name, kwargs))
        desired = {4: 0, 5: 3, 6: 6}[problem.spec.dims]
        return {"summary": {"problem": asdict(problem.spec), "accepted": True,
                            "wall_seconds": 1.0 + abs(action - desired)}}
    monkeypatch.setattr(runner, "run_episode", fake_run)
    ex.fixed_grid(prepared)
    out, manifest, _, specs = ex.load_run(prepared)
    assert manifest["fixed_by_dimension"] == {"4": 0, "5": 3, "6": 6}
    assert manifest["best_fixed_index"] == 3
    assert len(calls) == 27
    assert all(name.startswith("validation_") for name, _ in calls)
    assert all(options["compute_policy_features"] is False for _, options in calls)
    assert ex.baseline_for(manifest, specs["paper"][-1]) == 3
    ex._baseline_contract(out, manifest)


def test_evaluation_policies_use_dimension_baseline_and_static_queries_once(monkeypatch):
    manifest = {"best_fixed_index": 3, "fixed_by_dimension": {"4": 0}, "selected": {}}
    spec = SimpleNamespace(dims=4)
    query = []
    def make(out, selection, baseline, decisions=None):
        assert baseline == 0
        def learned(observation):
            query.append(observation)
            return len(query)
        return learned
    monkeypatch.setattr(ex, "make_learned_policy", make)
    assert ex._evaluation_policy(None, manifest, spec, "fixed", [])([]) == 3
    assert ex._evaluation_policy(None, manifest, spec, "fixed_by_dimension", [])([]) == 0
    static = ex._evaluation_policy(None, manifest, spec, "rl_static", [])
    assert static([1]) == static([2]) == 1
    assert len(query) == 1


def test_complete_matrix_rejects_duplicates_and_missing_runs():
    specs = {"test": [SimpleNamespace(name="case")]}
    rows = [{"split": "test", "problem": {"name": "case"}, "mode": mode, "repeat": repeat}
            for mode in ex.EVALUATION_MODES for repeat in range(3)]
    ex._assert_complete(rows, specs, ex.EVALUATION_MODES, 3)
    for bad in (rows[:-1], rows + rows[:1], rows[:-1] + rows[:1]):
        with pytest.raises(ValueError):
            ex._assert_complete(bad, specs, ex.EVALUATION_MODES, 3)


def test_stage_protocol_rejects_changed_resume(tmp_path):
    path = tmp_path / "protocol.json"
    ex.ensure_protocol(path, {"baseline": {"4": 0}, "target": "baseline"})
    ex.ensure_protocol(path, {"baseline": {"4": 0}, "target": "baseline"})
    with pytest.raises(RuntimeError):
        ex.ensure_protocol(path, {"baseline": {"4": 3}, "target": "baseline"})


def test_trained_model_helper_checks_hash_and_passes_baseline(tmp_path, monkeypatch):
    from rl_tt_dgmres_v3 import learning
    path = tmp_path / "fake_test_model.pt"
    path.write_bytes(b"unit test stand-in, not trained data")
    class FakeLearner:
        @classmethod
        def load(cls, path):
            return cls()
        def policy(self, observation, baseline, gate):
            return baseline, {"baseline": baseline}
    monkeypatch.setattr(learning, "RolloutValueLearner", FakeLearner)
    decisions = []
    policy = ex.load_controller(path, {}, 6, expected_sha256=ex.sha256(path), decisions=decisions)
    assert policy([0] * 32) == 6
    assert decisions == [{"baseline": 6}]
    with pytest.raises(RuntimeError):
        ex.load_controller(path, {}, 6, expected_sha256="wrong")


def test_training_reader_rejects_incomplete_collection(prepared):
    out, manifest, _, specs = ex.load_run(prepared)
    folder = out / "rollouts_0"
    ex.write_json(folder / "summary.json", {"complete": False})
    ex.write_json(folder / "collection_protocol.json", {})
    with pytest.raises(ValueError):
        ex._training_rows(out, manifest, specs, 0)


def fake_complete_collection(out, manifest, specs):
    folder = out / "rollouts_0"
    folder.mkdir()
    baseline = {"global": 3, "by_dimension": {"4": 0, "5": 3, "6": 6}, "validation_sha256": "test"}
    manifest.update(best_fixed_index=3, fixed_by_dimension=baseline["by_dimension"])
    protocol = {"specs": [asdict(s) for s in specs["train"]], "source_sha256": manifest["source_sha256"],
                "config": manifest["config"], "baseline": baseline}
    ex.write_json(folder / "collection_protocol.json", protocol)
    rows, receipts = [], {}
    for spec in specs["train"]:
        checkpoint = folder / f"{spec.name}_states.pt"
        checkpoint.write_bytes(b"mock checkpoint provenance, no tensor execution")
        trajectory = folder / f"{spec.name}_collection.json"
        ex.write_json(trajectory, {"unit_test_mock": True})
        receipt = folder / f"{spec.name}_states_receipt.json"
        ex.write_json(receipt, {"checkpoint_sha256": ex.sha256(checkpoint),
                               "trajectory_sha256": ex.sha256(trajectory)})
        receipts[str(receipt.relative_to(out))] = ex.sha256(receipt)
        for action in range(9):
            path = folder / f"{spec.name}_s0_a{action}.json"
            row = {"features": [0.0] * 32, "action": action, "remaining_wall": 2.0,
                   "time_budget": manifest["config"]["seconds_budget"], "success": True,
                   "data_origin": "real_rollout", "continuation_policy": "baseline",
                   "problem_id": spec.name, "state_id": f"{spec.name}/0", "split": "train",
                   "baseline_index": ex.baseline_for(manifest, spec), "instance_key": ex.instance_key(spec),
                   "source": str(path.relative_to(out))}
            ex.write_json(path, {"row": row, "detail": {"summary": {
                "wall_seconds": 2.0, "accepted": True, "problem": asdict(spec)}}})
            rows.append(row)
    ex.write_json(folder / "rows.json", rows)
    summary = {"complete": True, "problems": len(specs["train"]), "rows": len(rows),
               "states": len(specs["train"]), "rows_sha256": ex.sha256(folder / "rows.json"),
               "raw_files_sha256": {r["source"]: ex.sha256(out / r["source"]) for r in rows},
               "state_receipts_sha256": receipts}
    ex.write_json(folder / "summary.json", summary)
    return baseline, rows


def test_training_reader_preserves_exact_remaining_cost_and_dimension_baseline(prepared, monkeypatch):
    out, manifest, _, specs = ex.load_run(prepared)
    baseline, rows = fake_complete_collection(out, manifest, specs)
    monkeypatch.setattr(ex, "_baseline_contract", lambda *args: baseline)
    loaded = ex._training_rows(out, manifest, specs, 0)
    assert loaded == rows
    assert {r["time_budget"] for r in loaded} == {60.0}
    assert {r["baseline_index"] for r in loaded} == {0, 3, 6}


@pytest.mark.parametrize("tamper", ["raw_return", "checkpoint", "wrong_cost", "wrong_split", "missing_action"])
def test_training_reader_refuses_invented_or_changed_returns(prepared, monkeypatch, tamper):
    out, manifest, _, specs = ex.load_run(prepared)
    baseline, rows = fake_complete_collection(out, manifest, specs)
    monkeypatch.setattr(ex, "_baseline_contract", lambda *args: baseline)
    folder = out / "rollouts_0"
    if tamper == "raw_return":
        raw = ex.read_json(out / rows[0]["source"])
        raw["detail"]["summary"]["wall_seconds"] = 0.1
        ex.write_json(out / rows[0]["source"], raw)
    elif tamper == "checkpoint":
        (folder / f"{specs['train'][0].name}_states.pt").write_bytes(b"changed")
    else:
        if tamper == "wrong_cost":
            rows[0]["time_budget"] = 1.0
        elif tamper == "wrong_split":
            rows[0]["split"] = "test"
        else:
            rows.pop()
        ex.write_json(folder / "rows.json", rows)
        summary = ex.read_json(folder / "summary.json")
        summary["rows_sha256"] = ex.sha256(folder / "rows.json")
        summary["rows"] = len(rows)
        ex.write_json(folder / "summary.json", summary)
    with pytest.raises(ValueError):
        ex._training_rows(out, manifest, specs, 0)
