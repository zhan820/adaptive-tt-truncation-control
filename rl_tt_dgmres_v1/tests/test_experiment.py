from argparse import Namespace
from dataclasses import asdict
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_tt_dgmres_v1 import experiment as exp
from rl_tt_dgmres_v1.runner import ACTIONS, FEATURE_NAMES, EpisodeConfig


def test_problem_order_resume_matches_uninterrupted_sequence():
    uninterrupted_rng, uninterrupted_order, prefix = exp.restore_problem_order(20260930, 48, 192)
    rng, order, consumed = exp.restore_problem_order(20260930, 48, 160)
    remaining = []
    for _ in range(32):
        if not order:
            order = rng.permutation(48).tolist()
        remaining.append(order.pop())
    assert consumed + remaining == prefix
    assert order == uninterrupted_order
    assert rng.bit_generator.state == uninterrupted_rng.bit_generator.state


def test_resume_inherits_unspecified_arguments_and_rejects_changes(tmp_path):
    original = {"seed": 20260930, "episodes": 192, "output": "old", "threads": 1}
    exp.write_json(tmp_path / "manifest.json", {"args": original})
    args = Namespace(resume_from=str(tmp_path), output="new", seed=20260929, episodes=192, threads=1)
    exp.inherit_resume_arguments(args, {"resume_from", "output"})
    assert args.seed == 20260930
    assert args.output == "new"
    args.episodes = 200
    with pytest.raises(ValueError, match="cannot change episodes"):
        exp.inherit_resume_arguments(args, {"episodes"})


@pytest.fixture
def interrupted_run(tmp_path, monkeypatch):
    source = tmp_path / "interrupted"
    source.mkdir()
    config = EpisodeConfig()
    args = Namespace(seed=20260930, train_count=48, validation_count=12,
                     batch_size=32, episodes=192, eval_every=32)
    prior = {"config": asdict(config), "actions": list(ACTIONS),
             "features": list(FEATURE_NAMES), "hidden": 64}
    _, _, sequence = exp.restore_problem_order(args.seed, args.train_count, 178)
    training = []
    for episode, index in enumerate(sequence, 1):
        row = {"episode": episode, "problem": f"train_{index}", "transitions": 1,
               "seconds": 0.25, "loss": {"gradient_steps": episode * 16}}
        training.append(row)
        exp.write_json(source / "train" / f"episode_{episode:04d}.json",
                       {"summary": {"problem": {"name": row["problem"]}},
                        "transitions": [{"episode": episode}]})
    exp.write_json(source / "training_history.json", training)
    exp.write_json(source / "splits.json", {"train": [{"spec": {"name": f"train_{i}"}}
                                                       for i in range(48)]})
    validation = [{"episode": i, "score": [12, -float(i)]} for i in (32, 64, 96, 128, 160)]
    exp.write_json(source / "validation_history.json", validation)
    rows = [{"accepted": True, "wall_seconds": 1.0} for _ in range(12)]
    grid = [{"index": i, "delta": d, "rows": rows, "score": [12, -12.0]}
            for i, d in enumerate(ACTIONS)]
    exp.write_json(source / "baseline_validation.json", grid)
    for name in [f"fixed_{j}" for j in range(len(ACTIONS))] + [
            f"episode_{r['episode']:04d}" for r in validation] + ["episode_0178"]:
        exp.write_json(source / "validation" / name / "detail.json", {"name": name})
    for name in ("dataset.json", "warmup.json", "manifest.json"):
        exp.write_json(source / name, {"origin": name})
    for name in ("controller.pt", "last_checkpoint.pt"):
        (source / name).write_bytes(b"checkpoint-placeholder")
    exp.write_json(source / "source_snapshot" / "source.json", {"verified": True})

    def agent(best=False):
        metadata = {"features": list(FEATURE_NAMES), "config": asdict(config),
                    "episode": 160, "best_epoch": 32, "validation_score": [12, -32.0]}
        return SimpleNamespace(actions=list(ACTIONS), observation_dim=len(FEATURE_NAMES),
                               metadata=metadata, seed=args.seed, batch_size=32, hidden=64,
                               gamma=1.0, learning_rate=0.001, target_update=100,
                               replay=list(range(160)), capacity=50000, gradient_steps=2560)

    loaded = {"last_checkpoint.pt": agent(), "controller.pt": agent(True)}
    monkeypatch.setattr(exp.QAgent, "load", lambda path: loaded[Path(path).name])
    monkeypatch.setattr(exp, "verify_resume_sources", lambda *unused: {"audit": "checked"})
    return source, prior, args, config, loaded


def test_resume_rewinds_to_checkpoint_without_using_later_trajectories(interrupted_run):
    source, prior, args, config, _ = interrupted_run
    state = exp.load_resume_state(source, prior, args, config, {})
    assert state["completed"] == 160
    assert len(state["training"]) == 160
    assert state["excluded_episodes"] == list(range(161, 179))
    assert state["prior_training_solve_seconds"] == 40.0
    assert state["best_epoch"] == 32
    assert max(r["episode"] for r in state["validation"]) == 160
    assert len(state["files"]) == 160


def test_resume_staging_preserves_source_and_excludes_post_checkpoint_artifacts(interrupted_run, tmp_path):
    source, prior, args, config, _ = interrupted_run
    original = (source / "train" / "episode_0161.json").read_bytes()
    state = exp.load_resume_state(source, prior, args, config, {})
    output = tmp_path / "resumed"
    output.mkdir()
    hashes = exp.stage_resume_artifacts(source, output, state)
    assert len(hashes) == 160
    assert len(list((output / "train").glob("*.json"))) == 160
    assert not (output / "train" / "episode_0161.json").exists()
    assert not (output / "validation" / "episode_0178").exists()
    assert (source / "train" / "episode_0161.json").read_bytes() == original
    assert (output / "controller.pt").read_bytes() == (source / "controller.pt").read_bytes()
    assert (output / "origin_source_snapshot" / "source.json").is_file()
    assert (output / "origin_manifest.json").is_file()


def test_resume_rejects_changed_trajectory_problem(interrupted_run):
    source, prior, args, config, _ = interrupted_run
    path = source / "train" / "episode_0001.json"
    detail = exp.read_json(path)
    detail["summary"]["problem"]["name"] = "wrong_problem"
    exp.write_json(path, detail)
    with pytest.raises(ValueError, match="trajectory does not match"):
        exp.load_resume_state(source, prior, args, config, {})


def test_resume_rejects_checkpoint_replay_inconsistency(interrupted_run):
    source, prior, args, config, loaded = interrupted_run
    loaded["last_checkpoint.pt"].replay.pop()
    with pytest.raises(ValueError, match="replay/gradient count"):
        exp.load_resume_state(source, prior, args, config, {})


def test_resume_rejects_best_controller_from_later_validation(interrupted_run):
    source, prior, args, config, loaded = interrupted_run
    loaded["controller.pt"].metadata["best_epoch"] = 178
    with pytest.raises(ValueError, match="best retained validation"):
        exp.load_resume_state(source, prior, args, config, {})


def test_resume_rejects_changed_features(interrupted_run):
    source, prior, args, config, _ = interrupted_run
    prior["features"] = list(reversed(FEATURE_NAMES))
    with pytest.raises(ValueError, match="feature ordering"):
        exp.load_resume_state(source, prior, args, config, {})


def test_resume_rejects_corrupted_original_source_snapshot(tmp_path):
    name = "rl_tt_dgmres_v1/learning.py"
    path = tmp_path / "source_snapshot" / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"changed source")
    hashes = {name: hashlib.sha256(b"original source").hexdigest()}
    with pytest.raises(ValueError, match="snapshot fails SHA-256"):
        exp.verify_resume_sources(tmp_path, {"source_sha256": hashes}, hashes)


def test_resume_rejects_numerical_source_change(tmp_path):
    name = "rl_tt_dgmres_v1/learning.py"
    path = tmp_path / "source_snapshot" / name
    path.parent.mkdir(parents=True)
    path.write_bytes(b"original source")
    previous = {name: hashlib.sha256(path.read_bytes()).hexdigest()}
    current = {name: hashlib.sha256(b"new learner").hexdigest()}
    with pytest.raises(ValueError, match="Numerical/training source changed"):
        exp.verify_resume_sources(tmp_path, {"source_sha256": previous}, current)


def test_resume_accepts_only_explicit_runner_log_patch(tmp_path, monkeypatch):
    project = tmp_path / "project"
    name = "rl_tt_dgmres_v1/runner.py"
    original = ('"cycle": 0, "phase": phase, "requested_eps": float(eps), "effective_eps": 0.0,\n'
                '        "preserved_near_cancellation_calls": sum(r.get("preserved_near_cancellation", False)\n')
    archived = tmp_path / "original" / "source_snapshot" / name
    archived.parent.mkdir(parents=True)
    archived.write_text(original, encoding="utf-8")
    current = project / name
    current.parent.mkdir(parents=True)
    patched = original.replace('"cycle": 0', '"cycle": self.cycle').replace(
        '        "preserved_near_cancellation_calls":',
        '        "near_cancellation_check_seconds": sum(r.get("near_cancellation_check_seconds", 0.0)\n'
        '                                               for r in rounder.logs),\n'
        '        "preserved_near_cancellation_calls":')
    current.write_text(patched, encoding="utf-8")
    monkeypatch.setattr(exp, "__file__", str(project / "rl_tt_dgmres_v1" / "experiment.py"))
    previous = {name: hashlib.sha256(archived.read_bytes()).hexdigest()}
    hashes = {name: hashlib.sha256(current.read_bytes()).hexdigest()}
    changes = exp.verify_resume_sources(tmp_path / "original", {"source_sha256": previous}, hashes)
    assert "numerical path unchanged" in changes[name]["reason"]
    current.write_text(patched + "rounding_tolerance = 1.0\n", encoding="utf-8")
    hashes[name] = hashlib.sha256(current.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="beyond the two approved log-only"):
        exp.verify_resume_sources(tmp_path / "original", {"source_sha256": previous}, hashes)
