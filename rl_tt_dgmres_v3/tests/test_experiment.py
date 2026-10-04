from pathlib import Path

import pytest

from rl_tt_dgmres_v3.experiment import score, source_check, ensure_protocol


def test_selection_prefers_success_over_fast_failure():
    slow = [{"accepted": True, "wall_seconds": 10.0}]
    fast_failure = [{"accepted": False, "wall_seconds": 0.001}]
    assert tuple(score(slow, 20)) > tuple(score(fast_failure, 20))


def test_selection_same_success_prefers_smaller_total_time():
    fast = [{"accepted": True, "wall_seconds": 0.5}]
    slow = [{"accepted": True, "wall_seconds": 1.0}]
    assert tuple(score(fast, 20)) > tuple(score(slow, 20))


def test_changed_source_is_not_silently_resumed():
    with pytest.raises(RuntimeError, match="Source changed"):
        source_check({"source_sha256": {}})


def test_resumed_stage_rejects_changed_model_contract(tmp_path):
    path = tmp_path / "protocol.json"
    ensure_protocol(path, {"model_sha": "one", "actions": [1, 2]})
    ensure_protocol(path, {"model_sha": "one", "actions": [1, 2]})
    with pytest.raises(RuntimeError, match="protocol changed"):
        ensure_protocol(path, {"model_sha": "two", "actions": [1, 2]})
