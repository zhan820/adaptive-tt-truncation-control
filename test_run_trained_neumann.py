"""Lightweight deployment-guard tests; no solver calls or performance tests."""
from types import SimpleNamespace

import pytest

import run_trained_neumann as cli


def fixture_contract():
    manifest = {"train_dimensions": [4], "fixed_by_dimension": {"4": 7},
                "split_contract": {"excluded_development_seeds": [900],
                                   "paper_seed_excluded_from_training_validation_test": 20260906}}
    splits = {"train": [SimpleNamespace(seed=1)], "validation": [SimpleNamespace(seed=2)],
              "test": [SimpleNamespace(seed=3)], "paper": [SimpleNamespace(seed=20260906)]}
    return manifest, splits


def test_new_rhs_inside_training_scope_is_allowed():
    manifest, splits = fixture_contract()
    cli._new_case_allowed(manifest, splits, 4, 123456)


@pytest.mark.parametrize("seed", [1, 2, 3, 20260906, 900, -1])
def test_reserved_and_invalid_rhs_seeds_are_rejected(seed):
    manifest, splits = fixture_contract()
    with pytest.raises(ValueError):
        cli._new_case_allowed(manifest, splits, 4, seed)


def test_untrained_dimension_cannot_accidentally_transfer():
    manifest, splits = fixture_contract()
    with pytest.raises(ValueError, match="not included in training"):
        cli._new_case_allowed(manifest, splits, 10, 123456)


def test_existing_or_in_run_output_is_never_overwritten(tmp_path):
    run = (tmp_path / "frozen").resolve()
    run.mkdir()
    with pytest.raises(ValueError):
        cli._output_path(run, run / "deployment.json")
    existing = tmp_path / "existing.json"
    existing.write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        cli._output_path(run, existing)
    assert existing.read_text(encoding="utf-8") == "preserve me"
    with pytest.raises(ValueError):
        cli._output_path(run, tmp_path / "not_json.txt")
    assert cli._output_path(run, tmp_path / "new.json") == (tmp_path / "new.json").resolve()
