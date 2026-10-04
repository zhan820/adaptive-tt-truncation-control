"""Run a frozen Neumann controller on one new RHS, without training.

Example (PowerShell):
  python run_trained_neumann.py --run results/my_frozen_run --seed 123456 --dims 4 --mode rl --output deployment/new_rhs_123456.json

Only dimensions included in the saved training protocol are permitted. The
saved tolerance, time/rank budgets and thread count are reused unchanged.
Output is a separate deployment record, never an addition to the frozen test
set. Existing files and paths inside the frozen run cannot be written.
Exit status 0 means accepted; status 2 means a recorded nonconverged solve.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def _output_path(run_dir, requested):
    output = Path(requested).resolve()
    if output.suffix.lower() != ".json":
        raise ValueError("The independent deployment output must be a new .json file.")
    if output == run_dir or output.is_relative_to(run_dir):
        raise ValueError("Deployment output must remain outside the frozen run directory.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    return output


def _new_case_allowed(manifest, splits, dimensions, seed):
    if dimensions not in manifest["train_dimensions"]:
        raise ValueError("Requested dimension was not included in training; transfer is not enabled.")
    if str(dimensions) not in manifest["fixed_by_dimension"]:
        raise ValueError("The trained dimension has no verified dimension-specific baseline.")
    if seed < 0:
        raise ValueError("The RHS seed must be a nonnegative integer.")
    reserved = {int(spec.seed) for cases in splits.values() for spec in cases}
    reserved.update(manifest["split_contract"]["excluded_development_seeds"])
    reserved.add(manifest["split_contract"]["paper_seed_excluded_from_training_validation_test"])
    if seed in reserved:
        raise ValueError("Choose a new RHS seed: this seed belongs to training, validation, test, paper or development data.")


def run_deployment(args):
    from rl_tt_dgmres_neumann.experiment import (
        _frozen_run, baseline_for, clean, compact, make_learned_policy,
    )
    from rl_tt_dgmres_neumann.problems import ProblemSpec, build_problem, instance_key
    from rl_tt_dgmres_v3.runner import run_episode

    requested_run = Path(args.run).resolve()
    output = _output_path(requested_run, args.output)
    if args.mode not in {"rl", "fixed"}:
        raise ValueError("Mode must be rl or fixed.")
    out, manifest, config, splits = _frozen_run(SimpleNamespace(output=str(requested_run)))
    if manifest["stage"] != "complete":
        raise ValueError("Use a frozen run whose prescribed heldout evaluation is complete.")
    _new_case_allowed(manifest, splits, args.dims, args.seed)
    spec = ProblemSpec(name=f"deploy_n12_d{args.dims}_seed{args.seed}", n=12,
                       dims=args.dims, seed=args.seed, cp_rank=3, max_mode=None,
                       null_ratio=0.1, spectral_decay=2.0, preparation_tolerance=1e-13)
    baseline = baseline_for(manifest, spec)
    decisions = []
    if args.mode == "rl":
        policy = make_learned_policy(out, manifest["selected"], baseline, decisions)
    else:
        policy = lambda observation: baseline
    problem = build_problem(spec)
    detail = run_episode(problem, config, policy, evaluate_reference=True,
                         compute_policy_features=args.mode == "rl")
    record = {
        "record_type": "independent_frozen_controller_deployment",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "included_in_original_heldout": False,
        "used_for_training_or_model_selection": False,
        "paired_benchmark": False,
        "warmup_performed": False,
        "scope": "new RHS within the saved Neumann operator/dimension scope; no transfer claim",
        "mode": args.mode,
        "fixed_baseline": "dimension-specific action selected on the original validation set",
        "baseline_index": baseline,
        "problem": asdict(spec), "instance_key": instance_key(spec),
        "construction_metadata": problem.construction_metadata(),
        "provenance": {
            "frozen_run": str(out), "prepared_splits_sha256": manifest["splits_sha256"],
            "frozen_selection_sha256": manifest["frozen_selection_sha256"],
            "frozen_model": str(out / manifest["selected"]["model"]),
            "frozen_model_sha256": manifest["selected_model_sha256"],
            "model_used": args.mode == "rl",
            "selected_policy": manifest["selected"],
            "source_sha256": manifest["source_sha256"],
            "deployment_cli_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "config": asdict(config), "threads": manifest["threads"],
            "trained_dimensions": manifest["train_dimensions"],
            "timing": "common solver wall time includes online features, inference, safeguards and final audit; public problem construction, model loading and reference forward-error evaluation are outside solver timing",
        },
        "decisions": decisions,
        "result": compact(detail),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(clean(record), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return record, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Completed, frozen experiment directory")
    parser.add_argument("--seed", type=int, required=True, help="A new RHS seed absent from all original splits and pilots")
    parser.add_argument("--dims", type=int, default=4)
    parser.add_argument("--mode", choices=["rl", "fixed"], default="rl")
    parser.add_argument("--output", required=True, help="New JSON outside the frozen run directory")
    args = parser.parse_args(argv)
    try:
        record, output = run_deployment(args)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Deployment stopped: {error}\n")
    summary = json.loads(output.read_text(encoding="utf-8"))["result"]["summary"]
    print(json.dumps({"output": str(output), "accepted": summary["accepted"],
                      "status": summary["status"], "seconds": summary["wall_seconds"],
                      "eta": summary["eta"]}, ensure_ascii=False))
    return 0 if summary["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
