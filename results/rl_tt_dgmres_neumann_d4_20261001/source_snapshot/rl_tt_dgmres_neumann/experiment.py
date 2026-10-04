from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from rl_tt_dgmres_v1.experiment import clean, write_json
from .problems import (BUILDER_SOURCE, ProblemSpec, build_problem, instance_key,
                       operator_group_key, spec_from_dict)
from rl_tt_dgmres_v2.experiment import aggregate, paired_speedups


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def emit(event, **values):
    print(json.dumps(clean(dict(event=event, **values)), ensure_ascii=False), flush=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_protocol(path, protocol):
    """A resumed stage cannot reuse results under a changed data/model contract."""
    path = Path(path)
    normalized = clean(protocol)
    if path.exists():
        if read_json(path) != normalized:
            raise RuntimeError(f"Stage protocol changed: {path}. Use a new run directory.")
    else:
        write_json(path, normalized)


def hashes():
    root = Path(__file__).resolve().parent.parent
    paths = []
    for folder in ("rl_tt_dgmres_neumann", "rl_tt_dgmres_v1", "rl_tt_dgmres_v2", "rl_tt_dgmres_v3", "adaptive_tt_dgmres"):
        paths.extend((root / folder).glob("*.py"))
    paths.extend((root / "adaptive_tt_dgmres" / "backend").glob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def source_check(manifest):
    if hashes() != manifest["source_sha256"]:
        raise RuntimeError("Source changed after preparation; start a new development run, not a silent resume.")


def score(rows, budget):
    return [sum(bool(r["accepted"]) for r in rows),
            -sum(r["wall_seconds"] if r["accepted"] else 5 * budget for r in rows)]


PAPER_SEED = 20260906
PAPER_DIMENSIONS = (4, 5, 6, 10)
DEVELOPMENT_SEEDS = (2026100191,) + tuple(2026100191 + 1009 * d for d in (4, 5, 6))
PAPER_SOURCES = (
    Path("C:/Users/16081/Desktop/论文知识库/pytorch_version/backend_cells.json"),
    Path("C:/Users/16081/Desktop/论文知识库/output/上财模板_Nature方法修订_20260929/chapters/ch05_setup.tex"),
)
EVALUATION_MODES = ("fixed", "fixed_by_dimension", "rl", "rl_static")


def _paper_dimensions(values):
    dimensions = tuple(int(d) for d in values)
    if (not dimensions or len(set(dimensions)) != len(dimensions) or
            not set(dimensions).issubset(PAPER_DIMENSIONS)):
        raise ValueError("Paper dimensions must be a nonempty distinct subset of 4,5,6,10.")
    return dimensions


def make_splits(seed, train_count, validation_count, test_count, dimensions,
                paper_dimensions=PAPER_DIMENSIONS):
    dimensions = tuple(int(d) for d in dimensions)
    paper_dimensions = _paper_dimensions(paper_dimensions)
    if not dimensions or len(set(dimensions)) != len(dimensions) or min(dimensions) < 2:
        raise ValueError("Dimensions must be distinct integers at least two.")
    if min(train_count, validation_count, test_count) < len(dimensions):
        raise ValueError("Every RHS split must contain every selected dimension.")
    rng = random.Random(seed)
    used_seeds = {PAPER_SEED, *DEVELOPMENT_SEEDS}
    splits = {}
    for role, count in (("train", train_count), ("validation", validation_count), ("test", test_count)):
        values = []
        for i in range(count):
            rhs_seed = rng.randrange(1, 2**31 - 1)
            while rhs_seed in used_seeds:
                rhs_seed = rng.randrange(1, 2**31 - 1)
            used_seeds.add(rhs_seed)
            d = dimensions[i % len(dimensions)]
            values.append(ProblemSpec(name=f"{role}_d{d}_{i:04d}", n=12, dims=d, seed=rhs_seed,
                                      cp_rank=3, null_ratio=0.1, spectral_decay=2.0,
                                      preparation_tolerance=1e-13))
        rng.shuffle(values)
        splits[role] = values
    splits["paper"] = [ProblemSpec(name=f"paper_d{d}_seed{PAPER_SEED}", n=12, dims=d,
                                  seed=PAPER_SEED, cp_rank=3, null_ratio=0.1,
                                  spectral_decay=2.0, preparation_tolerance=1e-13)
                       for d in paper_dimensions]
    keys = [instance_key(s) for values in splits.values() for s in values]
    if len(set(keys)) != len(keys):
        raise ValueError("Instance overlap between splits.")
    return splits


def _contract(manifest):
    names = ("seed", "config", "threads", "features", "actions", "max_states_per_problem",
             "repeats", "train_dimensions", "paper_protocol", "splits_sha256",
             "source_sha256", "builder_source", "paper_provenance", "split_contract")
    return {name: manifest[name] for name in names}


def _validate_splits(specs, dimensions, paper_dimensions=PAPER_DIMENSIONS,
                     excluded_development_seeds=DEVELOPMENT_SEEDS):
    paper_dimensions = _paper_dimensions(paper_dimensions)
    if set(excluded_development_seeds) != set(DEVELOPMENT_SEEDS):
        raise ValueError("The declared development-seed exclusion protocol changed.")
    if set(specs) != {"train", "validation", "test", "paper"}:
        raise ValueError("Unexpected split roles.")
    seen_seeds, seen_instances = set(), set()
    for role in ("train", "validation", "test"):
        if {s.dims for s in specs[role]} != set(dimensions):
            raise ValueError("Every heldout/training RHS split must cover the configured dimensions.")
        for spec in specs[role]:
            key = instance_key(spec)
            if (spec.seed == PAPER_SEED or spec.seed in excluded_development_seeds or
                    spec.seed in seen_seeds or key in seen_instances):
                raise ValueError("RHS seed/instance overlap or paper/development-case leakage.")
            seen_seeds.add(spec.seed)
            seen_instances.add(key)
            if (spec.n != 12 or spec.cp_rank != 3 or spec.null_ratio != 0.1 or
                    spec.spectral_decay != 2.0 or spec.preparation_tolerance != 1e-13 or
                    spec.max_mode not in (None, 11)):
                raise ValueError("The prepared case differs from the declared paper problem family.")
    if {(s.dims, s.seed) for s in specs["paper"]} != {(d, PAPER_SEED) for d in paper_dimensions}:
        raise ValueError("Paper showcase cases changed.")
    paper_keys = [instance_key(s) for s in specs["paper"]]
    if (len(paper_keys) != len(paper_dimensions) or len(set(paper_keys)) != len(paper_dimensions) or
            seen_instances.intersection(paper_keys)):
        raise ValueError("Paper showcase overlaps or duplicates experimental instances.")
    for spec in specs["paper"]:
        if (spec.n != 12 or spec.cp_rank != 3 or spec.null_ratio != 0.1 or
                spec.spectral_decay != 2.0 or spec.preparation_tolerance != 1e-13 or
                spec.max_mode not in (None, 11)):
            raise ValueError("Historical paper construction parameters changed.")


def prepare(args):
    from rl_tt_dgmres_v3.runner import ACTIONS, FEATURE_NAMES, EpisodeConfig
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("Preparation requires a new directory.")
    dimensions = tuple(int(d.strip()) for d in args.train_dims.split(",") if d.strip())
    paper_dimensions = _paper_dimensions(d.strip() for d in args.paper_dims.split(",") if d.strip())
    specs = make_splits(args.seed, args.train_count, args.validation_count, args.test_count,
                       dimensions, paper_dimensions)
    _validate_splits(specs, dimensions, paper_dimensions)
    config = EpisodeConfig(seconds_budget=args.seconds_budget, max_iterations=args.max_iterations,
                           tolerance=args.tolerance, max_tt_rank=args.max_tt_rank,
                           recovery_max_tt_rank=args.recovery_max_rank,
                           diagnostic_max_rank=args.diagnostic_max_rank)
    config.validate()
    if args.threads < 1 or args.paper_seconds <= 0:
        raise ValueError("Threads and paper budget must be positive.")
    for path in PAPER_SOURCES:
        if not path.is_file():
            raise FileNotFoundError(f"Required paper provenance is unavailable: {path}")
    if sha256(PAPER_SOURCES[0]) != BUILDER_SOURCE["sha256"]:
        raise RuntimeError("Paper builder differs from the verified construction source.")
    out.mkdir(parents=True)
    provenance = []
    for path in PAPER_SOURCES:
        target = out / "paper_provenance" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)
        provenance.append({"original_path": str(path), "snapshot": str(target.relative_to(out)),
                           "sha256": sha256(target)})
    write_json(out / "splits.json", {k: [asdict(s) for s in v] for k, v in specs.items()})
    manifest = {
        "version": "neumann_1", "stage": "prepared", "seed": args.seed,
        "config": asdict(config), "source_sha256": hashes(), "threads": args.threads,
        "features": list(FEATURE_NAMES), "actions": [asdict(a) for a in ACTIONS],
        "train_dimensions": list(dimensions), "builder_source": BUILDER_SOURCE,
        "paper_provenance": provenance,
        "splits_sha256": sha256(out / "splits.json"),
        "split_counts": {k: len(v) for k, v in specs.items()},
        "max_states_per_problem": args.max_states, "repeats": args.repeats,
        "split_contract": {
            "unit": "complete RHS instance including all states, action branches and repetitions",
            "instance_disjoint": True, "operator_disjoint": False,
            "generalization": "new independent RHS on the same finite isotropic Neumann operators",
            "paper_seed_excluded_from_training_validation_test": PAPER_SEED,
            "excluded_development_seeds": list(DEVELOPMENT_SEEDS),
            "paper_cases": "known historical showcase, not an unseen test set",
            "rhs_family": "paper CP3, DCT normal/(1+q)^2, global mean removal and unit normalization",
        },
        "paper_protocol": {
            "dimensions": list(paper_dimensions),
            "seconds_budget": args.paper_seconds,
            "tolerance_by_dimension": {str(d): 1e-5 if d == 10 else 1e-7 for d in paper_dimensions},
            "modes": ["fixed_by_dimension", "rl"], "repeats": args.repeats,
            "unseen_dimension_fixed_rule": "global validation-selected fixed action",
            "purpose": "new shared V3 solver reproduction, no preconditioner; not original figure timings",
        },
        "validation_selection": "maximize success count, then minimize failure-penalized total time",
        "baseline_protocol": "nine-action validation grid; dimension-specific fixed is primary, global fixed secondary",
        "acceptance_protocol": {
            "same_backend": True, "same_numerical_safeguards": True,
            "same_precision_and_budget": True, "training_reference_labels": False,
            "all_online_feature_and_inference_cost_included": True,
            "heldout_order": "serial, randomly interleaved within instance and repeat",
            "primary_baseline": "fixed_by_dimension",
            "primary_claim": "no lower observed success count AND paired geometric time speedup > 1",
            "strong_claim": "primary claim AND 95% RHS-instance-bootstrap lower bound > 1",
            "heldout_not_used_for_selection": True,
        },
        "evaluation_started": False, "paper_started": False,
    }
    root = Path(__file__).resolve().parent.parent
    for name in manifest["source_sha256"]:
        target = out / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    write_json(out / "manifest.json", manifest)
    write_json(out / "prepared_contract.json", _contract(manifest))
    emit("prepared", output=str(out), split_counts=manifest["split_counts"])


def load_run(args, training=False):
    from rl_tt_dgmres_v3.runner import EpisodeConfig
    out = Path(args.output).resolve()
    manifest = read_json(out / "manifest.json")
    source_check(manifest)
    if read_json(out / "prepared_contract.json") != _contract(manifest):
        raise RuntimeError("Prepared protocol changed; use a new run directory.")
    for item in manifest["paper_provenance"]:
        if sha256(out / item["snapshot"]) != item["sha256"]:
            raise RuntimeError("Paper provenance snapshot changed.")
    if sha256(out / "splits.json") != manifest["splits_sha256"]:
        raise RuntimeError("Prepared split definitions changed.")
    if training and (manifest["evaluation_started"] or manifest.get("paper_started") or
                     manifest["stage"] in {"frozen", "evaluating", "complete"}):
        raise RuntimeError("Frozen/evaluated runs prohibit further training or selection.")
    torch.set_num_threads(manifest["threads"])
    specs = {k: [spec_from_dict(s) for s in v] for k, v in read_json(out / "splits.json").items()}
    _validate_splits(specs, manifest["train_dimensions"], manifest["paper_protocol"]["dimensions"],
                     manifest["split_contract"]["excluded_development_seeds"])
    return out, manifest, EpisodeConfig(**manifest["config"]), specs


def baseline_for(manifest, spec):
    return int(manifest["fixed_by_dimension"].get(str(spec.dims), manifest["best_fixed_index"]))


def warm_up(spec, config):
    from rl_tt_dgmres_v3.runner import run_episode
    run_episode(build_problem(spec), config, lambda observation: 3, evaluate_reference=False)


def compact(detail):
    return {k: v for k, v in detail.items() if k not in {"solution", "round_calls", "solver_history"}}


def fixed_grid(args):
    from rl_tt_dgmres_v3.runner import ACTIONS, run_episode
    out, manifest, config, specs = load_run(args, training=True)
    ensure_protocol(out / "fixed_protocol.json", {
        "contract": _contract(manifest), "validation_specs": [asdict(s) for s in specs["validation"]],
        "actions": list(range(len(ACTIONS))), "compute_policy_features": False,
    })
    warm_up(specs["train"][0], config)
    jobs = [(i, j) for i in range(len(specs["validation"])) for j in range(len(ACTIONS))]
    random.Random(manifest["seed"] + 8).shuffle(jobs)
    started = time.perf_counter()
    for done, (i, j) in enumerate(jobs, 1):
        spec = specs["validation"][i]
        path = out / "fixed_validation" / f"{spec.name}_a{j}.json"
        if not path.exists():
            detail = run_episode(build_problem(spec), config, lambda obs, a=j: a,
                                 evaluate_reference=False, compute_policy_features=False)
            write_json(path, compact(detail))
        summary = read_json(path)["summary"]
        emit("fixed_grid", done=done, total=len(jobs), action=j, problem=spec.name,
             success=summary["accepted"], seconds=summary["wall_seconds"])
    grid = []
    for j in range(len(ACTIONS)):
        rows = [read_json(out / "fixed_validation" / f"{s.name}_a{j}.json")["summary"]
                for s in specs["validation"]]
        dimension_scores = {str(d): score([r for r in rows if r["problem"]["dims"] == d],
                                          config.seconds_budget)
                            for d in manifest["train_dimensions"]}
        grid.append(dict(action=j, score=score(rows, config.seconds_budget),
                         dimension_scores=dimension_scores, rows=rows))
    best = max(grid, key=lambda row: tuple(row["score"]))
    per_dimension = {str(d): max(grid, key=lambda row: tuple(row["dimension_scores"][str(d)]))["action"]
                     for d in manifest["train_dimensions"]}
    write_json(out / "baseline_validation.json", grid)
    manifest.update(best_fixed_index=best["action"], fixed_by_dimension=per_dimension,
                    baseline_validation_sha256=sha256(out / "baseline_validation.json"),
                    stage="fixed_selected")
    manifest["fixed_grid_last_execution_seconds"] = time.perf_counter() - started
    write_json(out / "manifest.json", manifest)
    emit("fixed_selected", global_action=best["action"], per_dimension=per_dimension)


def _baseline_contract(out, manifest):
    if sha256(out / "baseline_validation.json") != manifest["baseline_validation_sha256"]:
        raise RuntimeError("Validation-selected fixed baseline changed.")
    grid = read_json(out / "baseline_validation.json")
    global_best = max(grid, key=lambda row: tuple(row["score"]))["action"]
    by_dimension = {str(d): max(grid, key=lambda row: tuple(row["dimension_scores"][str(d)]))["action"]
                    for d in manifest["train_dimensions"]}
    if global_best != manifest["best_fixed_index"] or by_dimension != manifest["fixed_by_dimension"]:
        raise RuntimeError("Baseline selection does not match the saved validation grid.")
    return {"global": global_best, "by_dimension": by_dimension,
            "validation_sha256": manifest["baseline_validation_sha256"]}


def load_controller(model_path, gate, baseline, *, expected_sha256=None, decisions=None):
    """Load a frozen pretrained controller without fitting or opening any test data."""
    from rl_tt_dgmres_v3.learning import RolloutValueLearner, GateConfig
    model_path = Path(model_path)
    if expected_sha256 is not None and sha256(model_path) != expected_sha256:
        raise RuntimeError("Pretrained model hash differs from its frozen selection.")
    learner = RolloutValueLearner.load(model_path)
    gate = GateConfig(**gate) if isinstance(gate, dict) else gate
    def policy(observation):
        action, decision = learner.policy(observation, baseline=int(baseline), gate=gate)
        if decisions is not None:
            decisions.append(decision)
        return action
    return policy


def make_learned_policy(out, selection, baseline, decisions=None):
    return load_controller(out / selection["model"], selection["gate"], baseline,
                           expected_sha256=selection["model_sha256"], decisions=decisions)


def collect(args):
    from rl_tt_dgmres_v3.runner import ACTIONS, run_episode
    out, manifest, config, specs = load_run(args, training=True)
    baseline_contract = _baseline_contract(out, manifest)
    iteration = args.iteration
    continuation_id = "baseline" if iteration == 0 else f"policy_iteration_{iteration}"
    if iteration == 0:
        selected = None
    else:
        selected = read_json(out / f"selection_{iteration - 1}.json")
        _verify_selection(out, manifest, selected, specs)
    folder = out / f"rollouts_{iteration}"
    folder.mkdir(exist_ok=True)
    chosen_specs = specs["train"][:args.limit] if args.limit else specs["train"]
    ensure_protocol(folder / "collection_protocol.json", {
        "continuation_policy": continuation_id, "selection": selected,
        "baseline": baseline_contract, "source_sha256": manifest["source_sha256"],
        "config": asdict(config), "specs": [asdict(s) for s in chosen_specs],
        "max_states": manifest["max_states_per_problem"],
        "candidate_actions": list(range(len(ACTIONS))),
        "origin": "same real outer-cycle checkpoint, one candidate action then stated continuation",
        "checkpoint_overhead": "offline cloning excluded; normal per-cycle initialization included",
    })
    summary_path = folder / "summary.json"
    if summary_path.exists() and read_json(summary_path).get("complete"):
        rows = _training_rows(out, manifest, specs, iteration)
        emit("collection_already_complete", iteration=iteration, rows=len(rows))
        return
    warm_up(specs["train"][0], config)
    started = time.perf_counter()
    rows = []
    for problem_index, spec in enumerate(chosen_specs):
        problem = build_problem(spec)
        baseline = baseline_for(manifest, spec)
        continuation = ((lambda obs, a=baseline: a) if iteration == 0 else
                        make_learned_policy(out, selected, baseline))
        state_file = folder / (spec.name + "_states.pt")
        state_receipt = folder / (spec.name + "_states_receipt.json")
        trajectory_file = folder / (spec.name + "_collection.json")
        if state_file.exists():
            receipt = read_json(state_receipt)
            if (receipt["checkpoint_sha256"] != sha256(state_file) or
                    receipt["trajectory_sha256"] != sha256(trajectory_file)):
                raise RuntimeError("Cached real trajectory/checkpoint changed.")
            states = torch.load(state_file, map_location="cpu", weights_only=False)
        else:
            states = []
            def observe(checkpoint):
                if (len(states) < manifest["max_states_per_problem"]
                        and not checkpoint.controller_state["force_safe"]):
                    states.append(checkpoint)
            trajectory = run_episode(problem, config, continuation, evaluate_reference=False,
                                     checkpoint_observer=observe)
            torch.save(states, state_file)
            write_json(trajectory_file, compact(trajectory))
            write_json(state_receipt, {"checkpoint_sha256": sha256(state_file),
                                      "trajectory_sha256": sha256(trajectory_file)})
        for state_index, checkpoint in enumerate(states):
            actions = list(range(len(ACTIONS)))
            random.Random(manifest["seed"] + 1009 * problem_index + 97 * state_index + iteration).shuffle(actions)
            state_id = f"{continuation_id}/{spec.name}/{state_index}"
            for action in actions:
                path = folder / f"{spec.name}_s{state_index}_a{action}.json"
                if path.exists():
                    saved = read_json(path)
                else:
                    calls = [0]
                    def branched(observation):
                        calls[0] += 1
                        return action if calls[0] == 1 else continuation(observation)
                    detail = run_episode(problem, config, branched, evaluate_reference=False,
                                         initial_checkpoint=checkpoint)
                    summary = detail["summary"]
                    features = list(checkpoint.features)
                    row = {
                        "features": features, "action": action,
                        "remaining_wall": summary["wall_seconds"], "time_budget": config.seconds_budget,
                        "success": bool(summary["accepted"]), "data_origin": "real_rollout",
                        "continuation_policy": continuation_id, "problem_id": spec.name,
                        "state_id": state_id, "operator_group_key": problem.operator_group_key,
                        "source": str(path.relative_to(out)), "state_index": state_index,
                        "executed_first_action": summary["executed_action_indices"][0]
                        if summary["executed_action_indices"] else None,
                        "checkpoint_elapsed_seconds": checkpoint.elapsed_seconds,
                        "split": "train", "baseline_index": baseline,
                        "instance_key": instance_key(spec),
                        "cost_semantics": "remaining_wall/original_episode_budget + failure_penalty",
                    }
                    saved = {"row": row, "detail": compact(detail)}
                    write_json(path, saved)
                rows.append(saved["row"])
            emit("rollout_state", iteration=iteration, problem=spec.name, state=state_index,
                 problems_done=problem_index + 1, problems_total=len(chosen_specs), rows=len(rows),
                 successes=sum(r["success"] for r in rows[-len(ACTIONS):]))
        write_json(folder / "rows.json", rows)
    write_json(folder / "rows.json", rows)
    write_json(folder / "summary.json", {
        "continuation_policy": continuation_id, "rows": len(rows),
        "states": len({r["state_id"] for r in rows}), "problems": len(chosen_specs),
        "last_execution_seconds": time.perf_counter() - started,
        "successes": sum(r["success"] for r in rows),
        "complete": len(chosen_specs) == len(specs["train"]),
        "rows_sha256": sha256(folder / "rows.json"),
        "raw_files_sha256": {r["source"]: sha256(out / r["source"]) for r in rows},
        "state_receipts_sha256": {str((folder / (s.name + "_states_receipt.json")).relative_to(out)):
                                  sha256(folder / (s.name + "_states_receipt.json")) for s in chosen_specs},
        "baseline_semantics": "dimension-conditioned validation-selected baseline",
    })
    emit("collection_complete", iteration=iteration, rows=len(rows), seconds=time.perf_counter()-started)



def _training_rows(out, manifest, specs, iteration):
    folder = out / f"rollouts_{iteration}"
    path = folder / "rows.json"
    summary = read_json(folder / "summary.json")
    protocol = read_json(folder / "collection_protocol.json")
    target = "baseline" if iteration == 0 else f"policy_iteration_{iteration}"
    if (not summary.get("complete") or summary["problems"] != len(specs["train"]) or
            summary.get("rows_sha256") != sha256(path) or
            protocol["specs"] != [asdict(s) for s in specs["train"]] or
            protocol["source_sha256"] != manifest["source_sha256"] or
            protocol["config"] != manifest["config"] or
            protocol["baseline"] != _baseline_contract(out, manifest)):
        raise ValueError("Fit requires a complete collection under the frozen training protocol.")
    rows = read_json(path)
    if summary["rows"] != len(rows):
        raise ValueError("Collection row count differs from its receipt.")
    if set(summary["raw_files_sha256"]) != {r["source"] for r in rows}:
        raise ValueError("Raw rollout provenance is incomplete.")
    for relative, digest in summary["raw_files_sha256"].items():
        if sha256(out / relative) != digest:
            raise ValueError("A raw environment return changed after collection.")
    for spec in specs["train"]:
        relative = str((folder / (spec.name + "_states_receipt.json")).relative_to(out))
        if sha256(out / relative) != summary["state_receipts_sha256"].get(relative):
            raise ValueError("A source trajectory receipt changed.")
        receipt = read_json(out / relative)
        if (sha256(folder / (spec.name + "_states.pt")) != receipt["checkpoint_sha256"] or
                sha256(folder / (spec.name + "_collection.json")) != receipt["trajectory_sha256"]):
            raise ValueError("A source real trajectory/checkpoint changed.")
    by_name = {s.name: s for s in specs["train"]}
    groups = {}
    for row in rows:
        spec = by_name.get(row.get("problem_id"))
        if spec is None or row.get("split") != "train" or row.get("data_origin") != "real_rollout":
            raise ValueError("Only genuine training-instance rollouts may be fitted.")
        if (row["continuation_policy"] != target or row["instance_key"] != instance_key(spec) or
                row["baseline_index"] != baseline_for(manifest, spec) or
                row["time_budget"] != manifest["config"]["seconds_budget"]):
            raise ValueError("Rollout target, baseline, instance, or cost normalization changed.")
        raw_path = (out / row["source"]).resolve()
        if not raw_path.is_relative_to(folder.resolve()):
            raise ValueError("Rollout raw source escapes its collection.")
        raw = read_json(raw_path)
        detail = raw["detail"]["summary"]
        if (raw["row"] != row or row["remaining_wall"] != detail["wall_seconds"] or
                row["success"] != bool(detail["accepted"]) or detail["problem"] != asdict(spec)):
            raise ValueError("Training row no longer matches its real environment return.")
        groups.setdefault(row["state_id"], []).append(row["action"])
    if summary["states"] != len(groups) or any(sorted(a) != list(range(9)) for a in groups.values()):
        raise ValueError("Each sampled checkpoint requires exactly all nine real action branches.")
    return rows


def fit(args):
    from rl_tt_dgmres_v3.learning import LearningConfig, RolloutValueLearner
    from rl_tt_dgmres_v3.runner import FEATURE_NAMES
    out, manifest, config, specs = load_run(args, training=True)
    path = out / f"rollouts_{args.iteration}" / "rows.json"
    rows = _training_rows(out, manifest, specs, args.iteration)
    target = "baseline" if args.iteration == 0 else f"policy_iteration_{args.iteration}"
    learning_config = LearningConfig(seed=manifest["seed"] + args.iteration, epochs=args.epochs,
                                     failure_penalty=config.failure_penalty)
    fit_protocol = {"rows_sha256": sha256(path), "config": asdict(learning_config),
                    "collection_summary_sha256": sha256(path.parent / "summary.json"),
                    "target_policy": target, "source_sha256": manifest["source_sha256"],
                    "baseline": _baseline_contract(out, manifest),
                    "generalization": manifest["split_contract"]["generalization"]}
    ensure_protocol(out / f"fit_protocol_{args.iteration}.json", fit_protocol)
    model_path = out / f"controller_iteration_{args.iteration}.pt"
    fit_path = out / f"fit_{args.iteration}.json"
    if model_path.exists():
        if not fit_path.exists() or read_json(fit_path)["model_sha256"] != sha256(model_path):
            raise RuntimeError("Incomplete or changed fitted artifact; preserve it and start a new run.")
        emit("fit_already_complete", iteration=args.iteration)
        return
    learner = RolloutValueLearner(input_dim=len(FEATURE_NAMES), config=learning_config)
    started = time.perf_counter()
    result = learner.fit(rows, target_policy=target, source_data=[str(path)], warm_start=False)
    learner.save(model_path, metadata={"experiment_protocol": fit_protocol})
    write_json(out / f"fit_{args.iteration}.json", {
        "result": result, "seconds": time.perf_counter()-started,
        "target_policy": target, "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    })
    emit("fit_complete", iteration=args.iteration, result=result)


def validate(args):
    from rl_tt_dgmres_v3.runner import run_episode
    from rl_tt_dgmres_v3.learning import GateConfig
    out, manifest, config, specs = load_run(args, training=True)
    _baseline_contract(out, manifest)
    model = f"controller_iteration_{args.iteration}.pt"
    gates = [
        GateConfig(min_gain=0.0, uncertainty_scale=0.0, max_failure_increase=0.15,
                   max_failure_probability=0.50, min_action_samples=3),
        GateConfig(min_gain=0.002, uncertainty_scale=0.5, max_failure_increase=0.05,
                   max_failure_probability=0.25, min_action_samples=3),
        GateConfig(min_gain=0.005, uncertainty_scale=1.0, max_failure_increase=0.02,
                   max_failure_probability=0.15, min_action_samples=3),
    ]
    selections = [dict(model=model, gate=asdict(g), baselines=_baseline_contract(out, manifest),
                       iteration=args.iteration, gate_index=i, model_sha256=sha256(out / model)) for i, g in enumerate(gates)]
    ensure_protocol(out / f"validation_protocol_{args.iteration}.json", {
        "selections": selections, "specs": [asdict(s) for s in specs["validation"]],
        "config": asdict(config), "source_sha256": manifest["source_sha256"],
    })
    warm_up(specs["train"][0], config)
    jobs = [(i, j) for i in range(len(specs["validation"])) for j in range(len(gates))]
    random.Random(manifest["seed"] + 200 + args.iteration).shuffle(jobs)
    for done, (i, j) in enumerate(jobs, 1):
        spec = specs["validation"][i]
        path = out / f"validation_{args.iteration}" / f"{spec.name}_g{j}.json"
        if not path.exists():
            detail = run_episode(build_problem(spec), config,
                                 make_learned_policy(out, selections[j], baseline_for(manifest, spec)),
                                 evaluate_reference=False)
            write_json(path, compact(detail))
        summary = read_json(path)["summary"]
        emit("validation", done=done, total=len(jobs), iteration=args.iteration,
             gate=j, success=summary["accepted"], seconds=summary["wall_seconds"])
    for j, selection in enumerate(selections):
        rows = [read_json(out / f"validation_{args.iteration}" / f"{s.name}_g{j}.json")["summary"]
                for s in specs["validation"]]
        selection.update(score=score(rows, config.seconds_budget), rows=rows,
                         validation_file_sha256={str((out / f"validation_{args.iteration}" / f"{s.name}_g{j}.json").relative_to(out)):
                         sha256(out / f"validation_{args.iteration}" / f"{s.name}_g{j}.json") for s in specs["validation"]})
    best = max(selections, key=lambda row: tuple(row["score"]))
    write_json(out / f"validation_{args.iteration}.json", selections)
    write_json(out / f"selection_{args.iteration}.json", {k: v for k, v in best.items() if k != "rows"})
    emit("selection", **{k: v for k, v in best.items() if k != "rows"})


def _verify_selection(out, manifest, selection, specs):
    if selection["baselines"] != _baseline_contract(out, manifest):
        raise RuntimeError("Learned selection uses a different fixed-baseline contract.")
    model = out / selection["model"]
    if sha256(model) != selection["model_sha256"]:
        raise RuntimeError("Selected model changed.")
    iteration, gate_index = selection["iteration"], selection["gate_index"]
    fit_protocol = read_json(out / f"fit_protocol_{iteration}.json")
    fit_record = read_json(out / f"fit_{iteration}.json")
    if (fit_record["model_sha256"] != sha256(model) or
            fit_protocol["source_sha256"] != manifest["source_sha256"] or
            fit_protocol["collection_summary_sha256"] != sha256(out / f"rollouts_{iteration}" / "summary.json") or
            fit_protocol["rows_sha256"] != sha256(out / f"rollouts_{iteration}" / "rows.json")):
        raise RuntimeError("Model no longer matches its training receipt.")
    _training_rows(out, manifest, specs, iteration)
    protocol = read_json(out / f"validation_protocol_{iteration}.json")
    if (protocol["config"] != manifest["config"] or
            protocol["source_sha256"] != manifest["source_sha256"] or
            protocol["specs"] != [asdict(s) for s in specs["validation"]]):
        raise RuntimeError("Validation receipt no longer matches the prepared experiment.")
    candidate = protocol["selections"][gate_index]
    for key in ("model", "gate", "baselines", "iteration", "gate_index", "model_sha256"):
        if candidate[key] != selection[key]:
            raise RuntimeError("Selection differs from its evaluated policy.")
    expected_files = [str((out / f"validation_{iteration}" / f"{s.name}_g{gate_index}.json").relative_to(out))
                      for s in specs["validation"]]
    if set(selection["validation_file_sha256"]) != set(expected_files):
        raise RuntimeError("Selection is missing validation instances.")
    rows = []
    for relative in expected_files:
        if sha256(out / relative) != selection["validation_file_sha256"][relative]:
            raise RuntimeError("Validation raw result changed.")
        rows.append(read_json(out / relative)["summary"])
    if score(rows, manifest["config"]["seconds_budget"]) != selection["score"]:
        raise RuntimeError("Selection score does not match the real validation runs.")


def freeze(args):
    out, manifest, config, specs = load_run(args, training=True)
    selections = [read_json(p) for p in sorted(out.glob("selection_*.json"))]
    if not selections:
        raise ValueError("Validate at least one learned policy before freezing.")
    for candidate in selections:
        _verify_selection(out, manifest, candidate, specs)
    selected = max(selections, key=lambda row: tuple(row["score"]))
    receipt = {"selection": selected, "baseline": _baseline_contract(out, manifest),
               "selection_files_sha256": {p.name: sha256(p) for p in sorted(out.glob("selection_*.json"))},
               "contract": _contract(manifest), "evaluation_modes": list(EVALUATION_MODES)}
    write_json(out / "frozen_selection.json", receipt)
    manifest.update(stage="frozen", selected=selected,
                    selected_model_sha256=sha256(out / selected["model"]),
                    frozen_selection_sha256=sha256(out / "frozen_selection.json"),
                    evaluation_modes=list(EVALUATION_MODES))
    write_json(out / "manifest.json", manifest)
    emit("frozen", selected=selected)


def _frozen_run(args):
    out, manifest, config, specs = load_run(args)
    if manifest["stage"] not in {"frozen", "evaluating", "complete"}:
        raise ValueError("Freeze validation-selected controller before test or paper evaluation.")
    receipt_path = out / "frozen_selection.json"
    if sha256(receipt_path) != manifest["frozen_selection_sha256"]:
        raise RuntimeError("Frozen selection receipt changed.")
    receipt = read_json(receipt_path)
    if (receipt["selection"] != manifest["selected"] or
            receipt["contract"] != _contract(manifest) or
            receipt["evaluation_modes"] != manifest["evaluation_modes"] or
            receipt["baseline"] != _baseline_contract(out, manifest)):
        raise RuntimeError("Frozen configuration changed.")
    for name, digest in receipt["selection_files_sha256"].items():
        if sha256(out / name) != digest:
            raise RuntimeError("Frozen selection candidate changed.")
    _verify_selection(out, manifest, manifest["selected"], specs)
    if sha256(out / manifest["selected"]["model"]) != manifest["selected_model_sha256"]:
        raise RuntimeError("Frozen model changed.")
    return out, manifest, config, specs


def _evaluation_policy(out, manifest, spec, mode, decisions):
    baseline = baseline_for(manifest, spec)
    if mode == "fixed":
        return lambda obs: manifest["best_fixed_index"]
    if mode == "fixed_by_dimension":
        return lambda obs: baseline
    learned = make_learned_policy(out, manifest["selected"], baseline, decisions)
    if mode == "rl":
        return learned
    if mode != "rl_static":
        raise ValueError("Unknown evaluation mode.")
    first = []
    def policy(observation):
        if not first:
            first.append(learned(observation))
        return first[0]
    return policy


def _run_evaluation(out, manifest, config, cases, split, modes, repeats, paper=False):
    from rl_tt_dgmres_v3.runner import run_episode
    folder = out / ("paper_evaluation" if paper else "evaluation")
    protocol = {
        "contract": _contract(manifest), "frozen_sha256": manifest["frozen_selection_sha256"],
        "specs": [asdict(s) for s in cases], "modes": list(modes), "repeats": repeats,
        "paper": paper,
    }
    ensure_protocol(folder / "protocol.json", protocol)
    completed_path = folder / "completed_receipt.json"
    if completed_path.exists():
        previous = read_json(completed_path)
        if (previous["protocol_sha256"] != sha256(folder / "protocol.json") or
                previous["frozen_selection_sha256"] != manifest["frozen_selection_sha256"]):
            raise RuntimeError("Completed evaluation belongs to a different protocol.")
        for relative, digest in previous["raw_files_sha256"].items():
            if sha256(out / relative) != digest:
                raise RuntimeError("Completed evaluation logs changed; refusing silent reuse.")
    rng = random.Random(manifest["seed"] + (9000 if paper else 5000))
    rows = []
    result_files = {}
    for problem_index, spec in enumerate(cases):
        local_config = config
        if paper:
            local_config = replace(config,
                tolerance=manifest["paper_protocol"]["tolerance_by_dimension"][str(spec.dims)],
                seconds_budget=manifest["paper_protocol"]["seconds_budget"])
        problem = build_problem(spec)
        for repeat in range(repeats):
            ordering = list(modes)
            rng.shuffle(ordering)
            for mode in ordering:
                path = folder / split / f"{spec.name}_{mode}_r{repeat}.json"
                if path.exists():
                    detail = read_json(path)
                    summary = detail["summary"]
                    if (summary["problem"] != asdict(spec) or summary["config"] != asdict(local_config) or
                            summary["mode"] != mode or summary["repeat"] != repeat or
                            summary["split"] != split):
                        raise RuntimeError("Cached evaluation differs from its exact case/configuration.")
                else:
                    decisions = []
                    policy = _evaluation_policy(out, manifest, spec, mode, decisions)
                    detail = run_episode(problem, local_config, policy, evaluate_reference=True,
                                         compute_policy_features=mode not in {"fixed", "fixed_by_dimension"})
                    detail["summary"].update(mode=mode, split=split, repeat=repeat,
                                              instance_key=instance_key(spec),
                                              baseline_index=baseline_for(manifest, spec))
                    detail["decisions"] = decisions
                    detail["construction_metadata"] = problem.construction_metadata()
                    write_json(path, compact(detail))
                rows.append(detail["summary"])
                result_files[str(path.relative_to(out))] = sha256(path)
                write_json(out / ("paper_summary.json" if paper else "evaluation_summary.json"), rows)
        emit("paper_case" if paper else "heldout", split=split, problem=spec.name,
             done=problem_index + 1, total=len(cases))
    summary_name = "paper_summary.json" if paper else "evaluation_summary.json"
    write_json(folder / "completed_receipt.json", {
        "summary_sha256": sha256(out / summary_name),
        "raw_files_sha256": result_files, "protocol_sha256": sha256(folder / "protocol.json"),
        "frozen_selection_sha256": manifest["frozen_selection_sha256"],
    })
    return rows


def evaluate(args):
    out, manifest, config, specs = _frozen_run(args)
    manifest.update(evaluation_started=True, stage="evaluating")
    write_json(out / "manifest.json", manifest)
    warm_up(specs["train"][0], config)
    rows = _run_evaluation(out, manifest, config, specs["test"], "test",
                           manifest["evaluation_modes"], manifest["repeats"])
    _assert_complete(rows, {"test": specs["test"]}, manifest["evaluation_modes"], manifest["repeats"])
    source_check(manifest)
    manifest["stage"] = "complete"
    write_json(out / "manifest.json", manifest)
    report(out, manifest, rows)


def paper(args):
    out, manifest, config, specs = _frozen_run(args)
    manifest["paper_started"] = True
    write_json(out / "manifest.json", manifest)
    warm_up(specs["train"][0], config)
    protocol = manifest["paper_protocol"]
    rows = _run_evaluation(out, manifest, config, specs["paper"], "paper",
                           protocol["modes"], protocol["repeats"], paper=True)
    _assert_complete(rows, {"paper": specs["paper"]}, protocol["modes"], protocol["repeats"])
    source_check(manifest)
    comparisons = paired_speedups(rows, "fixed_by_dimension", "rl", "paper")
    write_json(out / "paper_report.json", {
        "purpose": protocol["purpose"], "known_historical_cases_not_heldout": True,
        "never_used_for_selection": True, "source_seed": PAPER_SEED,
        "protocol": protocol, "paired_speedups": comparisons,
        "aggregates": {mode: aggregate([r for r in rows if r["mode"] == mode]) for mode in protocol["modes"]},
        "rows": rows,
    })
    manifest["paper_complete"] = True
    write_json(out / "manifest.json", manifest)
    emit("paper_complete", cases=len(specs["paper"]), speedup=comparisons["geometric_mean_speedup"])


def _assert_complete(rows, specs, modes, repeats):
    expected = {(split, s.name if hasattr(s, "name") else s["name"], mode, repeat)
                for split, cases in specs.items() for s in cases for mode in modes for repeat in range(repeats)}
    observed = [(r["split"], r["problem"]["name"], r["mode"], r["repeat"]) for r in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Missing, extra or duplicated evaluation keys; no final claim is generated.")


def report(out, manifest, rows):
    if manifest["stage"] != "complete":
        raise ValueError("Final claims require a completed heldout evaluation.")
    specs = read_json(out / "splits.json")
    _assert_complete(rows, {"test": specs["test"]}, manifest["evaluation_modes"], manifest["repeats"])
    receipt = read_json(out / "evaluation" / "completed_receipt.json")
    if (receipt["summary_sha256"] != sha256(out / "evaluation_summary.json") or
            receipt["frozen_selection_sha256"] != manifest["frozen_selection_sha256"] or
            receipt["protocol_sha256"] != sha256(out / "evaluation" / "protocol.json") or
            read_json(out / "evaluation_summary.json") != clean(rows)):
        raise RuntimeError("Evaluation summary no longer matches its completed frozen experiment.")
    raw_rows = []
    for relative, digest in receipt["raw_files_sha256"].items():
        if sha256(out / relative) != digest:
            raise RuntimeError("An evaluated raw result changed.")
        raw_rows.append(read_json(out / relative)["summary"])
    _assert_complete(raw_rows, {"test": specs["test"]}, manifest["evaluation_modes"], manifest["repeats"])
    row_key = lambda r: (r["split"], r["problem"]["name"], r["mode"], r["repeat"])
    if sorted(raw_rows, key=row_key) != sorted(clean(rows), key=row_key):
        raise RuntimeError("Reported rows disagree with their actual solve logs.")
    grouped = {mode: aggregate([r for r in rows if r["mode"] == mode]) for mode in manifest["evaluation_modes"]}
    by_dimension = {
        str(d): {mode: aggregate([r for r in rows if r["problem"]["dims"] == d and r["mode"] == mode])
                 for mode in manifest["evaluation_modes"]}
        for d in manifest["train_dimensions"]}
    comparisons = []
    for baseline, mode in (("fixed_by_dimension", "rl"), ("fixed", "rl"),
                           ("fixed_by_dimension", "rl_static"), ("rl_static", "rl")):
        comparison = paired_speedups(rows, baseline, mode, "test")
        values = [p["speedup"] for p in comparison["pairs"]]
        if values:
            logs = np.log(values)
            rng = np.random.default_rng(manifest["seed"] + 99)
            draws = rng.choice(logs, size=(10000, len(logs)), replace=True).mean(axis=1)
            comparison["bootstrap_95_percent_interval"] = np.exp(np.quantile(draws, [0.025, 0.975])).tolist()
        comparisons.append(comparison)
    primary = comparisons[0]
    fixed, rl = grouped["fixed_by_dimension"], grouped["rl"]
    gain = primary["geometric_mean_speedup"]
    evidence = {
        "primary_baseline": "fixed_by_dimension",
        "success_count_no_lower": rl["successes"] >= fixed["successes"],
        "paired_speedup_above_one": gain is not None and gain > 1,
        "paired_ci_lower_above_one": primary.get("bootstrap_95_percent_interval", [0])[0] > 1,
        "success_rate_statistical_noninferiority_proven": False,
        "operator_disjoint": False,
        "scope": manifest["split_contract"]["generalization"],
    }
    write_json(out / "aggregate.json", grouped)
    write_json(out / "aggregate_by_dimension.json", by_dimension)
    write_json(out / "paired_speedups.json", comparisons)
    write_json(out / "claim_audit.json", evidence)
    lines = ["# Neumann–Poisson 专项 RL TT-DGMRES", "",
        "采用论文的各向同性归一化 Neumann 算子和 CP3 右端项族；训练、验证、测试按完整右端项实例和随机种子隔离。测试与训练共享有限个算子，只评估同算子新右端项的泛化，不声称算子不相交泛化。原论文 seed 20260906 的展示例独立保存，不混入测试统计。",
        "固定策略和 RL 只读复用 V3 数值后端、单周期调度、保护、精度和预算。固定基线不计算不需要的策略特征；RL 的在线特征、推断和终审费用计入墙钟时间。模型加载及公共问题构造在计时外。",
        "主基线为验证集逐维数选优的固定动作；全局固定动作是辅助对照。学习门控与基线续跑同样使用逐维数固定动作，未见维数预先退回全局固定动作。",
        "训练标签来自同一真实重启状态的九动作分叉及完整续跑，拟合剩余时间/原始预算和失败风险。失败罚是成本目标的一部分，不构成收敛或普遍加速保证。",
        "模型、轮数及门槛在测试前冻结。测试所有失败保留；配对时间仅取双方所有重复均成功的实例，不能替代成功率。Bootstrap 的单位是新右端项实例，不支持对任意 Neumann 算子的推论。", "",
        "|方法|成功次数|全重复成功实例|", "|---|---:|---:|"]
    for name, group in grouped.items():
        lines.append(f"|{name}|{group['successes']}/{group['runs']}|{group['successful_instances_all_repeats']}/{group['instances']}|")
    lines.extend(["", "## 配对墙钟时间", ""])
    for c in comparisons:
        lines.append(f"- {c['baseline']}/{c['candidate']}: {c['geometric_mean_speedup']}; "
                     f"95% RHS-instance bootstrap interval {c.get('bootstrap_95_percent_interval')}; "
                     f"common successful instances {c['common_successful_instances']}.")
    lines.extend(["", f"验收记录：{json.dumps(evidence, ensure_ascii=False)}。",
                  "观测成功数不降低不等于已证明成功率非劣；小测试集区间只作有限样本描述。"
                  "rl_static 与 rl 的比较用于检查首周期选参以外的在线改参增益。"
                  "集成不确定性门控为经验规则，不是严格置信界。"
                  "论文已知案例使用本新求解器且无预条件，不冒充原图实现或原图计时。"])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit("complete", evidence=evidence, primary_speedup=gain)



def main(argv=None):
    parser = argparse.ArgumentParser(description="Paper-family Neumann real-rollout policy improvement")
    parser.add_argument("command", choices=["prepare", "fixed", "collect", "fit", "validate",
                                           "freeze", "evaluate", "paper", "report"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=202610021)
    parser.add_argument("--train-count", type=int, default=18)
    parser.add_argument("--validation-count", type=int, default=6)
    parser.add_argument("--test-count", type=int, default=6)
    parser.add_argument("--train-dims", default="4,5,6")
    parser.add_argument("--paper-dims", default="4,5,6,10")
    parser.add_argument("--max-states", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds-budget", type=float, default=60.0)
    parser.add_argument("--paper-seconds", type=float, default=60.0)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--max-tt-rank", type=int, default=96)
    parser.add_argument("--recovery-max-rank", type=int, default=256)
    parser.add_argument("--diagnostic-max-rank", type=int, default=512)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if (min(args.train_count, args.validation_count, args.test_count, args.repeats,
            args.max_states, args.threads, args.epochs) <= 0 or args.iteration < 0 or args.limit < 0):
        parser.error("Invalid count, iteration, thread count or fit epochs.")
    if args.command == "prepare":
        torch.set_num_threads(args.threads)
    if args.command == "report":
        out, manifest, _, _ = _frozen_run(args)
        report(out, manifest, read_json(out / "evaluation_summary.json"))
    else:
        {"prepare": prepare, "fixed": fixed_grid, "collect": collect, "fit": fit,
         "validate": validate, "freeze": freeze, "evaluate": evaluate, "paper": paper}[args.command](args)


if __name__ == "__main__":
    main()
