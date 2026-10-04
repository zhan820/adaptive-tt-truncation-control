from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from rl_tt_dgmres_v1.experiment import clean, write_json
from rl_tt_dgmres_v2.problems import build_problem, make_splits, spec_from_dict
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
    for folder in ("rl_tt_dgmres_v1", "rl_tt_dgmres_v2", "rl_tt_dgmres_v3", "adaptive_tt_dgmres"):
        paths.extend((root / folder).glob("*.py"))
    paths.extend((root / "adaptive_tt_dgmres" / "backend").glob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def source_check(manifest):
    if hashes() != manifest["source_sha256"]:
        raise RuntimeError("Source changed after preparation; start a new development run, not a silent resume.")


def score(rows, budget):
    return [sum(bool(r["accepted"]) for r in rows),
            -sum(r["wall_seconds"] if r["accepted"] else 5 * budget for r in rows)]


def prepare(args):
    from .runner import ACTIONS, FEATURE_NAMES, EpisodeConfig
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("Preparation requires a new directory.")
    out.mkdir(parents=True)
    specs = make_splits(args.seed, args.train_count, args.validation_count, args.test_count, args.ood_count)
    config = EpisodeConfig(seconds_budget=args.seconds_budget, max_iterations=args.max_iterations)
    config.validate()
    manifest = {
        "version": 3, "stage": "prepared", "seed": args.seed,
        "config": asdict(config), "source_sha256": hashes(),
        "features": list(FEATURE_NAMES), "actions": [asdict(a) for a in ACTIONS],
        "split_counts": {k: len(v) for k, v in specs.items()},
        "max_states_per_problem": args.max_states, "repeats": args.repeats,
        "validation_selection": "maximize success count, then minimize failure-penalized total solve time",
        "acceptance_protocol": {
            "same_backend": True, "same_numerical_safeguards": True,
            "same_precision_and_budget": True, "training_reference_labels": False,
            "all_online_feature_and_inference_cost_included": True,
            "heldout_order": "serial, randomly interleaved within instance and repeat",
            "primary_claim": "no lower heldout success count AND paired geometric time speedup > 1",
            "strong_claim": "primary claim AND 95% instance-bootstrap speedup lower bound > 1",
            "heldout_not_used_for_selection": True,
        },
        "evaluation_started": False,
    }
    root = Path(__file__).resolve().parent.parent
    for name in manifest["source_sha256"]:
        target = out / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    write_json(out / "splits.json", {k: [asdict(s) for s in v] for k, v in specs.items()})
    write_json(out / "manifest.json", manifest)
    emit("prepared", output=str(out), split_counts=manifest["split_counts"])


def load_run(args, training=False):
    from .runner import EpisodeConfig
    out = Path(args.output).resolve()
    manifest = read_json(out / "manifest.json")
    source_check(manifest)
    if training and manifest["evaluation_started"]:
        raise RuntimeError("This run has entered heldout evaluation; further training/selection is prohibited.")
    specs = {k: [spec_from_dict(s) for s in v] for k, v in read_json(out / "splits.json").items()}
    return out, manifest, EpisodeConfig(**manifest["config"]), specs


def warm_up(spec, config):
    from .runner import run_episode
    run_episode(build_problem(spec), config, lambda observation: 3, evaluate_reference=False)


def compact(detail):
    return {k: v for k, v in detail.items() if k not in {"solution", "round_calls", "solver_history"}}


def fixed_grid(args):
    from .runner import ACTIONS, run_episode
    out, manifest, config, specs = load_run(args, training=True)
    warm_up(specs["train"][0], config)
    jobs = [(i, j) for i in range(len(specs["validation"])) for j in range(len(ACTIONS))]
    random.Random(manifest["seed"] + 8).shuffle(jobs)
    started = time.perf_counter()
    for done, (i, j) in enumerate(jobs, 1):
        spec = specs["validation"][i]
        path = out / "fixed_validation" / f"{spec.name}_a{j}.json"
        if not path.exists():
            detail = run_episode(build_problem(spec), config, lambda obs, a=j: a, evaluate_reference=False,
                                 compute_policy_features=False)
            write_json(path, compact(detail))
        summary = read_json(path)["summary"]
        emit("fixed_grid", done=done, total=len(jobs), action=j, problem=spec.name,
             success=summary["accepted"], seconds=summary["wall_seconds"])
    grid = []
    for j in range(len(ACTIONS)):
        rows = [read_json(out / "fixed_validation" / f"{s.name}_a{j}.json")["summary"]
                for s in specs["validation"]]
        grid.append(dict(action=j, score=score(rows, config.seconds_budget), rows=rows))
    best = max(grid, key=lambda row: tuple(row["score"]))
    write_json(out / "baseline_validation.json", grid)
    manifest.update(best_fixed_index=best["action"], stage="fixed_selected")
    manifest["fixed_grid_last_execution_seconds"] = time.perf_counter() - started
    write_json(out / "manifest.json", manifest)
    emit("fixed_selected", action=best["action"], score=best["score"])


def make_learned_policy(out, selection, decisions=None):
    from .learning import RolloutValueLearner, GateConfig
    if selection.get("model_sha256") and sha256(out / selection["model"]) != selection["model_sha256"]:
        raise RuntimeError("Selection refers to a different model hash.")
    learner = RolloutValueLearner.load(out / selection["model"])
    gate = GateConfig(**selection["gate"])
    def policy(observation):
        action, decision = learner.policy(observation, baseline=selection["baseline"], gate=gate)
        if decisions is not None:
            decisions.append(decision)
        return action
    return policy


def collect(args):
    from .runner import ACTIONS, run_episode
    out, manifest, config, specs = load_run(args, training=True)
    baseline = manifest["best_fixed_index"]
    iteration = args.iteration
    continuation_id = "baseline" if iteration == 0 else f"policy_iteration_{iteration}"
    if iteration == 0:
        continuation = lambda obs: baseline
        selected = None
    else:
        selected = read_json(out / f"selection_{iteration - 1}.json")
        continuation = make_learned_policy(out, selected)
    folder = out / f"rollouts_{iteration}"
    folder.mkdir(exist_ok=True)
    chosen_specs = specs["train"][:args.limit] if args.limit else specs["train"]
    ensure_protocol(folder / "collection_protocol.json", {
        "continuation_policy": continuation_id, "selection": selected,
        "baseline": baseline, "source_sha256": manifest["source_sha256"],
        "config": asdict(config), "specs": [asdict(s) for s in chosen_specs],
        "max_states": manifest["max_states_per_problem"],
        "candidate_actions": list(range(len(ACTIONS))),
        "origin": "same real outer-cycle checkpoint, one candidate action then stated continuation",
        "checkpoint_overhead": "offline cloning excluded; normal per-cycle initialization included",
    })
    warm_up(specs["train"][0], config)
    started = time.perf_counter()
    rows = []
    for problem_index, spec in enumerate(chosen_specs):
        problem = build_problem(spec)
        state_file = folder / (spec.name + "_states.pt")
        if state_file.exists():
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
            write_json(folder / (spec.name + "_collection.json"), compact(trajectory))
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
                        "split": "train",
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
    })
    emit("collection_complete", iteration=iteration, rows=len(rows), seconds=time.perf_counter()-started)


def fit(args):
    from .learning import LearningConfig, RolloutValueLearner
    from .runner import FEATURE_NAMES
    out, manifest, config, specs = load_run(args, training=True)
    path = out / f"rollouts_{args.iteration}" / "rows.json"
    rows = read_json(path)
    target = "baseline" if args.iteration == 0 else f"policy_iteration_{args.iteration}"
    learning_config = LearningConfig(seed=manifest["seed"] + args.iteration, epochs=args.epochs)
    fit_protocol = {"rows_sha256": sha256(path), "config": asdict(learning_config),
                    "target_policy": target, "source_sha256": manifest["source_sha256"]}
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
    from .runner import run_episode
    from .learning import GateConfig
    out, manifest, config, specs = load_run(args, training=True)
    model = f"controller_iteration_{args.iteration}.pt"
    gates = [
        GateConfig(min_gain=0.0, uncertainty_scale=0.0, max_failure_increase=0.15,
                   max_failure_probability=0.50, min_action_samples=3),
        GateConfig(min_gain=0.002, uncertainty_scale=0.5, max_failure_increase=0.05,
                   max_failure_probability=0.25, min_action_samples=3),
        GateConfig(min_gain=0.005, uncertainty_scale=1.0, max_failure_increase=0.02,
                   max_failure_probability=0.15, min_action_samples=3),
    ]
    selections = [dict(model=model, gate=asdict(g), baseline=manifest["best_fixed_index"],
                       iteration=args.iteration, gate_index=i, model_sha256=sha256(out / model)) for i, g in enumerate(gates)]
    ensure_protocol(out / f"validation_protocol_{args.iteration}.json", {
        "selections": selections, "specs": [asdict(s) for s in specs["validation"]],
        "config": asdict(config), "source_sha256": manifest["source_sha256"],
    })
    warm_up(specs["train"][0], config)
    jobs = [(i, j) for i in range(len(specs["validation"])) for j in range(len(gates))]
    random.Random(manifest["seed"] + 200 + args.iteration).shuffle(jobs)
    policies = [make_learned_policy(out, s) for s in selections]
    for done, (i, j) in enumerate(jobs, 1):
        spec = specs["validation"][i]
        path = out / f"validation_{args.iteration}" / f"{spec.name}_g{j}.json"
        if not path.exists():
            detail = run_episode(build_problem(spec), config, policies[j], evaluate_reference=False)
            write_json(path, compact(detail))
        summary = read_json(path)["summary"]
        emit("validation", done=done, total=len(jobs), iteration=args.iteration,
             gate=j, success=summary["accepted"], seconds=summary["wall_seconds"])
    for j, selection in enumerate(selections):
        rows = [read_json(out / f"validation_{args.iteration}" / f"{s.name}_g{j}.json")["summary"]
                for s in specs["validation"]]
        selection.update(score=score(rows, config.seconds_budget), rows=rows)
    best = max(selections, key=lambda row: tuple(row["score"]))
    write_json(out / f"validation_{args.iteration}.json", selections)
    write_json(out / f"selection_{args.iteration}.json", {k: v for k, v in best.items() if k != "rows"})
    emit("selection", **{k: v for k, v in best.items() if k != "rows"})


def freeze(args):
    out, manifest, config, specs = load_run(args, training=True)
    selections = [read_json(p) for p in sorted(out.glob("selection_*.json"))]
    if not selections:
        raise ValueError("Validate at least one learned policy before freezing.")
    selected = max(selections, key=lambda row: tuple(row["score"]))
    if sha256(out / selected["model"]) != selected["model_sha256"]:
        raise RuntimeError("Selected score does not refer to the current model.")
    manifest.update(stage="frozen", selected=selected,
                    selected_model_sha256=hashlib.sha256((out / selected["model"]).read_bytes()).hexdigest(),
                    evaluation_modes=["fixed", "heuristic", "rl", "rl_static"])
    write_json(out / "manifest.json", manifest)
    emit("frozen", selected=selected)


def evaluate(args):
    from .runner import run_episode, heuristic_action
    out, manifest, config, specs = load_run(args)
    if manifest["stage"] not in {"frozen", "evaluating", "complete"}:
        raise ValueError("Freeze validation-selected controller first.")
    selected = manifest["selected"]
    if hashlib.sha256((out / selected["model"]).read_bytes()).hexdigest() != manifest["selected_model_sha256"]:
        raise RuntimeError("Frozen controller changed.")
    manifest.update(evaluation_started=True, stage="evaluating")
    write_json(out / "manifest.json", manifest)
    warm_up(specs["train"][0], config)
    rng = random.Random(manifest["seed"] + 5000)
    rows = []
    for split in ("test", "ood"):
        for problem_index, spec in enumerate(specs[split]):
            problem = build_problem(spec)
            for repeat in range(manifest["repeats"]):
                modes = list(manifest["evaluation_modes"])
                rng.shuffle(modes)
                for mode in modes:
                    path = out / "evaluation" / split / f"{spec.name}_{mode}_r{repeat}.json"
                    if path.exists():
                        detail = read_json(path)
                    else:
                        decisions = []
                        learned = make_learned_policy(out, selected, decisions)
                        if mode == "fixed":
                            policy = lambda obs: manifest["best_fixed_index"]
                        elif mode == "heuristic":
                            policy = heuristic_action
                        elif mode == "rl":
                            policy = learned
                        else:
                            first = []
                            def policy(observation):
                                if not first:
                                    first.append(learned(observation))
                                return first[0]
                        detail = run_episode(problem, config, policy, evaluate_reference=True,
                                             compute_policy_features=(mode != "fixed"))
                        detail["summary"].update(mode=mode, split=split, repeat=repeat)
                        detail["decisions"] = decisions
                        write_json(path, compact(detail))
                    rows.append(detail["summary"])
                    write_json(out / "evaluation_summary.json", rows)
            emit("heldout", split=split, problem=spec.name, done=problem_index + 1, total=len(specs[split]))
    expected = (len(specs["test"]) + len(specs["ood"])) * manifest["repeats"] * len(manifest["evaluation_modes"])
    if len(rows) != expected:
        raise RuntimeError("Incomplete evaluation.")
    source_check(manifest)
    manifest["stage"] = "complete"
    write_json(out / "manifest.json", manifest)
    report(out, manifest, rows)


def report(out, manifest, rows):
    if manifest["stage"] != "complete":
        raise ValueError("Final claims require a completed heldout evaluation.")
    specs = read_json(out / "splits.json")
    expected = {(split, s["name"], mode, repeat) for split in ("test", "ood")
                for s in specs[split] for mode in manifest["evaluation_modes"]
                for repeat in range(manifest["repeats"])}
    observed = [(r["split"], r["problem"]["name"], r["mode"], r["repeat"]) for r in rows]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError("Missing, extra or duplicated evaluation keys; no final claim is generated.")
    grouped = {f"{split}/{mode}": aggregate([r for r in rows if (split == "all" or r["split"] == split) and r["mode"] == mode])
               for split in ("test", "ood", "all") for mode in manifest["evaluation_modes"]}
    comparisons = []
    for split in ("test", "ood", None):
        for baseline, mode in (("fixed", "rl"), ("fixed", "rl_static"), ("rl_static", "rl")):
            comparison = paired_speedups(rows, baseline, mode, split)
            values = [p["speedup"] for p in comparison["pairs"]]
            if values:
                logs = np.log(values)
                rng = np.random.default_rng(manifest["seed"] + 99)
                draws = rng.choice(logs, size=(10000, len(logs)), replace=True).mean(axis=1)
                comparison["bootstrap_95_percent_interval"] = np.exp(np.quantile(draws, [0.025, 0.975])).tolist()
            comparisons.append(comparison)
    primary = next(c for c in comparisons if c["split"] is None and c["candidate"] == "rl" and c["baseline"] == "fixed")
    fixed, rl = grouped["all/fixed"], grouped["all/rl"]
    gain = primary["geometric_mean_speedup"]
    evidence = {
        "success_count_no_lower": rl["successes"] >= fixed["successes"],
        "paired_speedup_above_one": gain is not None and gain > 1,
        "paired_ci_lower_above_one": primary.get("bootstrap_95_percent_interval", [0])[0] > 1,
        "success_rate_statistical_noninferiority_proven": False,
    }
    write_json(out / "aggregate.json", grouped)
    write_json(out / "paired_speedups.json", comparisons)
    write_json(out / "claim_audit.json", evidence)
    lines = ["# RL TT-DGMRES V3：真实续跑成本驱动的策略改进", "",
             "固定策略和学习策略共享数值后端、外层周期调度、截断保护、验收阈值及时间预算。固定策略不计算其不需要的策略特征；RL 在线特征和推断耗时全部计入求解时间。",
             "训练时在真实周期状态分支：候选首动作后按基准/上一轮策略继续求解，使用实际完成时间与失败记录拟合动作价值，再进行策略改进。测试时网络冻结，仅在线选参数。",
             "固定参数在验证集九个动作中选优。所有测试实例及所有失败均保留；配对加速比仅针对双方所有重复成功的实例，不能替代成功率。", "",
             "|分组/方法|成功次数|全重复成功实例|", "|---|---:|---:|"]
    for name, group in grouped.items():
        lines.append(f"|{name}|{group['successes']}/{group['runs']}|{group['successful_instances_all_repeats']}/{group['instances']}|")
    lines += ["", "## 配对时间", ""]
    for c in comparisons:
        lines.append(f"- {c['split'] or 'all'} {c['baseline']}/{c['candidate']}: {c['geometric_mean_speedup']}, "
                     f"95% 实例自助区间 {c.get('bootstrap_95_percent_interval')}, 共同成功实例 {c['common_successful_instances']}。")
    lines += ["", "## 结果边界", "",
              f"验收记录：{json.dumps(evidence, ensure_ascii=False)}。", "",
              "这是有限测试集上的实测证据，不是任意奇异系统上的速度或收敛定理。成功数不降低不等于已统计证明成功率非劣。",
              "集成模型的不确定性门控是经验保护，不是已证明的置信界。rl_static 仅在首周期选择一次，以检验后续在线改动是否有增量收益。",
              "问题仍限于已知常量零空间的指数一对称图拉普拉斯 Kronecker 和；未据此推广到一般非正规或高指数问题。",
              "V3 统一按单周期外层调度以确保可恢复训练状态；因此不要把与 V2 的时间差全归因于学习。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit("complete", evidence=evidence, primary_speedup=gain)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Auditable real-rollout policy improvement for TT-DGMRES")
    parser.add_argument("command", choices=["prepare", "fixed", "collect", "fit", "validate", "freeze", "evaluate", "report"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=202610013)
    parser.add_argument("--train-count", type=int, default=36)
    parser.add_argument("--validation-count", type=int, default=12)
    parser.add_argument("--test-count", type=int, default=18)
    parser.add_argument("--ood-count", type=int, default=6)
    parser.add_argument("--max-states", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds-budget", type=float, default=20.0)
    parser.add_argument("--max-iterations", type=int, default=240)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    if min(args.train_count, args.validation_count, args.test_count, args.repeats, args.max_states) <= 0 or args.ood_count < 0:
        parser.error("Invalid problem/repeat counts.")
    torch.set_num_threads(1)
    if args.command == "report":
        out, manifest, _, _ = load_run(args)
        report(out, manifest, read_json(out / "evaluation_summary.json"))
    else:
        {"prepare": prepare, "fixed": fixed_grid, "collect": collect, "fit": fit,
         "validate": validate, "freeze": freeze, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
