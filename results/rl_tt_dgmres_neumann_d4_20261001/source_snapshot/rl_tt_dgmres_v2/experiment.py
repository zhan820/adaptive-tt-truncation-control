from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from rl_tt_dgmres_v1.experiment import clean, write_json, evaluation_score
from rl_tt_dgmres_v1.learning import QAgent
from rl_tt_dgmres_v1.runner import run_episode as run_v1, EpisodeConfig as V1Config
from .problems import make_splits, build_problem
from .runner import ACTIONS, RECOVERY_ACTION, FEATURE_NAMES, EpisodeConfig, run_episode, heuristic_action


def source_hashes():
    root = Path(__file__).resolve().parent.parent
    files = list((root / "rl_tt_dgmres_v2").glob("*.py"))
    files += list((root / "rl_tt_dgmres_v1").glob("*.py"))
    files += list((root / "adaptive_tt_dgmres").glob("*.py"))
    files += list((root / "adaptive_tt_dgmres" / "backend").glob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def save_snapshot(output, hashes):
    root = Path(__file__).resolve().parent.parent
    for name in hashes:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)


def paired_speedups(rows, baseline, candidate, split=None):
    selected = [r for r in rows if split is None or r["split"] == split]
    pairs = []
    for name in sorted({r["problem"]["name"] for r in selected}):
        groups = [[r for r in selected if r["problem"]["name"] == name and r["mode"] == mode]
                  for mode in (baseline, candidate)]
        if not all(groups) or not all(r["accepted"] for group in groups for r in group):
            continue
        times = [float(np.median([r["wall_seconds"] for r in group])) for group in groups]
        pairs.append({"problem": name, "baseline_seconds": times[0],
                      "candidate_seconds": times[1], "speedup": times[0] / times[1]})
    ratios = [p["speedup"] for p in pairs]
    return {"baseline": baseline, "candidate": candidate, "split": split, "pairs": pairs,
            "common_successful_instances": len(pairs),
            "geometric_mean_speedup": float(np.exp(np.mean(np.log(ratios)))) if ratios else None}


def aggregate(rows):
    good = [r for r in rows if r["accepted"]]
    instances = {r["problem"]["name"] for r in rows}
    complete = sum(all(r["accepted"] for r in rows if r["problem"]["name"] == name)
                   for name in instances)
    return {"runs": len(rows), "successes": len(good), "instances": len(instances),
            "successful_instances_all_repeats": complete,
            "median_success_seconds": float(np.median([r["wall_seconds"] for r in good])) if good else None,
            "median_success_iterations": float(np.median([r["iterations"] for r in good])) if good else None,
            "status_counts": dict(Counter(r["status"] for r in rows)),
            "recoveries": sum(r.get("recoveries", 0) for r in rows),
            "shield_interventions": sum(r.get("shield_interventions", 0) for r in rows),
            "executed_action_counts": dict(Counter(a for r in rows for a in r.get("executed_action_indices", [])))}


def evaluate_policy(problems, config, policy, output):
    rows = []
    for problem in problems:
        detail = run_episode(problem, config, policy, evaluate_reference=False)
        write_json(output / (problem.spec.name + ".json"), detail)
        rows.append(detail["summary"])
    return rows


def update_agent(agent, detail, steps):
    for t in detail["transitions"]:
        agent.observe(t["state"], t["action"], t["reward"], t["next_state"], t["done"])
    return agent.train_steps(steps)


def train_one(problems, validation, config, output, args, seed, fixed_index):
    agent = QAgent(len(FEATURE_NAMES), list(range(len(ACTIONS))), seed=seed,
                   batch_size=32, hidden=64, gamma=1.0, learning_rate=0.0005)
    rng = np.random.default_rng(seed + 17)
    history, validation_history, order = [], [], []
    best_score, best_epoch = None, None
    training_seconds = 0.0
    for episode in range(1, args.episodes + 1):
        if not order:
            order = rng.permutation(len(problems)).tolist()
        problem = problems[order.pop()]
        # Cover the joint action set with real off-policy rollouts before greedy deployment.
        warm = episode <= min(len(problems), args.episodes // 3)
        epsilon = max(0.08, 0.8 * (1 - episode / max(1, args.episodes)))
        if warm:
            action_index = (episode - 1) % len(ACTIONS)
            policy = lambda obs, j=action_index: j
        else:
            policy = lambda obs: agent.select_action(obs, epsilon)
        started = time.perf_counter()
        detail = run_episode(problem, config, policy, evaluate_reference=False)
        loss = update_agent(agent, detail, args.train_steps)
        training_seconds += time.perf_counter() - started
        row = {"episode": episode, "problem": problem.spec.name, "warm_start": warm,
               "epsilon": epsilon, "accepted": detail["summary"]["accepted"],
               "seconds": detail["summary"]["wall_seconds"], "loss": loss,
               "transitions": len(detail["transitions"]),
               "recoveries": detail["summary"]["recoveries"],
               "reward": detail["summary"]["reward_sum"]}
        history.append(row)
        write_json(output / "train" / f"episode_{episode:04d}.json", detail)
        write_json(output / "training_history.json", history)
        if episode % args.eval_every == 0 or episode == args.episodes:
            rows = evaluate_policy(validation, config, lambda obs: agent.select_action(obs, 0.0),
                                   output / "validation" / f"episode_{episode:04d}")
            score = evaluation_score(rows, config.seconds_budget)
            validation_history.append({"episode": episode, "score": score, "rows": rows})
            metadata = {"features": list(FEATURE_NAMES), "config": asdict(config),
                        "joint_actions": [asdict(a) for a in ACTIONS], "episode": episode,
                        "recovery_action": asdict(RECOVERY_ACTION),
                        "validation_score": score, "training_seed": seed}
            if best_score is None or score > best_score:
                best_score, best_epoch = score, episode
                agent.save(output / "controller.pt", metadata)
            agent.save(output / "last_checkpoint.pt", metadata)
            write_json(output / "validation_history.json", validation_history)
        if episode % 8 == 0 or episode == args.episodes:
            print(json.dumps({"event": "training", "seed": seed, **row}, ensure_ascii=False), flush=True)
    return QAgent.load(output / "controller.pt"), {
        "seed": seed, "best_epoch": best_epoch, "best_score": best_score,
        "episodes": len(history), "transitions": sum(r["transitions"] for r in history),
        "training_seconds": training_seconds, "gradient_steps": agent.gradient_steps,
        "accepted_training_episodes": sum(r["accepted"] for r in history)}


def make_report(output, manifest, rows):
    labels = [(split, mode) for split in ("test", "ood") for mode in manifest["evaluation_modes"]]
    grouped = {f"{split}/{mode}": aggregate([r for r in rows if r["split"] == split and r["mode"] == mode])
               for split, mode in labels}
    comparisons = [paired_speedups(rows, baseline, candidate, split)
                   for split in ("test", "ood", None)
                   for baseline in ("legacy_fixed", "v2_fixed", "v2_heuristic")
                   for candidate in manifest["evaluation_modes"] if candidate.startswith("neural_")]
    lines = ["# RL TT-DGMRES 第二版：联合控制与可靠性保护", "",
             "第二版独立实现；第一版、原论文和图表未修改。结果来自实际运行，不是预设加速结论。", "",
             "- 在线联合选择更新维数、乘法/正交化/迭代解更新截断精度；网络在测试时冻结。",
             "- 选择性重正交、Arnoldi列绝对误差预算、QR/SVD尾能量截断、提前核验、残差增长拒绝、检查点回退和有限秩上限扩展共同用于第二版所有策略。",
             "- 有进展的近不变子空间终止按正常重启继续；失败后切换高精度保守动作，必要时以真实变换残差的一维最小化构造修正。附加运算均计时。",
             "- 最终重新计算未截断的变换残差与零空间分量；尾能量与残差检查是浮点核验，不是严格收敛证明。",
             "- 训练/验证/测试算子组分离；验证集选择固定参数与模型，测试结果不反用于调参。",
             f"- 求解目标 eta≤{manifest['config']['tolerance']:.1e}；每个配置重复 {manifest['args']['repeats']} 次，顺序随机化。",
             f"- 固定主策略由验证集在 {len(ACTIONS)} 个联合动作中选出：{manifest['best_fixed_action']}。共同保护可能覆盖该主策略。",
             "", "| 分组 | 成功运行/总数 | 所有重复成功的实例 | 成功耗时中位数/s | Arnoldi步数中位数 |",
             "|---|---:|---:|---:|---:|"]
    for label, a in grouped.items():
        seconds = '—' if a['median_success_seconds'] is None else f"{a['median_success_seconds']:.4f}"
        steps = '—' if a['median_success_iterations'] is None else f"{a['median_success_iterations']:.1f}"
        lines.append(f"| {label} | {a['successes']}/{a['runs']} | {a['successful_instances_all_repeats']}/{a['instances']} | {seconds} | {steps} |")
    lines += ["", "## 配对加速比", "", "先对同一实例的重复时间取中位数，再计算比值的几何平均。仅包含双方全部重复成功的实例，不能替代成功率。", ""]
    for c in comparisons:
        if c["geometric_mean_speedup"] is not None:
            lines.append(f"- {c['split'] or '合计'}：{c['baseline']} / {c['candidate']} = {c['geometric_mean_speedup']:.3f}，共同成功实例 {c['common_successful_instances']} 个。")
    lines += ["", "## 解释边界", "",
              "与第一版的差别包括底层舍入实现、正交策略、保护机制及参数选择，不能把全部收益归因于强化学习。只有相对同一第二版底层的固定/启发式策略的比较，才反映学习选择的增量收益。",
              "固定主策略与神经策略使用相同的数值核验、秩上限、失败恢复、精度目标和时间预算；失败重算与推断开销均包含在在线时间内。训练成本单列。",
              "当前问题仍为已知零空间的指数一对称奇异系统，增加了局部图结构和规模覆盖，但尚不代表一般非正规、高指数系统。训练种子有限，当前结果应视为可复现的开发验证。",
              "软件优化和有限失败回退不提供普遍收敛保证；回退耗尽、秩上限或时间预算仍可能导致失败。"]
    # This is a generated experiment artifact, not an edit of a user-authored document.
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "aggregate.json", grouped)
    write_json(output / "paired_speedups.json", comparisons)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[20261001, 20261002])
    parser.add_argument("--train-count", type=int, default=96)
    parser.add_argument("--validation-count", type=int, default=12)
    parser.add_argument("--test-count", type=int, default=12)
    parser.add_argument("--ood-count", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=288)
    parser.add_argument("--eval-every", type=int, default=96)
    parser.add_argument("--train-steps", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds-budget", type=float, default=30)
    parser.add_argument("--max-iterations", type=int, default=240)
    parser.add_argument("--skip-legacy", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        parser.error("Output exists; use a fresh directory to preserve experiment provenance.")
    if min(args.train_count, args.validation_count, args.episodes, args.eval_every, args.repeats) < 1:
        parser.error("Positive training/validation/run counts are required.")
    torch.set_num_threads(1)
    config = EpisodeConfig(seconds_budget=args.seconds_budget, max_iterations=args.max_iterations)
    config.validate()
    started = time.perf_counter()
    hashes = source_hashes()
    output.mkdir(parents=True)
    save_snapshot(output, hashes)
    specs = make_splits(args.seed, args.train_count, args.validation_count, args.test_count, args.ood_count)
    write_json(output / "splits.json", {key: [asdict(s) for s in values] for key, values in specs.items()})
    manifest = {"stage": "building", "args": vars(args), "config": asdict(config),
                "actions": [asdict(a) for a in ACTIONS], "features": list(FEATURE_NAMES),
                "recovery_action": asdict(RECOVERY_ACTION),
                "source_sha256": hashes, "split_counts": {k: len(v) for k, v in specs.items()}}
    write_json(output / "manifest.json", manifest)
    problems = {key: [build_problem(s) for s in values] for key, values in specs.items()}
    write_json(output / "dataset.json", {k: [p.construction_metadata() for p in values]
                                          for k, values in problems.items()})
    # Warm kernels using a training problem, outside every reported solve time.
    run_episode(problems["train"][0], config, lambda obs: 3, evaluate_reference=False)
    grid = []
    manifest["stage"] = "fixed_validation"
    write_json(output / "manifest.json", manifest)
    for j, action in enumerate(ACTIONS):
        rows = evaluate_policy(problems["validation"], config, lambda obs, j=j: j,
                               output / "fixed_validation" / f"action_{j}")
        grid.append({"index": j, "action": asdict(action),
                     "score": evaluation_score(rows, config.seconds_budget), "rows": rows})
        write_json(output / "baseline_validation.json", grid)
        print(json.dumps({"event": "fixed_validation", "action": j, "score": grid[-1]["score"]}), flush=True)
    best = max(grid, key=lambda r: tuple(r["score"]))
    fixed_index = best["index"]
    manifest.update({"best_fixed_index": fixed_index, "best_fixed_action": asdict(ACTIONS[fixed_index]), "stage": "training"})
    write_json(output / "manifest.json", manifest)
    agents, training = {}, []
    for seed in args.training_seeds:
        agent, stats = train_one(problems["train"], problems["validation"], config,
                                 output / f"seed_{seed}", args, seed, fixed_index)
        agents[f"neural_{seed}"] = agent
        training.append(stats)
        write_json(output / "training_summary.json", training)
    chosen = max(training, key=lambda r: tuple(r["best_score"]))
    manifest["chosen_training_seed"] = chosen["seed"]
    shutil.copy2(output / f"seed_{chosen['seed']}" / "controller.pt", output / "controller.pt")
    modes = ([] if args.skip_legacy else ["legacy_fixed"]) + ["v2_fixed", "v2_heuristic"] + list(agents)
    manifest.update({"stage": "heldout_evaluation", "evaluation_modes": modes})
    write_json(output / "manifest.json", manifest)
    rows, rng = [], np.random.default_rng(args.seed + 5000)
    for split in ("test", "ood"):
        for i, problem in enumerate(problems[split]):
            for repeat in range(args.repeats):
                order = list(modes)
                rng.shuffle(order)
                for mode in order:
                    if mode == "legacy_fixed":
                        old_config = V1Config(max_iterations=config.max_iterations, seconds_budget=config.seconds_budget,
                                              tolerance=config.tolerance, max_tt_rank=config.max_tt_rank)
                        detail = run_v1(problem, old_config, lambda obs: 0)
                    else:
                        policy = (lambda obs: fixed_index) if mode == "v2_fixed" else heuristic_action if mode == "v2_heuristic" else (
                            lambda obs, agent=agents[mode]: agent.select_action(obs, 0.0))
                        detail = run_episode(problem, config, policy)
                    detail["summary"].update({"mode": mode, "split": split, "repeat": repeat})
                    rows.append(detail["summary"])
                    write_json(output / "evaluation" / split / f"{problem.spec.name}_{mode}_r{repeat}.json", detail)
                    write_json(output / "evaluation_summary.json", rows)
            print(json.dumps({"event": "heldout", "split": split, "done": i + 1,
                              "total": len(problems[split]), "problem": problem.spec.name}), flush=True)
    if source_hashes() != hashes:
        raise RuntimeError("Numerical/training source changed during the experiment; results require review.")
    manifest.update({"stage": "complete", "total_seconds": time.perf_counter() - started,
                     "training": training})
    write_json(output / "manifest.json", manifest)
    make_report(output, manifest, rows)
    print(json.dumps({"event": "complete", "output": str(output), "seconds": manifest["total_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
