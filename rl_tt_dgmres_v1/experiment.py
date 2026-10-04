from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .learning import QAgent
from .problems import build_problem, make_splits, operator_group_key
from .runner import ACTIONS, FEATURE_NAMES, EpisodeConfig, run_episode


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2), encoding="utf-8")


def source_hashes():
    root = Path(__file__).resolve().parent.parent
    paths = sorted((root / "rl_tt_dgmres_v1").glob("*.py"))
    paths += sorted((root / "adaptive_tt_dgmres").glob("*.py"))
    paths += sorted((root / "adaptive_tt_dgmres" / "backend").glob("*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def restore_problem_order(seed, count, completed):
    """Restore the separate problem-sampling RNG, without consuming agent RNG."""
    rng = np.random.default_rng(seed + 1)
    order, consumed = [], []
    for _ in range(completed):
        if not order:
            order = rng.permutation(count).tolist()
        consumed.append(order.pop())
    return rng, order, consumed


def inherit_resume_arguments(args, explicit_options):
    source = Path(args.resume_from).resolve()
    manifest = read_json(source / "manifest.json")
    original = manifest["args"]
    for name, value in vars(args).copy().items():
        if name in ("output", "resume_from"):
            continue
        if name not in original:
            raise ValueError(f"Original run is missing argument {name}.")
        if name in explicit_options and value != original[name]:
            raise ValueError(f"Resume cannot change {name}: {original[name]!r} -> {value!r}.")
        setattr(args, name, original[name])
    args.resume_from = str(source)
    return manifest


def verify_resume_sources(source, prior, current):
    snapshot = source / "source_snapshot"
    previous = prior["source_sha256"]
    if set(previous) != set(current):
        raise ValueError("Source file inventory changed; review before resuming.")
    changes = {}
    root = Path(__file__).resolve().parent.parent
    for name, expected in previous.items():
        relative = Path(name.replace("\\", "/"))
        archived = snapshot / relative
        if not archived.is_file() or hashlib.sha256(archived.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Original source snapshot fails SHA-256 verification: {name}")
        if current[name] == expected:
            continue
        normalized = relative.as_posix()
        if normalized == "rl_tt_dgmres_v1/experiment.py":
            reason = "Resume orchestration, provenance, and time-accounting support."
        elif normalized == "rl_tt_dgmres_v1/runner.py":
            original = archived.read_text(encoding="utf-8")
            patched = original.replace(
                '"cycle": 0, "phase": phase, "requested_eps": float(eps), "effective_eps": 0.0,',
                '"cycle": self.cycle, "phase": phase, "requested_eps": float(eps), "effective_eps": 0.0,')
            anchor = '        "preserved_near_cancellation_calls": sum(r.get("preserved_near_cancellation", False)'
            addition = ('        "near_cancellation_check_seconds": sum(r.get("near_cancellation_check_seconds", 0.0)\n'
                        '                                               for r in rounder.logs),\n')
            patched = patched.replace(anchor, addition + anchor)
            if patched != (root / relative).read_text(encoding="utf-8"):
                raise ValueError("Runner differs beyond the two approved log-only changes.")
            reason = "Log-only: preserved-vector cycle ID and diagnostic-time summary; numerical path unchanged."
        else:
            raise ValueError(f"Numerical/training source changed: {name}")
        changes[name] = {"original_sha256": expected, "current_sha256": current[name], "reason": reason}
    return changes


def load_resume_state(source, prior, args, config, current_hashes):
    """Validate all retained data before creating any new experiment output."""
    source = Path(source)
    if prior["config"] != asdict(config) or prior["actions"] != list(ACTIONS) or prior["features"] != list(FEATURE_NAMES):
        raise ValueError("Resume configuration, action ordering, or feature ordering differs.")
    changes = verify_resume_sources(source, prior, current_hashes)
    agent = QAgent.load(source / "last_checkpoint.pt")
    best_agent = QAgent.load(source / "controller.pt")
    for candidate in (agent, best_agent):
        if (candidate.actions != list(ACTIONS) or candidate.observation_dim != len(FEATURE_NAMES)
                or candidate.metadata.get("features") != list(FEATURE_NAMES)
                or candidate.metadata.get("config") != asdict(config)
                or candidate.seed != args.seed or candidate.batch_size != args.batch_size
                or candidate.hidden != prior["hidden"] or candidate.gamma != 1.0
                or candidate.learning_rate != 0.001 or candidate.target_update != 100):
            raise ValueError("Checkpoint is incompatible with the original experiment.")
    completed = agent.metadata.get("episode")
    if not isinstance(completed, int) or not 0 < completed <= args.episodes:
        raise ValueError("Checkpoint episode is missing or outside the original schedule.")
    all_training = read_json(source / "training_history.json")
    training = [r for r in all_training if r["episode"] <= completed]
    if [r["episode"] for r in training] != list(range(1, completed + 1)):
        raise ValueError("Retained training history is not a contiguous checkpoint prefix.")
    rng, order, consumed = restore_problem_order(args.seed, args.train_count, completed)
    splits = read_json(source / "splits.json")
    retained_files = []
    for row, problem_index in zip(training, consumed):
        relative = Path("train") / f"episode_{row['episode']:04d}.json"
        detail = read_json(source / relative)
        if (row["problem"] != splits["train"][problem_index]["spec"]["name"]
                or detail["summary"]["problem"]["name"] != row["problem"]
                or len(detail["transitions"]) != row["transitions"]):
            raise ValueError(f"Retained trajectory does not match problem order/history: {relative}")
        retained_files.append(relative)
    if (len(agent.replay) != min(sum(r["transitions"] for r in training), agent.capacity)
            or agent.gradient_steps != training[-1]["loss"]["gradient_steps"]):
        raise ValueError("Checkpoint replay/gradient count disagrees with retained history.")
    validation = [r for r in read_json(source / "validation_history.json") if r["episode"] <= completed]
    expected_validation = [i for i in range(1, completed + 1)
                           if i % args.eval_every == 0 or i == args.episodes]
    if [r["episode"] for r in validation] != expected_validation:
        raise ValueError("Retained validation history does not match the checkpoint schedule.")
    best = max(validation, key=lambda r: tuple(r["score"]))
    if (best_agent.metadata.get("best_epoch") != best["episode"]
            or tuple(best_agent.metadata.get("validation_score", [])) != tuple(best["score"])):
        raise ValueError("Saved best controller is not the best retained validation model.")
    grid = read_json(source / "baseline_validation.json")
    if [row["delta"] for row in grid] != list(ACTIONS) or [row["index"] for row in grid] != list(range(len(ACTIONS))):
        raise ValueError("Fixed-baseline action grid changed.")
    for row in grid:
        if (len(row["rows"]) != args.validation_count
                or tuple(row["score"]) != evaluation_score(row["rows"], config.seconds_budget)):
            raise ValueError("Fixed-baseline validation records are inconsistent.")
    return {"agent": agent, "completed": completed, "training": training, "validation": validation,
            "best_epoch": best["episode"], "best_score": tuple(best["score"]), "grid": grid,
            "rng": rng, "order": order, "files": retained_files, "changes": changes,
            "excluded_episodes": [r["episode"] for r in all_training if r["episode"] > completed],
            "prior_training_solve_seconds": sum(r["seconds"] for r in training)}


def stage_resume_artifacts(source, output, state):
    source, output = Path(source), Path(output)
    for name in ("splits.json", "dataset.json", "warmup.json", "baseline_validation.json",
                 "controller.pt", "last_checkpoint.pt"):
        shutil.copy2(source / name, output / name)
    for relative in state["files"]:
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    for name in [f"fixed_{j}" for j in range(len(ACTIONS))] + [
            f"episode_{r['episode']:04d}" for r in state["validation"]]:
        shutil.copytree(source / "validation" / name, output / "validation" / name)
    shutil.copytree(source / "source_snapshot", output / "origin_source_snapshot")
    shutil.copy2(source / "manifest.json", output / "origin_manifest.json")
    write_json(output / "training_history.json", state["training"])
    write_json(output / "validation_history.json", state["validation"])
    return {relative.as_posix(): hashlib.sha256((source / relative).read_bytes()).hexdigest()
            for relative in state["files"]}


def snapshot_sources(output, hashes):
    root = Path(__file__).resolve().parent.parent
    for name in hashes:
        relative = Path(name.replace("\\", "/"))
        target = Path(output) / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, target)


def evaluation_score(rows, budget):
    return (sum(bool(r["accepted"]) for r in rows),
            -sum(min(r["wall_seconds"], budget) if r["accepted"] else budget for r in rows))


def aggregate(rows):
    successful = [r for r in rows if r["accepted"]]
    return {
        "runs": len(rows), "successes": len(successful),
        "success_rate": len(successful) / max(len(rows), 1),
        "median_success_seconds": float(np.median([r["wall_seconds"] for r in successful]))
        if successful else None,
        "median_success_iterations": float(np.median([r["iterations"] for r in successful]))
        if successful else None,
        "median_success_forward_error": float(np.median([r["forward_error"] for r in successful
                                                         if r["forward_error"] is not None]))
        if successful and any(r["forward_error"] is not None for r in successful) else None,
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "action_counts": dict(Counter(a for r in rows for a in r["action_indices"])),
    }


def policy_factory(name, agent, fixed_index, actions):
    if name == "neural":
        return lambda obs: agent.select_action(obs, epsilon=0.0)
    if name == "fixed":
        return lambda obs: fixed_index
    if name == "heuristic":
        loose = min(range(len(actions)), key=lambda j: abs(math.log10(actions[j]) + 3))
        strict = min(range(len(actions)), key=lambda j: abs(math.log10(actions[j]) + 6))
        return lambda obs: loose if obs[0] > 0.5 else strict
    raise ValueError(name)


def evaluate(problems, config, agent, fixed_index, actions, output, label, repeats=1,
             modes=("neural",), order_seed=0):
    output = Path(output)
    rows = []
    rng = np.random.default_rng(order_seed)
    for i, problem in enumerate(problems):
        for repeat in range(repeats):
            order = list(modes)
            rng.shuffle(order)
            for mode in order:
                detail = run_episode(problem, config, policy_factory(mode, agent, fixed_index, actions),
                                     actions, evaluate_reference=True)
                summary = detail["summary"]
                summary.update({"mode": mode, "repeat": repeat, "split": label})
                rows.append(summary)
                write_json(output / f"{problem.spec.name}_{mode}_r{repeat}.json", detail)
        print(json.dumps({"event": "evaluation", "split": label, "problem": problem.spec.name,
                          "done": i + 1, "total": len(problems)}, ensure_ascii=False), flush=True)
    return rows


def make_report(output, manifest, training, validation_grid, evaluation_rows, best_fixed, best_epoch):
    grouped = {}
    for split in ("test", "ood"):
        for mode in ("fixed", "heuristic", "neural"):
            grouped[f"{split}/{mode}"] = aggregate(
                [r for r in evaluation_rows if r["split"] == split and r["mode"] == mode])
    lines = ["# RL 自适应 TT-DGMRES 第一版：训练与留出验证", "",
             "这是固定重启、周期选择截断容差的第一版研究原型。下表为实际运行结果，不是论文最终基准。", "",
             f"- 控制范围：{manifest['config']['scope']}；更新维数 {manifest['config']['update_dim']}，"
             f"每个正常周期最多 {manifest['config']['update_dim'] + 1} 步 Arnoldi。",
             f"- 停止目标：独立重算变换后向指标 eta ≤ {manifest['config']['tolerance']:.1e}，"
             f"并检查零空间分量 ≤ {manifest['config']['range_tolerance']:.1e}。",
             f"- 数据规模：{manifest['split_counts']}；训练 {len(training)} 个完整 episode，"
             f"实际转移 {sum(r['transitions'] for r in training)} 条。",
             f"- 网络：两层 {manifest['hidden']} 节点的标量特征 MLP，Double-DQN，γ=1；"
             "训练后冻结。参考解未进入策略输入或训练奖励。",
             f"- 验证集选定固定容差：{best_fixed:.1e}；学习模型选自第 {best_epoch} 个 episode。",
             "- 所有策略保持相同正交、截断审计、投影和残差核验；在线时间包含这些成本及网络推理。", "",
             "| 分组 | 成功/运行 | 成功运行时间中位数/秒 | 成功运行 Arnoldi 步数中位数 |",
             "|---|---:|---:|---:|"]
    for label, a in grouped.items():
        seconds = "—" if a["median_success_seconds"] is None else f"{a['median_success_seconds']:.4f}"
        iterations = "—" if a["median_success_iterations"] is None else f"{a['median_success_iterations']:.1f}"
        lines.append(f"| {label} | {a['successes']}/{a['runs']} | {seconds} | {iterations} |")
    pairs = []
    for split in ("test", "ood"):
        problem_names = sorted({r["problem"]["name"] for r in evaluation_rows if r["split"] == split})
        for name in problem_names:
            selected = {mode: [r for r in evaluation_rows if r["split"] == split and
                              r["problem"]["name"] == name and r["mode"] == mode]
                        for mode in ("fixed", "neural")}
            if all(selected.values()) and all(r["accepted"] for rows in selected.values() for r in rows):
                fixed = float(np.median([r["wall_seconds"] for r in selected["fixed"]]))
                neural = float(np.median([r["wall_seconds"] for r in selected["neural"]]))
                pairs.append({"split": split, "problem": name, "fixed_seconds": fixed,
                              "neural_seconds": neural, "speedup": fixed / neural})
    if pairs:
        ratio = float(np.exp(np.mean(np.log([p["speedup"] for p in pairs]))))
        lines += ["", f"共同全部成功的 {len(pairs)} 个实例中，以重复计时中位数形成配对，"
                  f"固定基线时间/学习策略时间的几何均值为 {ratio:.3f}。"
                  "这一子集统计不替代上表的整体成功率。"]
        lines += ["学习策略在这批共同成功实例上观察到时间收益；仍需多训练种子与更大规模重复验证。"
                  if ratio > 1 else "本批次尚未观察到整体时间优势；已保留全部结果，不能据此写成加速结论。"]
    lines += ["", "## 训练与数据", "",
              "`splits.json` 保存完整生成参数和算子分组键；`dataset.json` 保存算子、右端和参考解摘要。",
              "`train/` 保存每次求解的完整周期转移、截断误差审计及动作序列；失败也保留。",
              "`training_history.json` 保存探索率、TD 损失与回报，`validation/` 保存模型选择依据。",
              "`controller.pt` 为验证集选出的模型，`last_checkpoint.pt` 为训练末尾可续训状态。",
              "测试结果未用于选容差、模型或动作。数据由参数确定性生成，不需要外部数据下载。", "",
              "## 使用边界", "",
              "当前仅指数一、已知零空间的对称 Neumann 问题族；不表示高指数或一般非正规奇异系统已验证。",
              "独立核验是无额外 TT 截断的浮点计算，不是区间算术证明。当前版本对拒绝截断直接判失败，"
              "尚未实现检查点回滚重算。时间预算在迭代边界检查，单个正在执行的运算可能越过预算。",
              "重启长度保持固定。`matvec` 只控制乘法后的舍入；`arnoldi` 是另行运行的可选对照，"
              "不能把两类结果混在同一加速结论中。",
              "训练仅一个随机种子；重复计时不是重复训练。核存储量不是进程峰值内存。",
              "首轮采用有限预算累计耗时和失败罚，失败罚为软目标，不提供收敛保证。", "",
              "训练成本单列，未并入每次在线求解时间；如果没有正的单次时间收益，就不能声称训练可以摊销。"]
    if manifest.get("resume"):
        resume = manifest["resume"]
        lines += ["", "## 检查点恢复与计时口径", "",
                  f"从 `{resume['source']}` 的第 {resume['checkpoint_episode']} 次训练检查点恢复。"
                  f"原目录中检查点之后的 {len(resume['excluded_episodes'])} 条训练记录未纳入本次训练历史；"
                  "这些原始记录仍完整保留在来源目录。模型、优化器、经验回放和局部随机数状态一起恢复。",
                  "固定基线验证和检查点之前的验证结果直接继承；测试集未参与恢复或选择。",
                  "旧日志中极小余量恒等保留记录的 cycle 字段可能为 0；不改写历史日志。"
                  "恢复代码仅修正该日志标签及诊断时间汇总，数值路径不变。",
                  f"本次恢复进程总时间：{manifest['total_seconds']:.2f} 秒（不是从头训练的端到端时间）；"
                  f"本次新增训练与参数更新时间：{manifest['resumed_training_seconds']:.2f} 秒。",
                  f"已继承训练记录的求解时间合计 {resume['prior_training_solve_seconds']:.2f} 秒。"
                  "旧检查点未保存参数更新等完整训练阶段耗时，因此累计训练墙钟时间未知，不能将二者相加冒充完整训练时间。"]
    else:
        lines += ["", f"总流程时间：{manifest.get('total_seconds', 0):.2f} 秒；"
                  f"训练 episode 与参数更新阶段：{manifest.get('training_seconds', 0):.2f} 秒。"]
    (Path(output) / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(Path(output) / "aggregate.json", {"groups": grouped, "paired": pairs})


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--episodes", type=int, default=192)
    parser.add_argument("--train-count", type=int, default=48)
    parser.add_argument("--validation-count", type=int, default=12)
    parser.add_argument("--test-count", type=int, default=12)
    parser.add_argument("--ood-count", type=int, default=6)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--eval-repeats", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--update-dim", type=int, default=12)
    parser.add_argument("--max-iterations", type=int, default=156)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--seconds-budget", type=float, default=20)
    parser.add_argument("--scope", choices=("matvec", "arnoldi"), default="matvec")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--threads", type=int, default=1)
    raw_args = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_args)
    prior = None
    if args.resume_from:
        explicit = {token.split("=", 1)[0][2:].replace("-", "_")
                    for token in raw_args if token.startswith("--")}
        try:
            prior = inherit_resume_arguments(args, explicit)
        except (OSError, KeyError, ValueError) as error:
            parser.error(str(error))
    if any(getattr(args, field) < 1 for field in ("episodes", "train_count", "validation_count",
            "test_count", "ood_count", "eval_every", "eval_repeats", "train_steps", "batch_size", "threads")):
        parser.error("Counts, repeats, and thread count must be positive.")
    output = Path(args.output).resolve()
    if args.resume_from and (output == Path(args.resume_from) or Path(args.resume_from) in output.parents):
        parser.error("Resume output must be outside the original run directory.")
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be new or empty; prior experiment results are never overwritten.")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    config = EpisodeConfig(update_dim=args.update_dim, max_iterations=args.max_iterations,
                           tolerance=args.tolerance, seconds_budget=args.seconds_budget, scope=args.scope)
    config.validate()
    current_hashes = source_hashes()
    state = load_resume_state(Path(args.resume_from), prior, args, config, current_hashes) if prior else None
    output.mkdir(parents=True, exist_ok=True)
    provenance = stage_resume_artifacts(args.resume_from, output, state) if state else None
    snapshot_sources(output, current_hashes)
    splits = make_splits(seed=args.seed, train_count=args.train_count,
                        validation_count=args.validation_count, test_count=args.test_count,
                        ood_count=args.ood_count)
    split_data = {key: [{"spec": asdict(s), "operator_group": operator_group_key(s)}
                        for s in specs] for key, specs in splits.items()}
    if state and clean(split_data) != read_json(output / "splits.json"):
        raise ValueError("Regenerated split definitions differ from the original run.")
    if not state:
        write_json(output / "splits.json", split_data)
    groups = {key: {operator_group_key(s) for s in specs} for key, specs in splits.items()}
    keys = list(groups)
    if any(groups[a] & groups[b] for i, a in enumerate(keys) for b in keys[i + 1:]):
        raise ValueError("Operator groups overlap across data splits.")
    problems = {key: [build_problem(s) for s in specs] for key, specs in splits.items()}
    manifest = {"version": "0.1.0", "stage": "running", "seed": args.seed,
                "config": asdict(config), "actions": list(ACTIONS), "features": list(FEATURE_NAMES),
                "split_counts": {k: len(v) for k, v in splits.items()}, "hidden": 64,
                "training_seed_count": 1, "evaluation_repeats": args.eval_repeats,
                "source_sha256": current_hashes, "args": vars(args),
                "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                "platform": platform.platform(), "threads": args.threads,
                "method": "Double-DQN finite-episode undiscounted time cost, fixed restart"}
    if state:
        manifest["resume"] = {"source": args.resume_from, "checkpoint_episode": state["completed"],
                              "excluded_episodes": state["excluded_episodes"],
                              "prior_training_solve_seconds": state["prior_training_solve_seconds"],
                              "prior_full_training_wall_seconds": None,
                              "inherited_trajectory_sha256": provenance,
                              "source_changes": state["changes"],
                              "time_scope": "Current resume invocation only; cumulative wall time unknown."}
    write_json(output / "manifest.json", manifest)
    dataset = {k: [{"name": p.spec.name, "rhs_norm": p.rhs_norm,
               "ab_norm": p.ab_norm, "lambda_min_positive": p.lambda_min_positive,
               "operator_group": operator_group_key(p.spec), "construction": p.construction_metadata()}
               for p in rows] for k, rows in problems.items()}
    if state and clean(dataset) != read_json(output / "dataset.json"):
        raise ValueError("Regenerated dataset metadata differs from the original run.")
    if not state:
        write_json(output / "dataset.json", dataset)
    agent = state["agent"] if state else QAgent(len(FEATURE_NAMES), list(ACTIONS), seed=args.seed, gamma=1.0,
                   batch_size=args.batch_size, learning_rate=0.001, target_update=100)
    # A fixed solve warms numerical libraries; it is not used for selection or test reporting.
    validation_grid = state["grid"] if state else []
    if not state:
        warmup = run_episode(problems["train"][0], config, lambda obs: 0, ACTIONS)
        write_json(output / "warmup.json", warmup)
        for action, delta in enumerate(ACTIONS):
            rows = evaluate(problems["validation"], config, agent, action, ACTIONS,
                            output / "validation" / f"fixed_{action}", "validation", modes=("fixed",))
            validation_grid.append({"delta": delta, "index": action, "rows": rows,
                                    "score": evaluation_score(rows, config.seconds_budget)})
    best_fixed_row = max(validation_grid, key=lambda row: row["score"])
    best_fixed = best_fixed_row["index"]
    write_json(output / "baseline_validation.json", validation_grid)
    print(json.dumps({"event": "baseline_selected", "delta": ACTIONS[best_fixed]}, ensure_ascii=False), flush=True)
    rng = state["rng"] if state else np.random.default_rng(args.seed + 1)
    training = state["training"] if state else []
    validation_history = state["validation"] if state else []
    best_score = state["best_score"] if state else (-1, -math.inf)
    best_epoch = state["best_epoch"] if state else 0
    training_seconds = 0.0
    order = state["order"] if state else []
    if state:
        print(json.dumps({"event": "resumed", "checkpoint_episode": state["completed"],
                          "excluded_episodes": state["excluded_episodes"], "target": args.episodes}), flush=True)
    for episode in range((state["completed"] if state else 0) + 1, args.episodes + 1):
        if not order:
            order = rng.permutation(len(problems["train"])).tolist()
        problem = problems["train"][order.pop()]
        epsilon = max(0.05, 1.0 - episode / max(1, 0.8 * args.episodes))
        train_started = time.perf_counter()
        detail = run_episode(problem, config, lambda obs: agent.select_action(obs, epsilon),
                             ACTIONS, evaluate_reference=False)
        for transition in detail["transitions"]:
            agent.observe(transition["state"], transition["action"], transition["reward"],
                          transition["next_state"], transition["done"])
        losses = agent.train_steps(args.train_steps)
        training_seconds += time.perf_counter() - train_started
        row = {"episode": episode, "problem": problem.spec.name, "epsilon": epsilon,
               "transitions": len(detail["transitions"]), "loss": losses,
               "accepted": detail["summary"]["accepted"], "seconds": detail["summary"]["wall_seconds"],
               "reward": detail["summary"]["reward_sum"], "actions": detail["summary"]["action_indices"]}
        training.append(row)
        write_json(output / "train" / f"episode_{episode:04d}.json", detail)
        write_json(output / "training_history.json", training)
        print(json.dumps({"event": "train", **row}, ensure_ascii=False), flush=True)
        if episode % args.eval_every == 0 or episode == args.episodes:
            rows = evaluate(problems["validation"], config, agent, best_fixed, ACTIONS,
                            output / "validation" / f"episode_{episode:04d}", "validation")
            score = evaluation_score(rows, config.seconds_budget)
            validation_history.append({"episode": episode, "score": score, "aggregate": aggregate(rows)})
            write_json(output / "validation_history.json", validation_history)
            if score > best_score:
                best_score, best_epoch = score, episode
                agent.save(output / "controller.pt", metadata={"features": list(FEATURE_NAMES),
                           "config": asdict(config), "best_epoch": best_epoch,
                           "validation_score": score, "scope": args.scope, "seed": args.seed})
            agent.save(output / "last_checkpoint.pt", metadata={"episode": episode,
                       "features": list(FEATURE_NAMES), "config": asdict(config),
                       "training_seconds_this_invocation": training_seconds,
                       "training_time_scope": "resume invocation" if state else "full training"})
    selected_agent = QAgent.load(output / "controller.pt")
    evaluation_rows = []
    for split in ("test", "ood"):
        evaluation_rows += evaluate(problems[split], config, selected_agent, best_fixed, ACTIONS,
                                    output / "evaluation" / split, split, repeats=args.eval_repeats,
                                    modes=("fixed", "heuristic", "neural"), order_seed=args.seed + 3)
    write_json(output / "evaluation_summary.json", evaluation_rows)
    fields = ["split", "problem", "mode", "repeat", "accepted", "status", "wall_seconds", "iterations",
              "cycles", "eta", "relative_true_residual", "range_defect", "forward_error",
              "peak_basis_core_bytes", "peak_round_rank", "feature_seconds", "inference_seconds"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in evaluation_rows:
            writer.writerow({k: row["problem"]["name"] if k == "problem" else row.get(k) for k in fields})
    manifest.update({"stage": "complete", "total_seconds": time.perf_counter() - started,
                     "training_seconds": None if state else training_seconds, "best_epoch": best_epoch,
                     "best_fixed_delta": ACTIONS[best_fixed], "total_transitions": sum(r["transitions"] for r in training)})
    if state:
        manifest["resumed_training_seconds"] = training_seconds
        manifest["total_seconds_scope"] = "Current resume invocation, not end-to-end original training."
    write_json(output / "manifest.json", manifest)
    make_report(output, manifest, training, validation_grid, evaluation_rows, ACTIONS[best_fixed], best_epoch)
    print(json.dumps({"event": "complete", "output": str(output), "seconds": manifest["total_seconds"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
