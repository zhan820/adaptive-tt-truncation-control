from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch import nn

from .backend.paper_tt_dgmres import _stable_tt_norm
from .backend.tt_dgmres_seventh_fast_restart_sweep import tt_dgmres_sparse_diagnostics
from .controller import Action, GLOBAL_FEATURE_NAMES, SolverController, candidate_actions
from .features import TT_FEATURE_NAMES
from .model import (JointControllerNet, load_controller_checkpoint,
                    save_controller_checkpoint, tensorize_histories)
from .problems import ProblemSpec, build_problem, default_splits, final_audit, make_config
from .rounding import BudgetRounder, BudgetRoundError
from .solver import tt_dgmres_adaptive


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def solve(problem, *, mode="heuristic", model=None, first_action=None,
          fixed_action=None, initial_solution=None, prefix=None, first_frame=None,
          snapshots=False, max_iterations=180, tolerance=1e-6, base_eps=1e-8,
          baseline_restart=8, progress_guard=True):
    config = make_config(problem, restart=baseline_restart, max_iterations=max_iterations,
                         tolerance=tolerance, round_eps=base_eps)
    rounder = BudgetRounder()
    controller = None
    started = time.perf_counter()
    result = None
    error = None
    audit = {}
    try:
        if mode == "baseline":
            result = tt_dgmres_sparse_diagnostics(
                operator=problem.operator, rhs=problem.rhs, config=config,
                diagnostic_interval=max_iterations, diagnostic_max_tt_rank=256,
                solution_projector=problem.projector,
            )
        else:
            controller = SolverController(
                operator=problem.operator, rhs=problem.rhs, operator_scale=problem.norm_lower,
                rounder=rounder, base_eps=base_eps, mode=mode, model=model,
                first_action=first_action, fixed_action=fixed_action,
                prefix=prefix, first_frame=first_frame, snapshots=snapshots,
                progress_guard=progress_guard,
            )
            result = tt_dgmres_adaptive(
                operator=problem.operator, rhs=problem.rhs, config=config,
                diagnostic_interval=max_iterations, diagnostic_max_tt_rank=256,
                diagnostic_residual_round_tolerance=1e-12,
                diagnostic_transformed_round_tolerance=1e-12,
                solution_projector=problem.projector, cycle_controller=controller.on_cycle_start,
                cycle_observer=controller.on_cycle_end, algorithm_rounder=rounder,
                initial_solution=initial_solution,
            )
        audit = final_audit(problem, result.solution, include_reference=False)
    except BudgetRoundError as exc:
        # Infeasible rank/tolerance combinations remain in the training data.
        error = {"kind": type(exc).__name__, "message": str(exc), "record": exc.record}
    wall_seconds = time.perf_counter() - started
    accepted = bool(result is not None and result.converged and
                    audit["relative_true_residual"] < tolerance and
                    audit["solution_range_defect"] < 1e-9)
    # Reference-error evaluation is excluded from the online timing/learning input.
    if result is not None:
        audit["relative_solution_error_evaluation_only"] = _stable_tt_norm(
            result.solution - problem.reference_solution) / _stable_tt_norm(problem.reference_solution)
    summary = {
        "problem": asdict(problem.spec), "mode": mode,
        "accepted": accepted, "solver_converged": bool(result and result.converged),
        "wall_seconds": wall_seconds, "iterations": result.iterations if result else None,
        "cycles": result.cycles if result else None,
        "status": result.status if result else "rounding_rejected",
        "target_relative_true_residual": tolerance,
        "base_round_eps": base_eps, "baseline_restart": baseline_restart,
        "feature_seconds": controller.feature_seconds if controller else 0.0,
        "inference_seconds": controller.inference_seconds if controller else 0.0,
        "progress_guard_enabled": bool(progress_guard and mode == "neural"),
        "progress_guard_activations": controller.guard_activations if controller else 0,
        "fallback_cycles": sum(c["decision_source"] == "measured_progress_fallback"
                               for c in controller.records) if controller else 0,
        "rounding": rounder.cumulative_stats(), "audit": audit, "error": error,
        "local_budget_kind": "empirical_unweighted_relative_round_errors",
        "forward_error_certified": False,
    }
    detail = {"summary": summary, "cycles": controller.records if controller else [],
              "round_calls": rounder.logs, "history": result.history if result else {}}
    return summary, detail, controller, result


def collect_rows(specs, split, args, output):
    rows = []
    actions = candidate_actions()
    for spec in specs:
        problem = build_problem(spec)
        source_summary, _, source, _ = solve(
            problem, mode="fixed", fixed_action=Action(8, 0.1), snapshots=True,
            max_iterations=args.max_iterations, tolerance=args.tolerance, base_eps=args.base_eps,
        )
        if not source or not source.saved_states:
            raise RuntimeError(f"No source states for {spec.name}: {source_summary}")
        for state in source.saved_states[:args.states_per_problem]:
            state_id = f"{spec.name}:cycle{state['source_cycle']}"
            for action_index, action in enumerate(actions):
                summary, detail, branch, _ = solve(
                    problem, mode="teacher", first_action=action,
                    fixed_action=Action(12, 0.1), initial_solution=state["solution"],
                    prefix=state["history"][:-1], first_frame=state["history"][-1],
                    max_iterations=max(1, args.max_iterations - state["source_iteration"]),
                    tolerance=args.tolerance, base_eps=args.base_eps,
                )
                first = branch.records[0] if branch and branch.records else {}
                # Cost of a first action followed by a FIXED continuation policy,
                # not an optimal Q oracle. Failure receives an explicit penalty.
                cost = summary["wall_seconds"] + (0.0 if summary["accepted"] else args.failure_penalty)
                row = {
                    "split": split, "problem": asdict(spec), "state_id": state_id,
                    "history": state["history"], "action_index": action_index,
                    "action": asdict(action), "accepted": summary["accepted"],
                    "targets": {
                        "log_cost": math.log(max(cost, 1e-8)),
                        "cycle_log_seconds": math.log(max(first.get("wall_seconds", summary["wall_seconds"]), 1e-8)),
                        "log_reduction": first.get("log_reduction", 0.0),
                        "log_rank": math.log(max(first.get("solution_rank", 1), 1)),
                    },
                    "rollout_summary": summary,
                }
                rows.append(clean_json(row))
                write_json(output / "rollouts" / f"{spec.name}_s{state['source_cycle']}_a{action_index}.json",
                           clean_json(detail))
            print(json.dumps({"event": "collected_state", "split": split, "state": state_id,
                              "rows": len(rows)}, ensure_ascii=False), flush=True)
        write_json(output / f"{split}_data.json", rows)
    return rows


def fit_model(train_rows, validation_rows, args, output):
    torch.manual_seed(args.seed)
    model = JointControllerNet().float()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    encoded_actions = [a.encoded() for a in candidate_actions()]

    def batch(rows):
        inputs = tensorize_histories([r["history"] for r in rows], encoded_actions)
        action_index = torch.tensor([r["action_index"] for r in rows], dtype=torch.long)
        labels = {name: torch.tensor([r["targets"][name] for r in rows], dtype=torch.float32)
                  for name in ("log_cost", "cycle_log_seconds", "log_reduction", "log_rank")}
        return inputs, action_index, labels

    train_batch, val_batch = batch(train_rows), batch(validation_rows)
    criterion = nn.SmoothL1Loss()

    def loss_of(data):
        inputs, indices, targets = data
        prediction = model(**inputs)
        values = {k: criterion(prediction[k].gather(1, indices[:, None]).squeeze(1), v)
                  for k, v in targets.items()}
        total = values["log_cost"] + 0.2 * values["cycle_log_seconds"] + \
            0.1 * values["log_reduction"] + 0.1 * values["log_rank"]
        return total, values

    best, best_epoch, best_state = float("inf"), 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs + 1):
        model.train()
        loss, _ = loss_of(train_batch)
        if epoch:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss, _ = loss_of(train_batch)
            val_loss, val_parts = loss_of(val_batch)
        score = float(val_parts["log_cost"])
        if score < best:
            best, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": float(train_loss),
                        "validation_loss": float(val_loss), "validation_cost_loss": score})
    model.load_state_dict(best_state)
    model.eval()
    metadata = {
        "seed": args.seed, "best_epoch": best_epoch, "epochs": args.epochs,
        "train_rows": len(train_rows), "validation_rows": len(validation_rows),
        "training_seconds": time.perf_counter() - started,
        "training_objective": "log_cold_resumed_branch_time_with_cached_first_features_and_fixed_continuation",
        "cost_target_limitation": "resumed branch includes reinitialization and cached first features; surrogate for deployment cost-to-go",
        "is_optimal_Q_oracle": False, "failure_penalty_seconds": args.failure_penalty,
        "actions": [asdict(a) for a in candidate_actions()],
        "action_encoding": ["restart/32", "log10(alpha)"],
        "token_feature_names": list(TT_FEATURE_NAMES), "global_feature_names": GLOBAL_FEATURE_NAMES,
        "base_eps": args.base_eps, "tolerance": args.tolerance,
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    save_controller_checkpoint(output / "controller.pt", model, metadata=metadata)
    write_json(output / "training_history.json", history)
    write_json(output / "training_metadata.json", metadata)
    reloaded, _ = load_controller_checkpoint(output / "controller.pt")
    reloaded.eval()
    with torch.no_grad():
        original = model(**val_batch[0])["log_cost"]
        restored = reloaded(**val_batch[0])["log_cost"]
    if not torch.equal(original, restored):
        raise AssertionError("Checkpoint did not reproduce predictions.")
    print(json.dumps({"event": "trained", **metadata}, ensure_ascii=False), flush=True)
    return reloaded, metadata


def plot_results(details, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(d["summary"]["problem"]["name"] for d in details))
    fig, axes = plt.subplots(2, len(names), figsize=(6 * len(names), 7), squeeze=False)
    for column, name in enumerate(names):
        for detail in details:
            s = detail["summary"]
            if s["problem"]["name"] != name:
                continue
            h = detail["history"]
            if h:
                rhs_scale = build_problem(ProblemSpec(**s["problem"])).rhs_norm
                axes[0, column].semilogy(h["iteration"], np.array(h["true_residual_norm"]) / rhs_scale,
                                         marker=".", label=s["label"])
            if s["mode"] == "neural":
                cycles = detail["cycles"]
                axes[1, column].step([c["cycle"] for c in cycles], [c["restart"] for c in cycles],
                                     where="mid", label="executed m")
                axes[1, column].step([c["cycle"] for c in cycles], [c["proposed_restart"] for c in cycles],
                                     where="mid", ls="--", color="gray", label="network proposal")
                fallback = [c for c in cycles if c["decision_source"] == "measured_progress_fallback"]
                if fallback:
                    axes[1, column].scatter([c["cycle"] for c in fallback], [c["restart"] for c in fallback],
                                            marker="x", color="red", label="progress fallback")
                last_action = None
                for c in cycles:
                    if (c["restart"], c["alpha"]) != last_action:
                        axes[1, column].annotate(f"alpha={c['alpha']:g}", (c["cycle"], c["restart"]),
                                                 fontsize=8, xytext=(0, 8), textcoords="offset points")
                    last_action = c["restart"], c["alpha"]
        axes[0, column].axhline(details[0]["summary"]["target_relative_true_residual"], color="gray", ls=":")
        axes[0, column].set(title=name, xlabel="Arnoldi steps", ylabel="Recomputed relative residual")
        axes[0, column].legend(fontsize=7)
        axes[1, column].set(xlabel="Restart cycle", ylabel="Restart length", ylim=(2, 15))
        axes[1, column].legend(fontsize=7)
        for ax in axes[:, column]:
            ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "smoke_results.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU end-to-end learned TT-DGMRES prototype")
    parser.add_argument("--output", type=Path, default=Path("results/adaptive_tt_dgmres_smoke"))
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--states-per-problem", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=180)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--base-eps", type=float, default=1e-8)
    parser.add_argument("--failure-penalty", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--disable-progress-guard", action="store_true",
                        help="Ablate the measured-progress fallback; an undertrained policy may fail.")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.states_per_problem not in (1, 2) or args.max_iterations < 12:
        parser.error("Require epochs >=1, states-per-problem in {1,2}, max-iterations >=12.")
    if not 0 < args.tolerance < 1 or not 0 < args.base_eps < 1 or args.failure_penalty <= 0:
        parser.error("Require positive tolerance/base-eps <1 and failure penalty >0.")
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits = default_splits()
    manifest = {
        "purpose": "pipeline_smoke_test_not_publication_benchmark",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "tntorch": importlib.metadata.version("tntorch"), "device": "cpu",
        "splits": {k: [asdict(s) for s in v] for k, v in splits.items()},
        "source_sha256": {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sorted(Path(__file__).parent.rglob("*.py"))},
        "limitations": ["one index-one Neumann operator family", "single timing per case",
                        "empirical local budget, no residual-gap certificate", "fixed-continuation rollout pretraining"],
    }
    if args.reuse_data:
        old_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        for key in ("states_per_problem", "max_iterations", "tolerance", "base_eps", "failure_penalty", "seed", "threads"):
            if old_manifest["config"][key] != manifest["config"][key]:
                raise ValueError(f"Cannot reuse data with changed {key}.")
        if old_manifest["splits"] != manifest["splits"]:
            raise ValueError("Cannot reuse data with changed splits.")
        train = json.loads((output / "train_data.json").read_text(encoding="utf-8"))
        validation = json.loads((output / "validation_data.json").read_text(encoding="utf-8"))
        manifest["data_source_sha256"] = old_manifest.get("data_source_sha256", old_manifest["source_sha256"])
    else:
        write_json(output / "manifest.json", manifest)
        train = collect_rows(splits["train"], "train", args, output)
        validation = collect_rows(splits["validation"], "validation", args, output)
    write_json(output / "manifest.json", manifest)
    model, training = fit_model(train, validation, args, output)

    # Validation determines the single fixed baseline before accessing test cases.
    tuning = []
    validation_problem = build_problem(splits["validation"][0])
    for restart in (4, 8, 12):
        for eps in (1e-8, 1e-10):
            summary, _, _, _ = solve(validation_problem, mode="baseline", baseline_restart=restart,
                                     base_eps=eps, tolerance=args.tolerance, max_iterations=args.max_iterations)
            tuning.append(summary)
    successful = [s for s in tuning if s["accepted"]]
    if not successful:
        raise RuntimeError("All fixed baselines failed on validation.")
    best_fixed = min(successful, key=lambda s: s["wall_seconds"])
    write_json(output / "baseline_validation.json", clean_json(tuning))

    summaries, details = [], []
    for spec in splits["test"]:
        problem = build_problem(spec)
        for mode in ("baseline", "heuristic", "neural"):
            summary, detail, _, _ = solve(
                problem, mode=mode, model=model, tolerance=args.tolerance,
                max_iterations=args.max_iterations,
                baseline_restart=best_fixed["baseline_restart"],
                base_eps=best_fixed["base_round_eps"] if mode == "baseline" else args.base_eps,
                progress_guard=not args.disable_progress_guard,
            )
            summary["label"] = (f"fixed m={best_fixed['baseline_restart']}" if mode == "baseline" else
                                "neural+guard" if mode == "neural" and not args.disable_progress_guard else mode)
            summaries.append(summary)
            details.append(detail)
            write_json(output / "evaluation" / f"{spec.name}_{mode}.json", clean_json(detail))
            print(json.dumps({"event": "evaluated", "case": spec.name, "mode": mode,
                              "accepted": summary["accepted"], "seconds": summary["wall_seconds"],
                              "iterations": summary["iterations"]}, ensure_ascii=False), flush=True)
    write_json(output / "summary.json", clean_json({"training": training, "runs": summaries}))
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "mode", "accepted", "seconds", "iterations",
                                                   "relative_residual", "forward_error", "range_defect", "fallback_cycles"])
        writer.writeheader()
        for s in summaries:
            writer.writerow({"case": s["problem"]["name"], "mode": s["mode"], "accepted": s["accepted"],
                             "seconds": s["wall_seconds"], "iterations": s["iterations"],
                             "relative_residual": s["audit"].get("relative_true_residual"),
                             "forward_error": s["audit"].get("relative_solution_error_evaluation_only"),
                             "range_defect": s["audit"].get("solution_range_defect"),
                             "fallback_cycles": s["fallback_cycles"]})
    plot_results(details, output)
    lines = ["# 自适应 TT-DGMRES 首轮运行", "", "本批次用于验证训练与求解流程，不能作为加速或跨问题族泛化的论文结论。", "",
             f"模型参数量：{training['parameter_count']}；训练样本：{len(train)}；验证样本：{len(validation)}。", "",
             "| 算例 | 策略 | 验收 | 时间/秒 | 迭代步数 | 重算相对残差 | 参考解相对误差 | 回退周期数 |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        a = s["audit"]
        lines.append(f"| {s['problem']['name']} | {s['label']} | {s['accepted']} | {s['wall_seconds']:.3f} | {s['iterations']} | "
                     f"{a.get('relative_true_residual', float('nan')):.3e} | {a.get('relative_solution_error_evaluation_only', float('nan')):.3e} | {s['fallback_cycles']} |")
    lines += ["", "验收：独立重算相对真实残差低于目标，且已知左零空间的值域偏离低于 1e-9。参考解误差仅作评价。",
              "", "局部舍入预算是经验性的未加权相对误差和，尚未证明变换残差误差界；不是 Drazin 前向误差认证。",
              "", "模型学习的是首个动作后接固定后续策略的求解耗时；这是预训练，尚未进行完整策略迭代。",
              "", "学习策略带有实测进展回退：近期残差下降预计不足以满足剩余迭代预算时，使用较长重启和较小局部预算。日志分开保留网络提案和实际动作；该规则不构成全局收敛证明。",
              "", "训练、验证、测试使用不同算子实例；全部仍属于指标为一的 Neumann 族。耗时包括在线特征提取、推理和舍入审计，排除参考解评价与训练。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not all(s["accepted"] for s in summaries if s["mode"] == "neural"):
        raise RuntimeError("At least one learned-policy smoke evaluation failed; inspect summary.json.")
    print(json.dumps({"event": "complete", "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
