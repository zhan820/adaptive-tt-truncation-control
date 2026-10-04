"""Development-only cost checks on fresh paper-family RHSs, never test data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import torch

from rl_tt_dgmres_v1.experiment import write_json
from rl_tt_dgmres_v3.runner import ACTIONS, EpisodeConfig, run_episode
from rl_tt_dgmres_neumann.problems import ProblemSpec, build_problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--dims", default="4,5,6")
    parser.add_argument("--actions", default="3,4,6")
    parser.add_argument("--rank", type=int, default=96)
    parser.add_argument("--recovery-rank", type=int, default=256)
    parser.add_argument("--diagnostic-rank", type=int, default=512)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026100191)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError("Pilot logs are never overwritten.")
    dims = [int(x) for x in args.dims.split(",")]
    actions = [int(x) for x in args.actions.split(",")]
    if args.seed == 20260906 or any(not 0 <= a < len(ACTIONS) for a in actions):
        raise ValueError("Do not pilot on the paper seed or invalid actions.")
    config = EpisodeConfig(tolerance=1e-7, seconds_budget=args.seconds,
                           max_iterations=300, max_tt_rank=args.rank,
                           recovery_max_tt_rank=args.recovery_rank,
                           diagnostic_max_rank=args.diagnostic_rank)
    config.validate()
    if args.threads < 1:
        raise ValueError("Thread count must be positive.")
    torch.set_num_threads(args.threads)
    specs = [ProblemSpec(name=f"pilot_n12_d{d}", n=12, dims=d, seed=args.seed + 1009*d)
             for d in dims]
    out.mkdir(parents=True)
    write_json(out / "protocol.json", {
        "role": "development_only_not_heldout", "config": asdict(config),
        "specs": [asdict(s) for s in specs], "actions": actions, "threads": args.threads,
        "paper_seed_excluded": True, "no_training_or_test_selection_claim": True,
    })
    tiny = build_problem(ProblemSpec(name="pilot_warmup", n=4, dims=2, seed=args.seed))
    run_episode(tiny, config, lambda _: 3, compute_policy_features=False)
    jobs = [(i, action) for i in range(len(specs)) for action in actions]
    random.Random(args.seed).shuffle(jobs)
    summaries = []
    for done, (i, action) in enumerate(jobs, 1):
        problem = build_problem(specs[i])
        result = run_episode(problem, config, lambda _, a=action: a,
                             evaluate_reference=True, compute_policy_features=False)
        serializable = {key: value for key, value in result.items()
                        if key not in {"solution", "solver_history"}}
        write_json(out / f"{specs[i].name}_a{action}.json", serializable)
        summary = result["summary"]
        summaries.append({"action": action, **summary})
        write_json(out / "summary.json", summaries)
        print(json.dumps({"done": done, "total": len(jobs), "dims": specs[i].dims,
                          "action": action, "success": summary["accepted"],
                          "status": summary["status"], "eta": summary["eta"],
                          "seconds": summary["wall_seconds"], "iterations": summary["iterations"],
                          "peak_rank": summary["peak_round_rank"]}), flush=True)


if __name__ == "__main__":
    main()
