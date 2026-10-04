from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rl_tt_dgmres_v1.experiment import write_json
from rl_tt_dgmres_v1.learning import QAgent
from .problems import ProblemSpec, build_problem
from .runner import ACTIONS, RECOVERY_ACTION, FEATURE_NAMES, EpisodeConfig, run_episode
from dataclasses import asdict


def solve_with_checkpoint(problem, checkpoint, *, config=None):
    agent = QAgent.load(checkpoint)
    if (agent.metadata.get("features") != list(FEATURE_NAMES)
            or agent.metadata.get("joint_actions") != [asdict(a) for a in ACTIONS]
            or agent.metadata.get("recovery_action") != asdict(RECOVERY_ACTION)):
        raise ValueError("Incompatible feature or action schema.")
    trained_config = EpisodeConfig(**agent.metadata["config"])
    if config is not None and config != trained_config:
        raise ValueError("Changed numerical configuration requires separate validation.")
    return run_episode(problem, trained_config, lambda obs: agent.select_action(obs, 0.0))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--dims", type=int, default=4)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=731881)
    parser.add_argument("--family", default="symmetric_weighted_cycle")
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error("Choose a new output path.")
    torch.set_num_threads(1)
    problem = build_problem(ProblemSpec("inference", args.n, args.dims, args.seed,
                                       rhs_rank=args.rank, family=args.family))
    detail = solve_with_checkpoint(problem, args.checkpoint)
    write_json(args.output, detail)
    print(detail["summary"])


if __name__ == "__main__":
    main()
