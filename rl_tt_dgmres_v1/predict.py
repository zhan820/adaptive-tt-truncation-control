from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .experiment import write_json
from .learning import QAgent
from .problems import ProblemSpec, build_problem
from .runner import FEATURE_NAMES, EpisodeConfig, run_episode


def solve_with_checkpoint(problem, checkpoint, *, config=None):
    agent = QAgent.load(checkpoint)
    if agent.metadata.get("features") != list(FEATURE_NAMES):
        raise ValueError("Checkpoint feature schema does not match this implementation.")
    trained_config = EpisodeConfig(**agent.metadata["config"])
    if config is None:
        config = trained_config
    elif config != trained_config:
        raise ValueError("V1 inference requires the trained numerical configuration; retrain for changes.")
    return run_episode(problem, config, lambda observation: agent.select_action(observation, 0.0),
                       agent.actions, evaluate_reference=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--dims", type=int, default=3)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--anisotropy", type=float, default=2.0)
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=77891)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error("Output already exists; choose another path.")
    torch.set_num_threads(1)
    spec = ProblemSpec("inference", args.n, args.dims, args.seed, args.anisotropy,
                       args.rank, args.contamination)
    detail = solve_with_checkpoint(build_problem(spec), args.checkpoint)
    write_json(args.output, detail)
    print(detail["summary"])


if __name__ == "__main__":
    main()
