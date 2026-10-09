"""Compare a trained policy with an identically sized random initialization."""

import argparse
from pathlib import Path

import torch

from alphaboxes.checkpoint import load_agent, save_agent
from alphaboxes.evaluation import evaluate
from alphaboxes.network import module_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/ablation"))
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    model, metadata = load_agent(args.checkpoint)
    torch.manual_seed(metadata["seed"])
    untrained = module_spec(**model.model_config).build()
    initial_path = args.output / "untrained.pt"
    save_agent(
        initial_path,
        untrained.state_dict(),
        model.model_config["width"],
        model.model_config["depth"],
        {"seed": metadata["seed"], "iteration": 0},
    )
    for label, checkpoint in [("untrained", initial_path), ("trained", args.checkpoint)]:
        evaluate(
            checkpoint,
            args.output / f"{label}-policy.json",
            [(2, 2), (3, 3), (4, 4)],
            games=args.games,
            seed=args.seed,
            policy_only=True,
        )


if __name__ == "__main__":
    main()
