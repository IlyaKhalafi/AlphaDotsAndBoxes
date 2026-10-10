"""Command-line entry points; heavy training imports stay out of help/startup."""

import argparse
import os
from pathlib import Path


def match_arguments(parser, output: str) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(output))
    parser.add_argument("--sizes", default="2x2,3x3,4x4")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--exact-threshold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)


def main():
    parser = argparse.ArgumentParser(description="Train and play Alpha Dots & Boxes")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Run RLlib AlphaZero self-play")
    training.add_argument("--config", type=Path, default=Path("configs/bootstrap.json"))
    training.add_argument("--output", type=Path, default=Path("runs/bootstrap"))
    initialization = training.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument(
        "--initial-checkpoint", type=Path, help="Start from weights on new boards"
    )
    training.add_argument("--device", choices=["cpu", "cuda"])
    training.add_argument("--iterations", type=int)
    training.add_argument(
        "--seconds", type=float, help="Stop after a complete iteration and save progress"
    )
    serving = commands.add_parser("serve", help="Open the local game UI")
    serving.add_argument("--checkpoint", type=Path)
    serving.add_argument("--host", default="127.0.0.1")
    serving.add_argument("--port", type=int, default=8000)
    evaluation = commands.add_parser("evaluate", help="Evaluate both seats against baselines")
    match_arguments(evaluation, "runs/evaluation.json")
    evaluation.add_argument("--policy-only", action="store_true", help="Evaluate without search")
    evaluation.add_argument(
        "--opponents",
        nargs="+",
        choices=["random", "tactical", "exact", "tactical_endgame", "chain_control"],
        help="Select benchmarks; defaults preserve the original evaluation schedule",
    )
    duel = commands.add_parser(
        "duel", help="Compare checkpoints with balanced seats and random openings"
    )
    match_arguments(duel, "runs/duel.json")
    duel.set_defaults(sizes="5x5", seed=3031)
    duel.add_argument("--opponent-checkpoint", type=Path, required=True)
    duel.add_argument("--opening-moves", type=int, default=6)
    export = commands.add_parser(
        "export", help="Convert training weights to NumPy deployment weights"
    )
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    widen = commands.add_parser("widen", help="Expand a trained network without restarting play")
    widen.add_argument("--checkpoint", type=Path, required=True)
    widen.add_argument("--output", type=Path, required=True)
    widen.add_argument("--factor", type=int, default=3)
    widen.add_argument("--noise", type=float, default=0.01)
    widen.add_argument("--seed", type=int, default=45)
    args = parser.parse_args()
    if args.command in {"serve", "evaluate", "duel"}:
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.command == "train":
        from alphaboxes.training import TrainConfig, train

        overrides = {
            key: getattr(args, key)
            for key in ("device", "iterations")
            if getattr(args, key) is not None
        }
        if args.seconds is not None:
            overrides["max_seconds"] = args.seconds
        config = TrainConfig.from_json(args.config, **overrides)
        train(config, args.output, args.resume, args.initial_checkpoint)
    elif args.command == "serve":
        import uvicorn

        from alphaboxes.web.app import create_app

        uvicorn.run(create_app(args.checkpoint), host=args.host, port=args.port)
    elif args.command == "export":
        from alphaboxes.numpy_network import export_agent

        print(export_agent(args.checkpoint, args.output))
    elif args.command == "widen":
        import json

        from alphaboxes.growth import widen_agent

        print(
            json.dumps(
                widen_agent(args.checkpoint, args.output, args.factor, args.noise, args.seed)
            )
        )
    elif args.command == "duel":
        from alphaboxes.evaluation import compare_agents

        sizes = [tuple(map(int, size.split("x"))) for size in args.sizes.split(",")]
        compare_agents(
            args.checkpoint,
            args.opponent_checkpoint,
            args.output,
            sizes,
            args.games,
            args.simulations,
            args.exact_threshold,
            args.seed,
            args.opening_moves,
        )
    else:
        from alphaboxes.evaluation import evaluate

        sizes = [tuple(map(int, size.split("x"))) for size in args.sizes.split(",")]
        evaluate(
            args.checkpoint,
            args.output,
            sizes,
            args.games,
            args.simulations,
            args.exact_threshold,
            args.seed,
            args.policy_only,
            args.opponents,
        )


if __name__ == "__main__":
    main()
