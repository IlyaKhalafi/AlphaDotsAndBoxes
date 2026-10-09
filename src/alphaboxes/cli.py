"""Command-line entry points; heavy training imports stay out of help/startup."""

import argparse
from dataclasses import replace
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train and play Alpha Dots & Boxes")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Run RLlib AlphaZero self-play")
    training.add_argument("--config", type=Path, default=Path("configs/bootstrap.json"))
    training.add_argument("--output", type=Path, default=Path("runs/bootstrap"))
    training.add_argument("--resume", type=Path)
    training.add_argument("--device", choices=["cpu", "cuda"])
    training.add_argument("--iterations", type=int)
    serving = commands.add_parser("serve", help="Open the local game UI")
    serving.add_argument("--checkpoint", type=Path)
    serving.add_argument("--host", default="127.0.0.1")
    serving.add_argument("--port", type=int, default=8000)
    evaluation = commands.add_parser("evaluate", help="Evaluate both seats against baselines")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, default=Path("runs/evaluation.json"))
    evaluation.add_argument("--sizes", default="2x2,3x3,4x4")
    evaluation.add_argument("--games", type=int, default=40)
    evaluation.add_argument("--simulations", type=int, default=128)
    evaluation.add_argument("--exact-threshold", type=int, default=0)
    evaluation.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.command == "train":
        from alphaboxes.training import TrainConfig, train

        config = TrainConfig.from_json(args.config)
        overrides = {
            key: getattr(args, key)
            for key in ("device", "iterations")
            if getattr(args, key) is not None
        }
        train(replace(config, **overrides), args.output, args.resume)
    elif args.command == "serve":
        import uvicorn

        from alphaboxes.web.app import create_app

        uvicorn.run(create_app(args.checkpoint), host=args.host, port=args.port)
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
        )


if __name__ == "__main__":
    main()
