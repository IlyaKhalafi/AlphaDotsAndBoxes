"""Create a fixed oracle suite or compare a checkpoint's search on that suite."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from alphaboxes.diagnostics import benchmark_suite, generate_suite  # noqa: E402
from alphaboxes.search import SearchConfig  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3100)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--simulations", type=int, default=128)
    args = parser.parse_args()
    if args.checkpoint is None:
        if args.suite.exists():
            parser.error("Suite already exists; choose a separate path for new positions.")
        suite = generate_suite(args.seed, args.repetitions)
        args.suite.parent.mkdir(parents=True, exist_ok=True)
        args.suite.write_text(json.dumps(suite, indent=2) + "\n")
        print(json.dumps({"positions": len(suite["positions"]), "seed": args.seed}))
    else:
        if args.output is None:
            parser.error("Checkpoint comparisons require --output.")
        result = benchmark_suite(
            args.checkpoint, args.suite, args.output, SearchConfig(simulations=args.simulations)
        )
        print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
