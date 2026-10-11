"""Measure uncached NumPy evaluation on identical legal board positions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from alphaboxes.checkpoint import load_evaluator
from alphaboxes.game import State


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boards", type=int, nargs="+", default=[4, 5, 10])
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3604)
    args = parser.parse_args()
    if args.repeats < 1 or any(not 2 <= size <= 12 for size in args.boards):
        parser.error("Use positive repeats and boards of 2..12 boxes per side.")
    evaluators = [load_evaluator(path)[0] for path in args.checkpoints]
    results = []
    for size in args.boards:
        state = State.new(size, size)
        positions = [state]
        rng = np.random.default_rng(args.seed + size)
        targets = {state.board.num_edges // 2, max(13, state.board.num_edges // 5)}
        while len(state.legal_actions) > min(targets):
            state = state.play(int(rng.choice(state.legal_actions)))
            if len(state.legal_actions) in targets:
                positions.append(state)
        for state in positions:
            for path, evaluator in zip(args.checkpoints, evaluators, strict=True):
                timings = []
                for _ in range(args.repeats + 5):
                    evaluator.clear_cache()
                    started = perf_counter()
                    evaluator(state)
                    timings.append(perf_counter() - started)
                results.append(
                    {
                        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "width": evaluator.network.width,
                        "depth": evaluator.network.depth,
                        "parameters": sum(
                            value.size for value in evaluator.network.weights.values()
                        ),
                        "board": [size, size],
                        "remaining_edges": len(state.legal_actions),
                        "seconds": timings[5:],
                        "median_seconds": float(np.median(timings[5:])),
                    }
                )
    report = {
        "seed": args.seed,
        "numpy": np.__version__,
        "threads": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "repeats": args.repeats,
        "cache": "Cleared before every call; five warmup calls excluded",
        "timing_includes": "Graph encoding, NumPy network and legal-policy normalization",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
