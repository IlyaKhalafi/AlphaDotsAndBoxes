"""Measure supervised NumPy decisions across board sizes and game phases."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import monotonic

import numpy as np

from alphaboxes.checkpoint import load_evaluator
from alphaboxes.game import State
from alphaboxes.search import SearchConfig
from alphaboxes.timed_search import BoundedSearch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/larger.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boards", type=int, nargs="+", default=[4, 5, 10])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=1.75)
    parser.add_argument("--fpu-reduction", type=float, default=0.33)
    parser.add_argument("--chain-pruning", action="store_true")
    args = parser.parse_args()
    evaluator, _ = load_evaluator(args.checkpoint)
    config = SearchConfig(
        simulations=8192,
        time_limit=args.seconds,
        exact_threshold=12,
        fpu_reduction=args.fpu_reduction,
        chain_pruning=args.chain_pruning,
    )
    records = []
    with BoundedSearch(evaluator, config, timeout=1.9) as search:
        for size in args.boards:
            state = State.new(size, size)
            positions = {"opening": state}
            rng = np.random.default_rng(4251 + size)
            while len(state.legal_actions) > 12:
                state = state.play(int(rng.choice(state.legal_actions)))
                if len(state.legal_actions) == state.board.num_edges // 2:
                    positions["midgame"] = state
                if len(state.legal_actions) == 13:
                    positions["late"] = state
            positions["exact_root"] = state
            for phase, state in positions.items():
                for repeat in range(args.repeats):
                    search.clear_cache()
                    started = monotonic()
                    policy, value = search.policy(state)
                    elapsed = monotonic() - started
                    action = int(policy.argmax())
                    assert action in state.legal_actions and np.isclose(policy.sum(), 1)
                    records.append(
                        {
                            "board": [size, size],
                            "phase": phase,
                            "repeat": repeat,
                            "seconds": elapsed,
                            "remaining_edges": len(state.legal_actions),
                            "action": action,
                            "value": value,
                            **search.stats,
                        }
                    )
                    print(f"{size}x{size} {phase}: {elapsed:.4f}s", flush=True)
    report = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "internal_seconds": args.seconds,
        "response_deadline": 1.9,
        "fpu_reduction": args.fpu_reduction,
        "chain_pruning": args.chain_pruning,
        "threads": {
            key: os.environ.get(key)
            for key in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "records": records,
        "max_seconds": max(row["seconds"] for row in records),
        "fallbacks": sum(row["timed_out"] for row in records),
        "passed": all(row["seconds"] < 2 for row in records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("Decision latency exceeded two seconds.")


if __name__ == "__main__":
    main()
