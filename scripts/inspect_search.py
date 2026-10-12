"""Inspect how first-play settings distribute a fixed number of simulations."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import numpy as np

from alphaboxes.checkpoint import load_evaluator
from alphaboxes.game import State
from alphaboxes.search import MCTS, SearchConfig


def tree_metrics(root):
    """Recover simulation endpoints from fresh-tree parent/child visit counts."""
    pending = [(root, 0)]
    endpoints = []
    visited_nodes = 0
    while pending:
        node, depth = pending.pop()
        if not node.visits:
            continue
        visited_nodes += 1
        children = node.children or {}
        stops = node.visits - sum(child.visits for child in children.values())
        endpoints.extend([depth] * stops)
        pending.extend((child, depth + 1) for child in children.values())
    assert len(endpoints) == root.visits
    return {
        "visited_nodes": visited_nodes,
        "visited_root_actions": sum(child.visits > 0 for child in root.children.values()),
        "mean_endpoint_depth": float(np.mean(endpoints)),
        "median_endpoint_depth": float(np.median(endpoints)),
        "max_endpoint_depth": max(endpoints),
        "endpoint_depth_counts": {
            str(depth): endpoints.count(depth) for depth in sorted(set(endpoints))
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/larger.npz"))
    parser.add_argument("--boards", type=int, nargs="+", default=[4, 5, 10])
    parser.add_argument("--simulations", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=4451)
    parser.add_argument("--modes", nargs="+", default=["constant"])
    parser.add_argument("--parents", nargs="+", choices=("network", "mean"), default=["network"])
    parser.add_argument("--reductions", type=float, nargs="+", default=[0.33])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.simulations < 1 or any(not 3 <= size <= 12 for size in args.boards):
        parser.error("Use positive simulations and board sizes of 3..12.")
    evaluator, _ = load_evaluator(args.checkpoint)
    records = []
    for size in args.boards:
        rng = np.random.default_rng(args.seed + size)
        state, opening = State.new(size, size), []
        for _ in range(6):
            action = int(rng.choice(state.legal_actions))
            opening.append(action)
            state = state.play(action)
        settings = [(0.0, "constant", "network")] + [
            (reduction, mode, parent)
            for reduction in args.reductions
            for mode in args.modes
            for parent in args.parents
        ]
        for reduction, mode, parent in settings:
            evaluator.clear_cache()
            # Each planner is fresh. Reuse merely retains its tree for inspection;
            # no visits are inherited and no second policy call is performed.
            config = SearchConfig(
                simulations=args.simulations,
                fpu_reduction=reduction,
                fpu_mode=mode,
                fpu_parent=parent,
                exact_threshold=12,
                reuse_tree=True,
            )
            planner = MCTS(evaluator, config)
            started = monotonic()
            policy, value = planner.policy(state)
            elapsed = monotonic() - started
            assert planner.inherited_visits == 0
            root = planner._roots[0]
            metrics = tree_metrics(root)
            records.append(
                {
                    "board": [size, size],
                    "opening": opening,
                    "config": asdict(config),
                    "seconds": elapsed,
                    "root_neural_value": root.evaluation,
                    "search_value": value,
                    "action": int(policy.argmax()),
                    "completed_simulations": planner.completed_simulations,
                    "legal_actions": len(state.legal_actions),
                    **metrics,
                }
            )
            print(json.dumps(records[-1]), flush=True)
    report = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": args.seed,
        "records": records,
        "scope": "One shared random opening per board; illustrative search diagnostics",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
