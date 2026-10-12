"""Inspect raw network decisions on fresh, exactly solvable component positions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from alphaboxes.checkpoint import load_evaluator
from alphaboxes.curriculum import sample_endgame
from alphaboxes.endgame import DegreeTwoSolver
from alphaboxes.search import CachedEvaluator
from alphaboxes.teaching import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude-dataset", type=Path)
    parser.add_argument("--boards", type=int, nargs="+", default=[4, 5, 10, 12, 14])
    parser.add_argument("--positions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=41171)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.positions < 1 or any(size < 3 for size in args.boards):
        parser.error("Use positive position counts and boards of at least 3×3.")
    source_paths = [Path(__file__)] + [
        Path("src/alphaboxes") / name
        for name in (
            "curriculum.py",
            "endgame.py",
            "components.py",
            "numpy_network.py",
            "graph.py",
            "teaching.py",
        )
    ]
    sources = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    excluded = (
        {CachedEvaluator._key(state) for state in load_dataset(args.exclude_dataset).states}
        if args.exclude_dataset
        else set()
    )
    evaluators = [load_evaluator(path) for path in args.checkpoints]
    results, suite = [], []
    solver = DegreeTwoSolver()
    for size in args.boards:
        rng = np.random.default_rng(args.seed + size)
        positions = {}
        for _ in range(args.positions * 100):
            if len(positions) == args.positions:
                break
            state = sample_endgame((size, size), rng)
            key = CachedEvaluator._key(state)
            if key not in excluded:
                positions[key] = state
        if len(positions) != args.positions:
            raise ValueError("Insufficient fresh distinct positions for this board.")
        observations = [[] for _ in evaluators]
        for state in positions.values():
            margin, actions = solver.solve(state)
            expected = int(np.sign(margin))
            suite.append([size, state.edges, state.owners, state.player, margin])
            for index, (evaluator, _) in enumerate(evaluators):
                policy, value = evaluator(state)
                action = int(policy.argmax())
                achieved = solver.action_value(state, action)
                predicted = 0 if abs(value) < 0.1 else int(np.sign(value))
                observations[index].append(
                    {
                        "value_error": abs(value - expected),
                        "outcome_correct": predicted == expected,
                        "best_margin": action in actions,
                        "optimal_policy_mass": float(policy[list(actions)].sum()),
                        "box_regret": (margin - achieved) * state.board.num_boxes,
                    }
                )
        for (_, metadata), rows in zip(evaluators, observations, strict=True):
            results.append(
                {
                    "checkpoint_sha256": metadata["checkpoint_sha256"],
                    "board": [size, size],
                    "positions": len(rows),
                    **{key: float(np.mean([row[key] for row in rows])) for key in rows[0]},
                }
            )
    report = {
        "seed": args.seed,
        "classification_draw_threshold": 0.1,
        "excluded_dataset_overlap": 0 if args.exclude_dataset else None,
        "excluded_dataset_sha256": hashlib.sha256(args.exclude_dataset.read_bytes()).hexdigest()
        if args.exclude_dataset
        else None,
        "suite_sha256": hashlib.sha256(json.dumps(suite).encode()).hexdigest(),
        "source_sha256": sources,
        "scope": "Raw network on closed chain/loop positions; no full-game strength claim",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
