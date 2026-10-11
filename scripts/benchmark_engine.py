"""Compare timed alpha-beta and MCTS using the same NumPy checkpoint.

Run with one BLAS thread for reproducible CPU resource use. Matches swap seats
on identical random openings; wall-clock budgets make exact traces nondeterministic.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from time import monotonic

import numpy as np

from alphaboxes.checkpoint import load_evaluator
from alphaboxes.engine import AlphaBeta, EngineConfig
from alphaboxes.game import State
from alphaboxes.opponents import solve_remaining
from alphaboxes.search import MCTS, SearchConfig, choose_child


def planner(evaluator, engine, seconds):
    if engine == "alpha_beta":
        return AlphaBeta(evaluator, EngineConfig(time_limit=seconds))
    return MCTS(evaluator, SearchConfig(simulations=8192, time_limit=seconds, exact_threshold=12))


def measure(search, state):
    started = monotonic()
    policy, value = search.policy(state)
    return {
        "seconds": monotonic() - started,
        "action": int(policy.argmax()),
        "value": value,
        "simulations": getattr(search, "completed_simulations", 0),
        "depth": getattr(search, "depth", None),
        "nodes": getattr(search, "nodes", None),
        "cutoffs": getattr(search, "cutoffs", None),
        "table_hits": getattr(search, "table_hits", None),
    }


def latency(evaluator, args):
    results = []
    for size in args.boards:
        state = State.new(size, size)
        states = {len(state.legal_actions): state}
        rng = np.random.default_rng(args.seed)
        targets = {size * (size + 1), max(13, int(state.board.num_edges * 0.18))}
        while not state.terminal:
            state = state.play(int(rng.choice(state.legal_actions)))
            if len(state.legal_actions) in targets:
                states[len(state.legal_actions)] = state
        for remaining, state in states.items():
            for seconds in args.seconds:
                for engine in ("mcts", "alpha_beta"):
                    samples = []
                    for _ in range(args.repeats):
                        evaluator.clear_cache()
                        samples.append(measure(planner(evaluator, engine, seconds), state))
                    results.append(
                        {
                            "board": [size, size],
                            "remaining_edges": remaining,
                            "engine": engine,
                            "time_limit": seconds,
                            "samples": samples,
                        }
                    )
    return results


def matches(evaluators, args):
    games = []
    for size in args.match_boards:
        for index in range(args.games):
            for evaluator in evaluators.values():
                evaluator.clear_cache()
            rng = np.random.default_rng(args.seed + index // 2)
            state = State.new(size, size)
            opening = []
            for _ in range(6):
                action = int(rng.choice(state.legal_actions))
                opening.append(action)
                state = state.play(action)
            alpha_player = index % 2
            moves = []
            while not state.terminal:
                engine = "alpha_beta" if state.player == alpha_player else "mcts"
                step = measure(planner(evaluators[engine], engine, args.match_seconds), state)
                moves.append({"player": state.player, "engine": engine, **step})
                state = state.play(step["action"])
            margin = state.scores[alpha_player] - state.scores[1 - alpha_player]
            games.append(
                {
                    "board": [size, size],
                    "index": index,
                    "alpha_player": alpha_player,
                    "opening": opening,
                    "scores": list(state.scores),
                    "alpha_margin": margin,
                    "moves": moves,
                }
            )
            print(f"{size}x{size} game {index + 1}: alpha-beta margin {margin}", flush=True)
    return games


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/larger.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boards", type=int, nargs="+", default=[4, 5, 10])
    parser.add_argument("--seconds", type=float, nargs="+", default=[0.5, 2])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--match-boards", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--match-seconds", type=float, default=0.1)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3501)
    args = parser.parse_args()
    if args.games < 0 or args.games % 2 or args.repeats < 1:
        parser.error("Use a nonnegative even match count and positive repeats.")
    if not all(2 <= size <= 12 for size in args.boards + args.match_boards):
        parser.error("Benchmark board sizes must be 2..12.")
    for seconds in args.seconds + [args.match_seconds]:
        EngineConfig(time_limit=seconds)
    evaluators = {engine: load_evaluator(args.checkpoint)[0] for engine in ("mcts", "alpha_beta")}
    modules = [
        sys.modules[f"alphaboxes.{name}"]
        for name in (
            "engine",
            "game",
            "graph",
            "numpy_network",
            "opponents",
            "search",
            "checkpoint",
        )
    ]
    extensions = [sys.modules[fn.__module__] for fn in (choose_child, solve_remaining) if fn]
    report = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threads": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "seed": args.seed,
        "match_seconds": args.match_seconds,
        "games_per_board": args.games,
        "root_exact_edges": 12,
        "acceleration": {
            "search": choose_child is not None,
            "endgame": solve_remaining is not None,
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                *(Path(module.__file__) for module in modules + extensions),
            )
        },
        "latency": latency(evaluators["mcts"], args),
        "games": matches(evaluators, args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
