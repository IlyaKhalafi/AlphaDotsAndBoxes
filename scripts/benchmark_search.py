"""Measure fixed-state search or exact endgames without training weights."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from alphaboxes.game import State
from alphaboxes.opponents import solve
from alphaboxes.search import MCTS, NeuralEvaluator, SearchConfig


def positions(size, count, seed, mode):
    rng = np.random.default_rng(seed)
    states = []
    for index in range(count):
        state = State.new(*size)
        moves = max(0, state.board.num_edges - 12) if mode == "oracle" else 8 + index
        for _ in range(min(moves, state.board.num_edges - 1)):
            state = state.play(int(rng.choice(state.legal_actions)))
        states.append(state)
    return states


def benchmark(args):
    states = positions(args.size, args.states, args.seed, args.mode)
    metadata = {}
    search = None
    if args.mode == "search":
        import torch

        from alphaboxes.checkpoint import load_agent

        if args.checkpoint is None:
            raise ValueError("Search benchmarking requires --checkpoint.")
        torch.set_num_threads(1)
        if args.device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(1024**3 / total)
        module, metadata = load_agent(args.checkpoint)
        search = MCTS(
            NeuralEvaluator(module.to(args.device)),
            SearchConfig(simulations=args.simulations, exact_threshold=12),
            args.seed,
        )
        search.policy(states[0])
    durations = []
    for _ in range(args.repeats):
        if search:
            search.evaluator.clear_cache()
        started = time.perf_counter()
        if search:
            results = []
            for start in range(0, len(states), args.batch):
                results.extend(search.policies(states[start : start + args.batch]))
            if args.device == "cuda":
                torch.cuda.synchronize()
        else:
            results = [solve(state) for state in states]
        durations.append(time.perf_counter() - started)
    result = {
        "mode": args.mode,
        "device": args.device if search else "cpu",
        "batch_size": args.batch if search else 1,
        "size": args.size,
        "states": args.states,
        "seed": args.seed,
        "simulations": args.simulations if search else None,
        "seconds": durations,
        "median_seconds": statistics.median(durations),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "predictions": [{"policy": policy.tolist(), "value": value} for policy, value in results]
        if search
        else [{"margin": margin, "actions": actions} for margin, actions in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["search", "oracle"], default="search")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--states", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--simulations", type=int, default=512)
    parser.add_argument("--size", default="5x5")
    parser.add_argument("--seed", type=int, default=20261010)
    args = parser.parse_args()
    args.size = tuple(map(int, args.size.split("x")))
    if len(args.size) != 2 or min(args.batch, args.states, args.repeats, args.simulations) < 1:
        parser.error("Use a rows×cols size and positive benchmark counts.")
    benchmark(args)


if __name__ == "__main__":
    main()
