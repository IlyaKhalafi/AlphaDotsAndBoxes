"""Compare NumPy inference and fixed-work search with a trusted source snapshot."""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import statistics
import struct
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from alphaboxes.game import State
from alphaboxes.graph import encode, neighbor_indices
from alphaboxes.numpy_network import NumpyEvaluator, NumpyNetwork, load_network
from alphaboxes.search import MCTS, SearchConfig


def load_reference(source: Path, name: str):
    module_name = f"alphaboxes._benchmark_reference_{name}"
    spec = importlib.util.spec_from_file_location(module_name, source / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def trace_hash(root):
    """Fingerprint every visited node, including backups and player perspective."""
    digest = hashlib.sha256()
    pending = [root]
    while pending:
        node = pending.pop()
        digest.update(
            struct.pack(
                "<qqddqd",
                node.action,
                node.visits,
                node.value_sum,
                node.prior,
                node.player,
                node.evaluation,
            )
        )
        if node.children:
            pending.extend(child for child in reversed(node.children.values()) if child.visits)
    return digest.hexdigest()


def inference_probe(networks, shapes, repeats, seed):
    rows = []
    for shape in shapes:
        state = State.new(*shape)
        rng = np.random.default_rng(seed)
        for moves in (0, 6, state.board.num_edges // 2):
            while state.board.num_edges - len(state.legal_actions) < moves:
                state = state.play(int(rng.choice(state.legal_actions)))
            sparse = state.board.num_nodes >= 256
            obs = encode(state, include_adjacency=not sparse)
            options = [
                {"neighbors": neighbor_indices(*shape)} if sparse else {},
                {"board_shape": shape} if sparse else {},
            ]
            samples, outputs = [[], []], [None, None]
            for repeat in range(repeats + 5):
                for index in (repeat % 2, 1 - repeat % 2):
                    started = perf_counter()
                    outputs[index] = networks[index].forward(obs, **options[index])
                    elapsed = perf_counter() - started
                    if repeat >= 5:
                        samples[index].append(elapsed)
            exact = np.array_equal(outputs[0][0], outputs[1][0]) and (
                outputs[0][1] == outputs[1][1]
            )
            if not exact:
                raise AssertionError(f"Inference differs on {shape}, {moves} moves.")
            rows.append(
                {
                    "board": shape,
                    "played_edges": moves,
                    "median_seconds": [statistics.median(s) for s in samples],
                    "seconds": samples,
                    "exact_logits_and_value": exact,
                }
            )
    return rows


def search_probe(reference, networks, shapes, simulations, repeats, moves, seed):
    rows = []
    for shape in shapes:
        initial = State.new(*shape)
        rng = np.random.default_rng(seed)
        for _ in range(6):
            initial = initial.play(int(rng.choice(initial.legal_actions)))
        for repeat in range(repeats):
            outputs = [[], []]
            for index in (repeat % 2, 1 - repeat % 2):
                state = initial
                roots = []
                evaluator_type = reference[0].NumpyEvaluator if index == 0 else NumpyEvaluator
                evaluator = evaluator_type(networks[index], cache_size=32768)
                search_type = reference[1].MCTS if index == 0 else MCTS
                search = search_type(
                    evaluator,
                    SearchConfig(
                        simulations=simulations,
                        noise_fraction=0,
                        exact_threshold=12,
                        fpu_reduction=0.33,
                        sparse_selection=True,
                    ),
                    seed=seed,
                    root_observer=roots.append,
                )
                for _move in range(moves):
                    gc.collect()
                    started = perf_counter()
                    policy, value = search.policy(state)
                    elapsed = perf_counter() - started
                    outputs[index].append(
                        {
                            "seconds": elapsed,
                            "action": int(policy.argmax()),
                            "value": value,
                            "policy": policy.tolist(),
                            "visited_tree_sha256": trace_hash(roots.pop()),
                            "simulations": search.completed_simulations,
                        }
                    )
                    state = state.play(int(policy.argmax()))
            for move in range(moves):
                pair = [outputs[index][move] for index in range(2)]
                for field in ("action", "value", "policy", "visited_tree_sha256", "simulations"):
                    if pair[0][field] != pair[1][field]:
                        raise AssertionError(f"Search {field} differs on {shape}, move {move}.")
                rows.append({"board": shape, "repeat": repeat, "move": move, "players": pair})
                print(
                    json.dumps(
                        {"board": shape, "move": move, "seconds": [p["seconds"] for p in pair]}
                    ),
                    flush=True,
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/larger.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", default=["5x5", "10x10", "12x12"])
    parser.add_argument("--inference-repeats", type=int, default=40)
    parser.add_argument("--simulations", type=int, default=0)
    parser.add_argument("--search-repeats", type=int, default=2)
    parser.add_argument("--search-moves", type=int, default=3)
    parser.add_argument("--seed", type=int, default=61001)
    args = parser.parse_args()
    shapes = [tuple(map(int, size.split("x"))) for size in args.sizes]
    if any(len(shape) != 2 or min(shape) < 1 for shape in shapes):
        parser.error("Use positive rows×cols board dimensions.")
    if (
        min(args.inference_repeats, args.search_repeats, args.search_moves) < 1
        or args.simulations < 0
    ):
        parser.error("Use positive repeat and move counts, and nonnegative simulations.")
    source = args.reference_source / "alphaboxes"
    reference_np = load_reference(source, "numpy_network")
    reference_search = load_reference(source, "search")
    reference_search.ChildSelection = load_reference(source, "selection").ChildSelection
    network, metadata = load_network(args.checkpoint)
    networks = [
        reference_np.NumpyNetwork(network.weights, network.width, network.depth, network.structure),
        NumpyNetwork(network.weights, network.width, network.depth, network.structure),
    ]
    result = {
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "reference_source_sha256": {
            name: hashlib.sha256((source / f"{name}.py").read_bytes()).hexdigest()
            for name in ("numpy_network", "search", "selection")
        },
        "candidate_source_sha256": {
            name: hashlib.sha256((Path("src/alphaboxes") / f"{name}.py").read_bytes()).hexdigest()
            for name in ("numpy_network", "search", "selection")
        },
        "numpy_version": np.__version__,
        "thread_environment": {
            name: os.environ.get(name) for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "seed": args.seed,
        "player_order": ["reference", "candidate"],
        "search_config": {
            "simulations": args.simulations,
            "cpuct": 1.5,
            "noise_fraction": 0,
            "exact_threshold": 12,
            "fpu_reduction": 0.33,
            "sparse_selection": True,
            "prediction_cache_entries": 32768,
            "repeats": args.search_repeats,
            "moves": args.search_moves,
        },
        "inference": inference_probe(networks, shapes, args.inference_repeats, args.seed),
        "search": search_probe(
            (reference_np, reference_search),
            networks,
            shapes,
            args.simulations,
            args.search_repeats,
            args.search_moves,
            args.seed,
        )
        if args.simulations
        else [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
