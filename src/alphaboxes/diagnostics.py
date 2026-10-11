"""Fixed legal positions and exact move-quality checks for controlled experiments."""

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from alphaboxes.chains import components
from alphaboxes.checkpoint import load_evaluator
from alphaboxes.game import State
from alphaboxes.opponents import solve
from alphaboxes.search import MCTS, SearchConfig


def exact_actions(state: State) -> dict[int, float]:
    """Optimal final box margin after each move, from the current player's view."""
    result = {}
    for action in state.legal_actions:
        child = state.play(action)
        margin, _ = solve(child)
        result[action] = margin if child.player == state.player else -margin
    return result


def position_record(state: State, label: str) -> dict:
    if state.terminal or len(state.legal_actions) > 18:
        raise ValueError("Diagnostic positions need between one and 18 legal edges.")
    scores = exact_actions(state)
    best = max(scores.values())
    tags = []
    captures = [len(state.captures(action)) for action in state.legal_actions]
    if max(captures):
        tags.append("capture_available")
    if max(captures) == 2:
        tags.append("double_capture")
    if components(state):
        tags.append("chain_or_loop_endgame")
    if any(np.sign(value) != np.sign(best) for value in scores.values()):
        tags.append("outcome_sensitive")
    return {
        "id": label,
        "state": {
            "size": [state.board.rows, state.board.cols],
            "edges": state.edges,
            "owners": state.owners,
            "player": state.player,
        },
        "tags": tags,
        "action_margins": {str(key): value for key, value in scores.items()},
        "best_margin": best,
        "best_outcome": float(np.sign(best)),
    }


def generate_suite(seed: int = 3100, repetitions: int = 4) -> dict:
    if repetitions < 1:
        raise ValueError("Use at least one position per board and remaining-edge count.")
    rng = np.random.default_rng(seed)
    records = []
    for size in [(3, 3), (4, 4), (5, 5), (2, 5)]:
        for remaining in (12, 14, 16):
            for index in range(repetitions):
                state = State.new(*size)
                while len(state.legal_actions) > remaining:
                    state = state.play(int(rng.choice(state.legal_actions)))
                records.append(position_record(state, f"{size[0]}x{size[1]}-{remaining}-{index}"))
    # Reachable fixtures deliberately require giving boxes back to retain control.
    state = State.new(3, 5)
    for action in range(20):
        state = state.play(action)
    for action in range(20, 24):
        state = state.play(action)
    records.append(position_record(state, "chain_handout_two"))
    state = State.new(4, 4)
    group = {b: (b // 4 // 2, b % 4 // 2) for b in range(16)}
    loop_edges = {
        e
        for e, boxes in enumerate(state.board.edge_boxes)
        if len(boxes) == 2 and group[boxes[0]] == group[boxes[1]]
    }
    for action in range(state.board.num_edges):
        if action not in loop_edges:
            state = state.play(action)
    state = state.play(min(loop_edges))
    records.append(position_record(state, "loop_handout_four"))
    return {
        "format_version": 1,
        "seed": seed,
        "repetitions": repetitions,
        "generation": "Random legal moves plus two shared control fixtures",
        "oracle": "Exact minimax final box margin; winning-outcome and margin metrics differ",
        "positions": records,
    }


def restore_position(record: dict) -> State:
    data = record["state"]
    geometry = State.new(*data["size"]).board
    return State(geometry, tuple(data["edges"]), tuple(data["owners"]), data["player"])


def benchmark_suite(checkpoint: Path, suite_path: Path, output: Path, config: SearchConfig) -> dict:
    snapshot = suite_path.read_bytes()
    suite = json.loads(snapshot)
    evaluator, metadata = load_evaluator(checkpoint)
    rows = []
    for index, record in enumerate(suite["positions"]):
        state = restore_position(record)
        evaluator.clear_cache()
        search = MCTS(evaluator, config, suite["seed"] + index)
        started = time.perf_counter()
        policy, value = search.policy(state)
        seconds = time.perf_counter() - started
        actions = np.flatnonzero(policy == policy.max())
        # The same deterministic tie draw is used for every search variant.
        rng = np.random.default_rng(np.random.SeedSequence([suite["seed"], index]))
        action = int(rng.choice(actions))
        margins = {int(key): value for key, value in record["action_margins"].items()}
        optimal_outcome = [
            a for a, margin in margins.items() if np.sign(margin) == record["best_outcome"]
        ]
        rows.append(
            {
                "id": record["id"],
                "tags": record["tags"],
                "action": action,
                "outcome_correct": bool(np.sign(margins[action]) == record["best_outcome"]),
                "margin_optimal": bool(np.isclose(margins[action], record["best_margin"])),
                "margin_regret_boxes": (record["best_margin"] - margins[action])
                * state.board.num_boxes,
                "optimal_outcome_mass": float(policy[optimal_outcome].sum()),
                "value": value,
                "value_absolute_error": abs(value - record["best_outcome"]),
                "seconds": seconds,
                "policy": policy.tolist(),
            }
        )
    grouped = defaultdict(list)
    for row in rows:
        grouped["all"].append(row)
        for tag in row["tags"]:
            grouped[tag].append(row)
    summary = {
        tag: {
            "positions": len(group),
            "outcome_accuracy": float(np.mean([r["outcome_correct"] for r in group])),
            "margin_accuracy": float(np.mean([r["margin_optimal"] for r in group])),
            "mean_margin_regret_boxes": float(np.mean([r["margin_regret_boxes"] for r in group])),
            "mean_value_absolute_error": float(np.mean([r["value_absolute_error"] for r in group])),
            "seconds": sum(r["seconds"] for r in group),
        }
        for tag, group in grouped.items()
    }
    result = {
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "suite_sha256": hashlib.sha256(snapshot).hexdigest(),
        "search": asdict(config),
        "summary": summary,
        "positions": rows,
        "limitation": "Fixed simulation count; solver time is additional. Fixtures shared across seeds.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
