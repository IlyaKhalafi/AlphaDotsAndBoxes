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


def component_position(size, paths=(), loops=()) -> State:
    """Build reachable unopened components; every box has exactly two free sides."""
    state = State.new(*size)
    free = set()
    for groups, closed in ((paths, False), (loops, True)):
        for group in groups:
            pairs = list(zip(group[:-1], group[1:], strict=True))
            if closed:
                pairs.append((group[-1], group[0]))
            for first, second in pairs:
                shared = set(state.board.boxes[first]) & set(state.board.boxes[second])
                if len(shared) != 1:
                    raise ValueError("Component boxes must be adjacent.")
                free.update(shared)
            if not closed:
                for endpoint in (group[0], group[-1]):
                    external = [
                        e
                        for e in state.board.boxes[endpoint]
                        if len(state.board.edge_boxes[e]) == 1 and e not in free
                    ]
                    if not external:
                        raise ValueError("A path endpoint must reach the board boundary.")
                    free.add(min(external))
    if any(sum(e in free for e in box) != 2 for box in state.board.boxes):
        raise ValueError("Components must partition every box into paths or loops.")
    for action in range(state.board.num_edges):
        if action not in free:
            state = state.play(action)
    return state


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
    structured = [
        ("chains_5_10", (3, 5), [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 14, 13, 12, 11, 10]], []),
        ("chains_4_12", (4, 4), [[0, 1, 2, 3], [4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15]], []),
        ("loops_6_4_chain_5", (3, 5), [[10, 11, 12, 13, 14]], [[0, 1, 2, 7, 6, 5], [3, 4, 9, 8]]),
        ("loop_8_chains_2_5", (3, 5), [[4, 9], [10, 11, 12, 13, 14]], [[0, 1, 2, 3, 8, 7, 6, 5]]),
        (
            "loops_4_4_chain_8",
            (4, 4),
            [[8, 9, 10, 11, 15, 14, 13, 12]],
            [[0, 1, 5, 4], [2, 3, 7, 6]],
        ),
        ("long_chain_15", (3, 5), [[0, 1, 2, 3, 4, 9, 8, 7, 6, 5, 10, 11, 12, 13, 14]], []),
    ]
    for label, size, paths, loops in structured:
        records.append(position_record(component_position(size, paths, loops), label))
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
        "limitation": (
            "Fixed simulation count; solver time is additional. Fixtures shared across seeds."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
