"""Audit recorded decisions against the structural exact endgame solver."""

import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic

from summarize_engine import summarize

from alphaboxes.endgame import DegreeTwoSolver, EndgameTimeout, position
from alphaboxes.game import State


def audit(receipt, seconds, minimum, maximum):
    summarize(receipt)  # Validate complete legal histories before inspecting choices.
    solver = DegreeTwoSolver()
    counts = {
        side: {
            "eligible": 0,
            "solved": 0,
            "timeouts": 0,
            "suboptimal": 0,
            "lost_wins": 0,
            "lost_box_margin": 0,
        }
        for side in ("candidate", "reference")
    }
    errors = []
    for game in receipt["games"]:
        state = State.new(*game["board"])
        for action in game["opening"]:
            state = state.play(action)
        for move_index, move in enumerate(game["moves"]):
            child = state.play(move["action"])
            remaining = move["remaining_edges"]
            if minimum <= remaining <= maximum and position(state) is not None:
                row = counts[move["side"]]
                row["eligible"] += 1
                try:
                    deadline = monotonic() + seconds
                    value, actions = solver.solve(state, deadline)
                    if move["action"] not in actions:
                        # solve() has already completed every child's exact value.
                        actual = solver.value(child, monotonic() + seconds)
                        if child.player != state.player:
                            actual = -actual
                        regret = round((value - actual) * state.board.num_boxes)
                        row["suboptimal"] += 1
                        row["lost_wins"] += int(value > 0 and actual <= 0)
                        row["lost_box_margin"] += regret
                        errors.append(
                            {
                                "board": game["board"],
                                "game": game["index"],
                                "move": move_index,
                                "side": move["side"],
                                "remaining_edges": remaining,
                                "action": move["action"],
                                "optimal_actions": list(actions),
                                "optimal_value": value,
                                "actual_value": actual,
                                "box_margin_regret": regret,
                            }
                        )
                    row["solved"] += 1
                except EndgameTimeout:
                    row["timeouts"] += 1
            state = child
    return {"counts": counts, "errors": errors, "solver_nodes": solver.nodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--seconds", type=float, default=0.05)
    parser.add_argument("--min-remaining", type=int, default=13)
    parser.add_argument("--max-remaining", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.seconds <= 1 or not 1 <= args.min_remaining <= args.max_remaining:
        parser.error("Use a positive solve budget up to one second and an ordered edge range.")
    reports = []
    for path in args.receipts:
        snapshot = path.read_bytes()
        reports.append(
            {
                "receipt": path.name,
                "receipt_sha256": hashlib.sha256(snapshot).hexdigest(),
                **audit(json.loads(snapshot), args.seconds, args.min_remaining, args.max_remaining),
            }
        )
    result = {
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path("src/alphaboxes/endgame.py"))
        },
        "solve_seconds": args.seconds,
        "min_remaining": args.min_remaining,
        "max_remaining": args.max_remaining,
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for report in reports:
        print(json.dumps({"receipt": report["receipt"], "counts": report["counts"]}))


if __name__ == "__main__":
    main()
