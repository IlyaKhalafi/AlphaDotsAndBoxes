"""Measure exact local regrets where the chain-and-loop oracle recognizes a game."""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from time import monotonic

from alphaboxes import components, endgame, game
from alphaboxes.endgame import DegreeTwoSolver
from alphaboxes.game import State


def inspect(games, seconds=0.05):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Oracle time budget must be finite and positive.")
    solver = DegreeTwoSolver()
    counts = {side: Counter() for side in ("candidate", "reference")}
    mistakes, first_exact = [], []
    for episode in games:
        state = State.new(*episode["board"])
        for action in episode["opening"]:
            state = state.play(action)
        found_exact = False
        for ply, move in enumerate(episode["moves"]):
            side = "candidate" if state.player == episode["alpha_player"] else "reference"
            if move["side"] != side or move["player"] != state.player:
                raise ValueError("Move side disagrees with the replayed state.")
            row = counts[side]
            row["decisions"] += 1
            deadline = monotonic() + seconds
            try:
                best = solver.value(state, deadline)
                chosen = (
                    solver.action_value(state, move["action"], deadline)
                    if best is not None
                    else None
                )
            except TimeoutError:
                best = chosen = None
                row["oracle_timeouts"] += 1
            if best is not None and chosen is not None:
                best_boxes = round(best * state.board.num_boxes)
                chosen_boxes = round(chosen * state.board.num_boxes)
                regret = best_boxes - chosen_boxes
                if regret < 0:
                    raise ValueError("Chosen action exceeds the oracle's optimal margin.")
                outcome_reduced = ((chosen_boxes > 0) - (chosen_boxes < 0)) < (
                    (best_boxes > 0) - (best_boxes < 0)
                )
                record = {
                    "game": episode["index"],
                    "ply": ply,
                    "side": side,
                    "remaining_edges": len(state.legal_actions),
                    "action": move["action"],
                    "best_margin_boxes": best_boxes,
                    "chosen_margin_boxes": chosen_boxes,
                    "outcome_reducing": outcome_reduced,
                }
                row["exactly_diagnosed_decisions"] += 1
                row["margin_regret_boxes"] += regret
                row["suboptimal_margin_decisions"] += regret > 0
                row["outcome_reducing_decisions"] += outcome_reduced
                if not found_exact:
                    first_exact.append(record)
                    found_exact = True
                if regret:
                    mistakes.append(record)
            state = state.play(move["action"])
        if not state.terminal or list(state.scores) != episode["scores"]:
            raise ValueError("Games must be complete and correctly scored.")
    return {
        "counts": {side: dict(row) for side, row in counts.items()},
        "first_exact_decisions": first_exact,
        "mistakes": mistakes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seconds", type=float, default=0.05)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("Game limit must be positive.")
    snapshot = args.receipt.read_bytes()
    receipt = json.loads(snapshot)
    games = sorted(receipt["games"], key=lambda episode: (episode["board"], episode["index"]))
    if args.limit is not None:
        games = games[: args.limit]
    report = {
        "scope": (
            "Exact oracle answers only in its recognized domain. "
            "Regrets are local, not additive predicted match losses. "
            "Timeouts are excluded; coverage can depend on CPU scheduling."
        ),
        "receipt_sha256": hashlib.sha256(snapshot).hexdigest(),
        "game_indices": [episode["index"] for episode in games],
        "canonical_games_sha256": hashlib.sha256(
            json.dumps(games, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "candidate_sha256": receipt.get("checkpoint_sha256"),
        "oracle_seconds_per_decision": args.seconds,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(game.__file__),
                Path(endgame.__file__),
                Path(components.__file__),
            )
        },
        **inspect(games, args.seconds),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"]))


if __name__ == "__main__":
    main()
