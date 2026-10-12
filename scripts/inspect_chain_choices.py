"""Replay complete benchmark games and count opened-chain decision choices."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from alphaboxes.chain_moves import allowed_actions
from alphaboxes.game import State


def inspect(receipt):
    counts = {side: Counter() for side in ("candidate", "reference")}
    for game in receipt["games"]:
        state = State.new(*game["board"])
        for action in game["opening"]:
            state = state.play(action)
        for move in game["moves"]:
            side = "candidate" if state.player == game["alpha_player"] else "reference"
            if side != move["side"] or state.player != move["player"]:
                raise ValueError("Move side disagrees with the replayed state.")
            retained = allowed_actions(state)
            row = counts[side]
            row["decisions"] += 1
            if len(retained) < len(state.legal_actions):
                row["restricted_positions"] += 1
                row["retained_choices"] += len(retained)
                row["outside_retained_choices"] += move["action"] not in retained
                if len(retained) == 2:
                    row["capture_or_handout_positions"] += 1
                    row["noncapture_choices"] += not state.captures(move["action"])
            state = state.play(move["action"])
        if not state.terminal or list(state.scores) != game["scores"]:
            raise ValueError("Receipt must contain complete, correctly scored games.")
    return {side: dict(row) for side, row in counts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for path in args.receipts:
        snapshot = path.read_bytes()
        reports.append(
            {
                "receipt": path.name,
                "receipt_sha256": hashlib.sha256(snapshot).hexdigest(),
                "choices": inspect(json.loads(snapshot)),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "interpretation": (
                    "Outside retained choices are diagnostic, not quantified score regrets."
                ),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "reports": reports,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
