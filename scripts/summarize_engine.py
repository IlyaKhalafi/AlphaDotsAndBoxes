"""Validate paired engine histories and summarize scores and search work."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from alphaboxes.game import State


def summarize(receipt):
    rows = []
    for board in sorted({tuple(game["board"]) for game in receipt["games"]}):
        games = sorted(
            (game for game in receipt["games"] if tuple(game["board"]) == board),
            key=lambda game: game["index"],
        )
        if len(games) != receipt["games_per_board"] or len(games) % 2:
            raise ValueError("Receipt does not contain the planned number of paired games.")
        if [game["index"] for game in games] != list(range(len(games))):
            raise ValueError("Game indices must be distinct and consecutive.")
        for index in range(0, len(games), 2):
            first, second = games[index : index + 2]
            if first["opening"] != second["opening"] or (
                first["alpha_player"],
                second["alpha_player"],
            ) != (0, 1):
                raise ValueError("Opening pairs must swap seats on the same position.")
        for game in games:
            state = State.new(*board)
            for action in game["opening"]:
                state = state.play(action)
            for move in game["moves"]:
                side = "candidate" if state.player == game["alpha_player"] else "reference"
                engine = "alpha_beta" if side == "candidate" else receipt["reference_engine"]
                if (
                    move["player"] != state.player
                    or move["side"] != side
                    or move["engine"] != engine
                    or move["remaining_edges"] != len(state.legal_actions)
                ):
                    raise ValueError("Move receipt disagrees with the replayed state or player.")
                state = state.play(move["action"])
            if not state.terminal or list(state.scores) != game["scores"]:
                raise ValueError("Incomplete game or incorrect recorded final scores.")
            if (
                state.scores[game["alpha_player"]] - state.scores[1 - game["alpha_player"]]
                != game["alpha_margin"]
            ):
                raise ValueError("Candidate margin disagrees with final scores.")
        wins = sum(game["alpha_margin"] > 0 for game in games)
        draws = sum(game["alpha_margin"] == 0 for game in games)
        row = {
            "board": list(board),
            "games": len(games),
            "wins": wins,
            "draws": draws,
            "losses": len(games) - wins - draws,
            "score": (wins + draws / 2) / len(games),
            "mean_box_margin": float(np.mean([game["alpha_margin"] for game in games])),
            "players": {},
        }
        for side in ("candidate", "reference"):
            neural = [
                move
                for game in games
                for move in game["moves"]
                if move["side"] == side and move["remaining_edges"] > 12
            ]
            alpha = [move for move in neural if move["engine"] == "alpha_beta"]
            row["players"][side] = {
                "neural_moves": len(neural),
                "mean_seconds": float(np.mean([move["seconds"] for move in neural]))
                if neural
                else None,
                "median_depth": float(np.median([move["depth"] for move in alpha]))
                if alpha
                else None,
                "depth_counts": dict(sorted(Counter(move["depth"] for move in alpha).items())),
                "median_nodes": float(np.median([move["nodes"] for move in alpha]))
                if alpha
                else None,
                "median_simulations": float(np.median([move["simulations"] for move in neural]))
                if neural
                else None,
            }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for path in args.receipts:
        snapshot = path.read_bytes()
        receipt = json.loads(snapshot)
        reports.append(
            {
                "receipt": path.name,
                "receipt_sha256": hashlib.sha256(snapshot).hexdigest(),
                "candidate_sha256": receipt["checkpoint_sha256"],
                "reference_sha256": receipt["reference_checkpoint_sha256"],
                "reference_engine": receipt["reference_engine"],
                "results": summarize(receipt),
            }
        )
    report = {
        "validation": "Full legal replays, final scores and seat-swapped openings checked",
        "timing": "Only moves with more than 12 remaining edges enter search-work statistics",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
