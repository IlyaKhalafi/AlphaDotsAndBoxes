"""Validate paired engine histories and summarize scores and search work."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from alphaboxes.game import State


def summarize(receipt):
    stopped = receipt.get("status", "").startswith("stopped_")
    if (
        not stopped
        and receipt.get("planned_games_per_board", receipt["games_per_board"])
        != receipt["games_per_board"]
    ):
        raise ValueError("Completed receipts require all planned opening pairs.")
    if not receipt["games"] and receipt["games_per_board"]:
        raise ValueError("Receipt does not contain the planned number of paired games.")
    rows = []
    for board in sorted({tuple(game["board"]) for game in receipt["games"]}):
        games = sorted(
            (game for game in receipt["games"] if tuple(game["board"]) == board),
            key=lambda game: game["index"],
        )
        if len(games) != receipt["games_per_board"] or len(games) % 2:
            raise ValueError("Receipt does not contain the planned number of paired games.")
        indices = [game["index"] for game in games]
        if any(type(index) is not int for index in indices) or len(set(indices)) != len(indices):
            raise ValueError("Game indices must be distinct integers.")
        if stopped:
            planned = receipt.get("planned_games_per_board")
            if (
                type(planned) is not int
                or planned < len(games)
                or planned % 2
                or any(not 0 <= index < planned for index in indices)
            ):
                raise ValueError("Stopped receipts require valid planned game bounds.")
        elif indices != list(range(len(games))):
            raise ValueError("Completed game indices must be consecutive.")
        for index in range(0, len(games), 2):
            first, second = games[index : index + 2]
            if (
                first["index"] % 2
                or second["index"] != first["index"] + 1
                or first["opening"] != second["opening"]
            ) or (
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
                engine = (
                    receipt.get("candidate_engine", "alpha_beta")
                    if side == "candidate"
                    else receipt["reference_engine"]
                )
                if (
                    move["player"] != state.player
                    or move["side"] != side
                    or move["engine"] != engine
                    or move["remaining_edges"] != len(state.legal_actions)
                ):
                    raise ValueError("Move receipt disagrees with the replayed state or player.")
                if receipt.get("budget_mode") == "simulations" and engine == "mcts":
                    simulations = move["simulations"]
                    if not isinstance(simulations, int) or not (
                        0
                        <= simulations
                        <= receipt.get(f"{side}_simulations", receipt["match_simulations"])
                    ):
                        raise ValueError("Move exceeds the recorded simulation budget.")
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
        game_scores = np.array(
            [
                1.0 if game["alpha_margin"] > 0 else 0.5 if game["alpha_margin"] == 0 else 0.0
                for game in games
            ]
        )
        pair_scores = game_scores.reshape(-1, 2).mean(axis=1)
        rng = np.random.default_rng(4199)
        bootstrap = pair_scores[
            rng.integers(len(pair_scores), size=(20000, len(pair_scores)))
        ].mean(axis=1)
        row = {
            "board": list(board),
            "games": len(games),
            "wins": wins,
            "draws": draws,
            "losses": len(games) - wins - draws,
            "score": (wins + draws / 2) / len(games),
            "paired_bootstrap_95pct": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
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
            openings = [game["moves"][0] for game in games if game["moves"][0]["side"] == side]
            alpha_openings = [move for move in openings if move["engine"] == "alpha_beta"]
            row["players"][side] = {
                "max_seconds": max(
                    move["seconds"]
                    for game in games
                    for move in game["moves"]
                    if move["side"] == side
                ),
                "deadline_fallbacks": sum(
                    bool(move.get("timed_out", False))
                    for game in games
                    for move in game["moves"]
                    if move["side"] == side
                ),
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
                "simulation_counts": dict(
                    sorted(Counter(move["simulations"] for move in neural).items())
                ),
                "structural_solve_hits": sum(move.get("endgame_hits", 0) for move in neural),
                "proven_nodes": sum(move.get("proven_nodes", 0) for move in neural),
                "inherited_visits": sum(move.get("inherited_visits", 0) for move in neural),
                "decisions_with_reused_visits": sum(
                    move.get("inherited_visits", 0) > 0 for move in neural
                ),
                **{
                    key: sum(move.get(key, 0) for move in neural)
                    for key in (
                        "rollout_episodes",
                        "rollout_moves",
                        "rollout_oracle_finishes",
                        "forced_capture_positions",
                        "forced_capture_moves",
                        "forced_capture_terminal_finishes",
                    )
                },
                "opening_positions": len(openings),
                "opening_depth_counts": dict(
                    sorted(Counter(move["depth"] for move in alpha_openings).items())
                ),
                "opening_median_nodes": float(np.median([move["nodes"] for move in alpha_openings]))
                if alpha_openings
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
                "status": receipt.get("status", "completed"),
                "planned_games_per_board": receipt.get(
                    "planned_games_per_board", receipt["games_per_board"]
                ),
                "candidate_sha256": receipt["checkpoint_sha256"],
                "reference_sha256": receipt["reference_checkpoint_sha256"],
                "reference_engine": receipt["reference_engine"],
                "candidate_engine": receipt.get("candidate_engine", "alpha_beta"),
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
