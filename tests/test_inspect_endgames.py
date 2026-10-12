import json
import os
import subprocess
import sys
from pathlib import Path

from alphaboxes.game import State
from alphaboxes.opponents import solve


def test_audit_detects_a_capture_that_gives_up_a_forced_win(tmp_path):
    state = State.new(2, 3)
    opening = [e for e, (r1, _, r2, _) in enumerate(state.board.edges) if r1 == r2]
    opening.extend((state.board.boxes[0][2], state.board.boxes[0][3]))
    for action in opening:
        state = state.play(action)
    first = next(action for action in state.legal_actions if state.captures(action))
    history = []
    while not state.terminal:
        action = first if not history else solve(state)[1][0]
        history.append(
            {
                "player": state.player,
                "engine": "mcts",
                "remaining_edges": len(state.legal_actions),
                "action": action,
                "seconds": 0.01,
                "simulations": 0,
                "value": 0.0,
            }
        )
        state = state.play(action)
    games = []
    for seat in (0, 1):
        games.append(
            {
                "board": [2, 3],
                "index": seat,
                "alpha_player": seat,
                "opening": opening,
                "scores": list(state.scores),
                "alpha_margin": 0,
                "moves": [
                    {**move, "side": "candidate" if move["player"] == seat else "reference"}
                    for move in history
                ],
            }
        )
    receipt = tmp_path / "handout.json"
    receipt.write_text(
        json.dumps(
            {
                "games_per_board": 2,
                "reference_engine": "mcts",
                "candidate_engine": "mcts",
                "games": games,
            }
        )
    )
    output = tmp_path / "audit.json"
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/inspect_endgames.py"),
            "--receipts",
            str(receipt),
            "--min-remaining",
            "1",
            "--output",
            str(output),
        ],
        cwd=root,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
        check=True,
        capture_output=True,
        timeout=30,
    )
    result = json.loads(output.read_text())["reports"][0]
    for side in ("candidate", "reference"):
        assert result["counts"][side]["suboptimal"] == 1
        assert result["counts"][side]["lost_wins"] == 1
        assert result["counts"][side]["lost_box_margin"] == 2
    assert len(result["errors"]) == 2
