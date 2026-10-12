import importlib.util
from pathlib import Path

import pytest

from alphaboxes.game import State

spec = importlib.util.spec_from_file_location(
    "summarize_engine", Path(__file__).resolve().parents[1] / "scripts/summarize_engine.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def stopped_receipt():
    games = []
    for index in (0, 1, 4, 5):
        state = State.new(1, 1)
        seat, moves = index % 2, []
        for action in range(4):
            moves.append(
                {
                    "action": action,
                    "player": state.player,
                    "side": "candidate" if state.player == seat else "reference",
                    "engine": "mcts",
                    "remaining_edges": len(state.legal_actions),
                    "simulations": 2,
                    "seconds": 0.001,
                }
            )
            state = state.play(action)
        games.append(
            {
                "index": index,
                "board": [1, 1],
                "opening": [],
                "alpha_player": seat,
                "scores": list(state.scores),
                "alpha_margin": state.scores[seat] - state.scores[1 - seat],
                "moves": moves,
            }
        )
    return {
        "games": games,
        "games_per_board": 4,
        "planned_games_per_board": 6,
        "status": "stopped_confidence_gate_unreachable",
        "candidate_engine": "mcts",
        "reference_engine": "mcts",
        "budget_mode": "simulations",
        "match_simulations": 2,
    }


def test_stopped_run_retains_complete_out_of_order_pairs():
    result = module.summarize(stopped_receipt())[0]
    assert result["games"] == 4
    assert result["wins"] == result["losses"] == 2
    assert result["score"] == 0.5


def test_completed_run_cannot_hide_missing_opening_pairs():
    receipt = stopped_receipt()
    receipt["status"] = "completed"
    with pytest.raises(ValueError, match="consecutive"):
        module.summarize(receipt)


@pytest.mark.parametrize("planned", [None, 3, 5, True, 6.0])
def test_stopped_run_requires_valid_planned_bounds(planned):
    receipt = stopped_receipt()
    receipt["planned_games_per_board"] = planned
    with pytest.raises(ValueError, match="planned"):
        module.summarize(receipt)


@pytest.mark.parametrize("indices", [(0, 2, 4, 5), (0, 1, 6, 7), (0, 0, 4, 5), (False, 1, 4, 5)])
def test_stopped_run_rejects_invalid_pair_indices(indices):
    receipt = stopped_receipt()
    for game, index in zip(receipt["games"], indices, strict=True):
        game["index"] = index
    with pytest.raises(ValueError):
        module.summarize(receipt)
