import importlib.util
from pathlib import Path

import pytest

from alphaboxes.game import State

spec = importlib.util.spec_from_file_location(
    "inspect_endgame_choices",
    Path(__file__).resolve().parents[1] / "scripts/inspect_endgame_choices.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
inspect = module.inspect


def single_box_game():
    state = State.new(1, 1)
    moves = []
    for action in range(4):
        moves.append(
            {
                "action": action,
                "player": state.player,
                "side": "candidate" if state.player == 0 else "reference",
            }
        )
        state = state.play(action)
    return {
        "index": 0,
        "board": [1, 1],
        "alpha_player": 0,
        "opening": [],
        "moves": moves,
        "scores": list(state.scores),
    }


def test_replays_complete_game_and_separates_oracle_coverage():
    result = inspect([single_box_game()])
    assert sum(row["decisions"] for row in result["counts"].values()) == 4
    assert sum(row.get("exactly_diagnosed_decisions", 0) for row in result["counts"].values()) == 2
    assert result["mistakes"] == []
    assert len(result["first_exact_decisions"]) == 1


def test_rejects_incomplete_or_incorrect_history():
    episode = single_box_game()
    episode["moves"].pop()
    with pytest.raises(ValueError, match="complete"):
        inspect([episode])
    episode = single_box_game()
    episode["moves"][0]["side"] = "reference"
    with pytest.raises(ValueError, match="side"):
        inspect([episode])


def test_detects_action_that_turns_a_win_into_a_loss():
    opening = [6, 2, 9, 5, 4, 3, 10, 1, 0]
    state = State.new(2, 2)
    for action in opening:
        state = state.play(action)
    moves = []
    while not state.terminal:
        action = 8 if not moves else min(state.legal_actions)
        moves.append(
            {
                "action": action,
                "player": state.player,
                "side": "candidate" if state.player == 0 else "reference",
            }
        )
        state = state.play(action)
    episode = {
        "index": 0,
        "board": [2, 2],
        "alpha_player": 0,
        "opening": opening,
        "moves": moves,
        "scores": list(state.scores),
    }
    result = inspect([episode])
    mistake = result["mistakes"][0]
    assert mistake["best_margin_boxes"] == 4
    assert mistake["chosen_margin_boxes"] == -2
    assert mistake["outcome_reducing"]
    assert result["counts"]["candidate"]["outcome_reducing_decisions"] == 1


def test_timeouts_do_not_become_oracle_labels(monkeypatch):
    def timeout(*args):
        raise TimeoutError

    monkeypatch.setattr(module.DegreeTwoSolver, "value", timeout)
    result = inspect([single_box_game()])
    assert sum(row["oracle_timeouts"] for row in result["counts"].values()) == 4
    assert result["first_exact_decisions"] == result["mistakes"] == []
    assert all("exactly_diagnosed_decisions" not in row for row in result["counts"].values())


def test_counts_early_gifts_without_claiming_they_are_strategic_errors():
    state = State.new(1, 2)
    opening = list(state.board.boxes[0][:2])
    for action in opening:
        state = state.play(action)
    risky = next(a for a in state.legal_actions if a not in module.safe_actions(state))
    moves = []
    while not state.terminal:
        action = risky if not moves else min(state.legal_actions)
        moves.append(
            {
                "action": action,
                "player": state.player,
                "side": "candidate" if state.player == 0 else "reference",
            }
        )
        state = state.play(action)
    episode = {
        "index": 0,
        "board": [1, 2],
        "alpha_player": 0,
        "opening": opening,
        "moves": moves,
        "scores": list(state.scores),
    }
    result = inspect([episode])
    assert result["counts"]["candidate"]["early_gifts_with_safe_moves"] == 1
    assert result["early_gifts"][0]["action"] == risky
    assert result["early_gifts"][0]["safe_actions"] > 0


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_oracle_budget(seconds):
    with pytest.raises(ValueError, match="budget"):
        inspect([], seconds)
