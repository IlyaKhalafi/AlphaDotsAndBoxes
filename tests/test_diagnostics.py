import numpy as np
import pytest

from alphaboxes.diagnostics import (
    component_position,
    exact_actions,
    position_record,
    restore_position,
)
from alphaboxes.game import State
from alphaboxes.opponents import solve


def test_action_oracle_respects_extra_turns_and_existing_scores():
    rng = np.random.default_rng(12)
    for _ in range(10):
        state = State.new(2, 2)
        while len(state.legal_actions) > 6:
            state = state.play(int(rng.choice(state.legal_actions)))
        scores = exact_actions(state)
        best, actions = solve(state)
        assert max(scores.values()) == pytest.approx(best)
        assert all(scores[action] == pytest.approx(best) for action in actions)
        record = position_record(state, "test")
        assert restore_position(record) == state
        assert record["best_outcome"] == np.sign(best)


def test_diagnostics_reject_oversized_or_finished_positions():
    with pytest.raises(ValueError):
        position_record(State.new(5, 5), "large")
    state = State.new(1, 1)
    for action in range(4):
        state = state.play(action)
    with pytest.raises(ValueError):
        position_record(state, "finished")


def test_unopened_chain_fixture_is_reachable_and_has_no_capture():
    state = component_position((3, 5), paths=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 14, 13, 12, 11, 10]])
    assert len(state.legal_actions) == 17
    assert state.scores == (0, 0)
    assert not any(state.captures(a) for a in state.legal_actions)
