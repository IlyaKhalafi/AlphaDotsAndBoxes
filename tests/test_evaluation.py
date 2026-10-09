import pytest

from alphaboxes.evaluation import wilson
from alphaboxes.game import State
from alphaboxes.opponents import endgame_action, solve


def test_wilson_includes_observed_win_rate():
    for wins in [0, 5, 10]:
        lower, upper = wilson(wins, 10)
        assert 0 <= lower <= wins / 10 <= upper <= 1
    assert wilson(10, 10)[0] < 1


def test_endgame_baseline_is_optimal():
    import numpy as np

    state = State.new(2, 2)
    assert endgame_action(state, np.random.default_rng(6)) in solve(state)[1]


@pytest.mark.parametrize("rows,cols", [(0, 2), (-1, 3)])
def test_invalid_board_size(rows, cols):
    with pytest.raises(ValueError):
        State.new(rows, cols)
