import numpy as np
import pytest

from alphaboxes.game import State
from alphaboxes.search import MCTS, SearchConfig


def prediction(state):
    policy = np.zeros(state.board.num_edges, np.float32)
    legal = list(state.legal_actions)
    policy[legal] = np.arange(1, len(legal) + 1, dtype=np.float32)
    policy /= policy.sum()
    value = (state.scores[state.player] - state.scores[1 - state.player]) / state.board.num_boxes
    return policy, value


@pytest.mark.parametrize("mode", ["constant", "policy_mass", "branching"])
@pytest.mark.parametrize("cached", [False, True])
def test_scalar_sqrt_preserves_complete_search_trace(monkeypatch, mode, cached):
    from math import sqrt

    roots = []
    state = State.new(5, 5).play(0).play(1).play(2)
    config = SearchConfig(
        simulations=256, fpu_reduction=0.33, fpu_mode=mode, sparse_selection=cached
    )
    for function in (np.sqrt, sqrt):
        monkeypatch.setattr("alphaboxes.search.sqrt", function)
        root = []
        planner = MCTS(prediction, config, root_observer=root.append)
        policy, value = planner.policy(state)
        roots.append((policy, value, [(c.visits, c.value_sum) for c in root[0].children.values()]))
    np.testing.assert_array_equal(roots[0][0], roots[1][0])
    assert roots[0][1:] == roots[1][1:]
