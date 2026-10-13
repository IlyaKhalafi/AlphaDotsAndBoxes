import numpy as np
import pytest

from alphaboxes.game import State
from alphaboxes.search import MCTS, Node, SearchConfig


@pytest.mark.parametrize("sparse", [False, True])
def test_bounded_first_play_explores_beyond_a_known_losing_branch(sparse):
    state = State.new(4, 4)
    node = Node(state, player=0, evaluation=-0.95)
    losing = Node(state, player=0, prior=0.99, visits=100, value_sum=-100)
    unseen = Node(state, player=0, prior=0.01)
    node.children = {0: losing, 1: unseen}
    node.children.update({i: Node(state, player=0, prior=0) for i in range(2, 16)})
    scale = 1.5 * np.sqrt(101)
    ordinary = MCTS(None, SearchConfig(fpu_reduction=0.33, sparse_selection=sparse))
    bounded = MCTS(None, SearchConfig(fpu_reduction=0.33, fpu_floor=-1, sparse_selection=sparse))
    assert ordinary._select(node, scale) is losing
    assert bounded._select(node, scale) is unseen
    node.evaluation = 0.25
    assert ordinary._select(node, scale) is bounded._select(node, scale)


@pytest.mark.parametrize("floor", [float("nan"), float("inf"), -1.01, 1.01])
def test_first_play_floor_rejects_values_outside_the_outcome_support(floor):
    with pytest.raises(ValueError, match="First-play floor"):
        SearchConfig(fpu_floor=floor)
