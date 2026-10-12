from dataclasses import replace

import numpy as np
import pytest

from alphaboxes.game import State
from alphaboxes.search import MCTS, Node, SearchConfig


@pytest.mark.parametrize("shape", [(3, 4), (5, 5), (10, 10)])
@pytest.mark.parametrize("reduction", [0.0, 0.33])
@pytest.mark.parametrize("explore", [False, True])
def test_cached_selection_preserves_search_and_reused_noisy_root(
    shape, reduction, explore, monkeypatch
):
    monkeypatch.setattr("alphaboxes.search.choose_child", None)
    def evaluator(state):
        policy = np.linspace(1, 3, state.board.num_edges, dtype=np.float64)
        policy /= policy.sum()
        value = np.tanh((state.scores[state.player] - state.scores[1 - state.player]) / 5)
        return policy, float(value)

    config = SearchConfig(simulations=96, fpu_reduction=reduction, reuse_tree=True)
    roots = [[], []]
    planners = [
        MCTS(evaluator, config, seed=919, root_observer=roots[0].append),
        MCTS(
            evaluator,
            replace(config, sparse_selection=True),
            seed=919,
            root_observer=roots[1].append,
        ),
    ]
    state = State.new(*shape)
    for _ in range(3):
        predictions = [planner.policy(state, explore=explore) for planner in planners]
        np.testing.assert_array_equal(predictions[0][0], predictions[1][0])
        assert predictions[0][1] == predictions[1][1]
        assert planners[0].inherited_visits == planners[1].inherited_visits
        assert [child.visits for child in roots[0][-1].children.values()] == [
            child.visits for child in roots[1][-1].children.values()
        ]
        assert any(node.selection is not None for node in roots[1][-1].children.values()) or (
            roots[1][-1].selection is not None
        )
        state = state.play(int(predictions[0][0].argmax()))


def test_rounded_prior_ties_keep_original_dictionary_order():
    state = State.new(3, 4)
    node = Node(state, evaluation=0.75, player=0)
    node.children = {
        30 - index: Node(state, prior=1e-30 * (index + 1), player=index % 2) for index in range(16)
    }
    original = MCTS(None, SearchConfig(fpu_reduction=0.33))
    cached = MCTS(None, SearchConfig(fpu_reduction=0.33, sparse_selection=True))
    for _ in range(20):
        expected = original._select(node, 1.5)
        actual = cached._select(node, 1.5)
        assert actual is expected is node.children[30]
        actual.visits += 1
        actual.value_sum += 0.42


def test_cached_selection_tracks_backups_with_mixed_players_and_zero_visits():
    rng = np.random.default_rng(920)
    state = State.new(5, 5)
    node = Node(state, evaluation=-0.2, player=1)
    node.children = {
        index: Node(
            state,
            prior=float(rng.random()),
            visits=int(rng.integers(4)),
            value_sum=float(rng.uniform(-1, 1)),
            player=int(rng.integers(2)),
        )
        for index in range(60)
    }
    original = MCTS(None, SearchConfig(fpu_reduction=0.33))
    cached = MCTS(None, SearchConfig(fpu_reduction=0.33, sparse_selection=True))
    for index in range(400):
        scale = 1.5 * np.sqrt(index + 1)
        expected = original._select(node, scale)
        actual = cached._select(node, scale)
        assert actual is expected
        if index % 7:
            actual.visits += 1
            actual.value_sum += float(rng.uniform(-1, 1))
