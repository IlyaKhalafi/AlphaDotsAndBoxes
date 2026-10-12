import numpy as np
import pytest

import alphaboxes.opponents as opponents
import alphaboxes.search as search
from alphaboxes.game import State

accelerator = pytest.importorskip("alphaboxes._endgame")


def test_compiled_solver_matches_python_across_sizes_and_extra_turns(monkeypatch):
    rng = np.random.default_rng(901)
    compiled = accelerator.solve_remaining
    for size in ((1, 1), (1, 2), (2, 3), (4, 4), (5, 5), (6, 6), (2, 7)):
        for _ in range(3):
            state = State.new(*size)
            while len(state.legal_actions) > 8:
                state = state.play(int(rng.choice(state.legal_actions)))
            monkeypatch.setattr(opponents, "solve_remaining", None)
            expected = opponents.solve(state)
            monkeypatch.setattr(opponents, "solve_remaining", compiled)
            assert opponents.solve(state) == expected


def test_compiled_solver_rejects_invalid_buffers():
    masks = np.array([1], dtype=np.uint32)
    with pytest.raises(ValueError, match="Neighbor index"):
        accelerator.solve_remaining(masks, np.array([[2, -1]], dtype=np.int32))
    with pytest.raises(ValueError, match="incident edge"):
        accelerator.solve_remaining(
            np.array([0], dtype=np.uint32), np.array([[0, -1]], dtype=np.int32)
        )
    with pytest.raises(ValueError, match="at most 18"):
        accelerator.solve_remaining(masks, np.zeros((19, 2), dtype=np.int32))


def test_compiled_child_scoring_preserves_ties_policies_and_turns(monkeypatch):
    compiled = pytest.importorskip("alphaboxes._search").choose_child
    rng = np.random.default_rng(771)
    for size in ((1, 2), (2, 3), (4, 4), (5, 5)):
        state = State.new(*size)
        for _ in range(state.board.num_edges // 2):
            state = state.play(int(rng.choice(state.legal_actions)))

        def evaluator(state):
            return np.arange(1, state.board.num_edges + 1), 0.25

        config = search.SearchConfig(simulations=128)
        monkeypatch.setattr(search, "choose_child", None)
        reference, expected = search.MCTS(evaluator, config, 11).policy(state, explore=True)
        monkeypatch.setattr(search, "choose_child", compiled)
        actual, value = search.MCTS(evaluator, config, 11).policy(state, explore=True)
        np.testing.assert_array_equal(actual, reference)
        assert value == expected


@pytest.mark.parametrize(
    "size", [(1, 1), (1, 4), (2, 3), (4, 4), (5, 5), (10, 10), (12, 12), (3, 7)]
)
def test_compiled_component_features_match_python_through_complete_games(size):
    from dataclasses import replace

    from alphaboxes.components import python_structural_features
    from alphaboxes.curriculum import sample_endgame

    compiled = pytest.importorskip("alphaboxes._components").structural_features
    rng = np.random.default_rng(922)
    for state in (State.new(*size), sample_endgame(size, rng)):
        while True:
            for position in (state, replace(state, player=1 - state.player)):
                expected, actual = python_structural_features(position), compiled(position)
                for first, second in zip(expected, actual, strict=True):
                    np.testing.assert_array_equal(first, second)
            if state.terminal:
                break
            state = state.play(int(rng.choice(state.legal_actions)))
