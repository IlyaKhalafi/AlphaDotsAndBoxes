from dataclasses import replace

import numpy as np
import pytest

from alphaboxes.endgame import DegreeTwoSolver
from alphaboxes.game import State
from alphaboxes.rollouts import RolloutEvaluator, playout


class Neural:
    def __call__(self, state):
        policy = np.zeros(state.board.num_edges, np.float32)
        policy[list(state.legal_actions)] = 1 / len(state.legal_actions)
        return policy, -0.25

    def clear_cache(self):
        pass


@pytest.mark.parametrize("size", [(1, 2), (2, 3), (4, 4), (5, 5)])
def test_mutable_playouts_replay_legally_with_identical_outcomes(size):
    rng = np.random.default_rng(944)
    for _ in range(3):
        initial = State.new(*size)
        for _ in range(initial.board.num_edges // 3):
            initial = initial.play(int(rng.choice(initial.legal_actions)))
        actions = []
        outcome, steps, exact = playout(initial, rng, on_move=actions.append)
        replay = initial
        for action in actions:
            replay = replay.play(action)
        assert replay.terminal
        assert steps == len(actions) == len(initial.legal_actions)
        assert outcome == replay.outcome(initial.player) and not exact


def test_exact_playout_finish_preserves_current_player_and_scores():
    state = State.new(2, 2)
    for action, boxes in enumerate(state.board.edge_boxes):
        if len(boxes) == 1:
            state = state.play(action)
    value, steps, exact = playout(state, np.random.default_rng(4), DegreeTwoSolver())
    assert (value, steps, exact) == (-1, 0, True)


def test_rollout_cache_misses_and_player_relabeling_are_deterministic():
    evaluator = RolloutEvaluator(Neural(), rollouts=2, weight=0.5, cache_size=0)
    state = State.new(3, 3).play(0).play(1)
    first = evaluator(state)
    second = evaluator(state)
    flipped = replace(
        state,
        edges=tuple(1 - owner if owner >= 0 else -1 for owner in state.edges),
        owners=tuple(1 - owner if owner >= 0 else -1 for owner in state.owners),
        player=1 - state.player,
    )
    third = evaluator(flipped)
    assert first[1] == second[1] == third[1]
    np.testing.assert_array_equal(first[0], second[0])
    assert evaluator.episodes == 6
