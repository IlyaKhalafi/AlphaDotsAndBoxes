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


@pytest.mark.parametrize("size", [(1, 2), (2, 3), (4, 4), (5, 5), (10, 10), (12, 12), (3, 7)])
@pytest.mark.parametrize("use_oracle", [False, True])
def test_compiled_playouts_preserve_every_action_and_finish(size, use_oracle):
    compiled = pytest.importorskip("alphaboxes._rollout").playout
    rng = np.random.default_rng(959)
    state = State.new(*size)
    for moves in (0, state.board.num_edges // 3, state.board.num_edges // 2):
        while state.board.num_edges - len(state.legal_actions) < moves:
            state = state.play(int(rng.choice(state.legal_actions)))
        paths, results = [], []
        for implementation in (playout, compiled):
            actions = []
            oracle = DegreeTwoSolver() if use_oracle else None
            results.append(
                implementation(state, np.random.default_rng(957), oracle, actions.append)
            )
            paths.append(actions)
        assert results[0] == results[1]
        assert paths[0] == paths[1]


def test_compiled_evaluator_preserves_predictions_including_cache_misses():
    pytest.importorskip("alphaboxes._rollout")
    first = RolloutEvaluator(Neural(), rollouts=2, cache_size=0)
    second = RolloutEvaluator(Neural(), rollouts=2, cache_size=0, backend="cython")
    rng = np.random.default_rng(972)
    state = State.new(5, 5)
    for _ in range(16):
        policy, value = first(state)
        other_policy, other_value = second(state)
        np.testing.assert_array_equal(policy, other_policy)
        assert value == other_value
        state = state.play(int(rng.choice(state.legal_actions)))
    assert first.statistics() == second.statistics()


def test_late_rollouts_preserve_early_neural_predictions():
    evaluator = RolloutEvaluator(Neural(), max_remaining_fraction=0.5)
    state = State.new(3, 3)
    assert evaluator(state)[1] == -0.25 and evaluator.episodes == 0
    for action in range(state.board.num_edges // 2):
        state = state.play(action)
    evaluator(state)
    assert evaluator.episodes == 2
