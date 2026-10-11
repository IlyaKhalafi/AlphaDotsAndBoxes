import numpy as np
import pytest

from alphaboxes.engine import AlphaBeta, EngineConfig
from alphaboxes.game import State
from alphaboxes.opponents import solve


def uniform(state):
    return np.ones(state.board.num_edges), 0.0


def test_full_depth_matches_oracle_with_and_without_transpositions():
    rng = np.random.default_rng(3500)
    for _ in range(16):
        state = State.new(2, 2)
        while len(state.legal_actions) > 7:
            state = state.play(int(rng.choice(state.legal_actions)))
        expected = np.sign(solve(state)[0])
        for table_size in (0, 4096):
            engine = AlphaBeta(
                uniform, EngineConfig(time_limit=10, exact_threshold=0, table_size=table_size)
            )
            policy, value = engine.policy(state)
            assert engine.depth == 7
            assert value == expected
            assert policy.sum() == 1
            action = int(policy.argmax())
            assert action in state.legal_actions
            child = state.play(action)
            child_value = (
                np.sign(solve(child)[0]) if not child.terminal else child.outcome(child.player)
            )
            assert (child_value if child.player == state.player else -child_value) == expected


def test_capture_keeps_perspective_and_engine_prunes_and_reuses():
    capturing = State.new(1, 2)
    for action in (0, 1, 2, 3, 4, 6):
        capturing = capturing.play(action)
    engine = AlphaBeta(uniform, EngineConfig(time_limit=10, exact_threshold=0))
    policy, value = engine.policy(capturing)
    assert policy[5] == value == 1
    state = State.new(1, 2)
    policy, value = engine.policy(state)
    assert value == np.sign(solve(state)[0])
    assert engine.cutoffs > 0 and engine.table_hits > 0
    assert len(engine.table) <= engine.config.table_size
    engine.clear_tree()
    assert not engine.table


def test_timeout_retains_only_completed_iteration(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("alphaboxes.engine.monotonic", lambda: clock[0])

    def evaluator(state):
        clock[0] += 0.1
        # An incomplete second iteration must not overwrite depth one's result.
        return np.arange(state.board.num_edges, dtype=float), state.legal_actions[0] / 10

    state = State.new(1, 2)
    expected = AlphaBeta(
        evaluator, EngineConfig(time_limit=10, max_depth=1, exact_threshold=0)
    ).policy(state)
    clock[0] = 0
    engine = AlphaBeta(evaluator, EngineConfig(time_limit=1.1, exact_threshold=0))
    actual = engine.policy(state)
    assert engine.depth == 1
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]


def test_expired_initial_evaluation_returns_legal_prior_fallback(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("alphaboxes.engine.monotonic", lambda: clock[0])

    def slow(state):
        clock[0] += 1
        return np.arange(state.board.num_edges), 0.4

    state = State.new(2, 2).play(11)
    engine = AlphaBeta(slow, EngineConfig(time_limit=0.1, exact_threshold=0))
    policy, value = engine.policy(state)
    assert engine.depth == 0 and value == 0.4
    assert policy.argmax() == 10


def test_terminal_rejected_and_invalid_limits():
    state = State.new(1, 1)
    for action in range(4):
        state = state.play(action)
    with pytest.raises(ValueError):
        AlphaBeta(uniform).policy(state)
    for seconds in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            EngineConfig(time_limit=seconds)


def test_exact_root_uses_the_existing_margin_solver():
    state = State.new(1, 2)
    policy, value = AlphaBeta(uniform).policy(state)
    margin, actions = solve(state)
    assert set(np.flatnonzero(policy)) == set(actions)
    assert value == np.sign(margin)
