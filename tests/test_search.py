import numpy as np

from alphaboxes.game import State
from alphaboxes.opponents import solve
from alphaboxes.search import MCTS, CachedEvaluator, SearchConfig, perspective


def uniform(state):
    return np.ones(state.board.num_edges), 0.0


def test_backups_keep_sign_on_extra_turn():
    assert perspective(0.8, 0, 0) == 0.8
    assert perspective(0.8, 0, 1) == -0.8
    state = State.new(1, 2)
    for action in [0, 1, 2, 3, 4, 6]:
        state = state.play(action)
    policy, value = MCTS(uniform, SearchConfig(simulations=8)).policy(state)
    assert policy[5] == 1
    assert value == 1


def test_search_matches_exact_on_near_terminal_positions():
    rng = np.random.default_rng(4)
    for _ in range(12):
        state = State.new(1, 2)
        while len(state.legal_actions) > 4:
            state = state.play(int(rng.choice(state.legal_actions)))
        _, optimal = solve(state)
        action = MCTS(uniform, SearchConfig(simulations=256)).action(state)
        assert action in optimal


def test_policy_is_legal_and_normalized():
    state = State.new(2, 3).play(1)
    policy, value = MCTS(uniform, SearchConfig(simulations=20)).policy(state, explore=True)
    assert np.isclose(policy.sum(), 1)
    assert policy[1] == 0
    assert -1 <= value <= 1


def test_exact_endgame():
    s = State.new(1, 2)
    policy, value = MCTS(uniform, SearchConfig(exact_threshold=7)).policy(s)
    assert set(np.flatnonzero(policy)) == set(solve(s)[1])
    assert value == 0


def test_batched_search_matches_independent_search_with_extra_turns():
    capturing = State.new(1, 2)
    for action in (0, 1, 2, 3, 4, 6):
        capturing = capturing.play(action)
    states = [State.new(2, 2), capturing, State.new(1, 1).play(0)]

    def evaluator(state):
        return np.arange(1, state.board.num_edges + 1), (state.scores[state.player] - 1) / 3

    config = SearchConfig(simulations=64)
    batched = MCTS(evaluator, config).policies(states)
    for state, (policy, value) in zip(states, batched, strict=True):
        expected, expected_value = MCTS(evaluator, config).policy(state)
        np.testing.assert_array_equal(policy, expected)
        assert value == expected_value


def test_batched_cache_deduplicates_misses_and_survives_eviction():
    class RecordingEvaluator(CachedEvaluator):
        def _predict(self, state):
            return uniform(state)

        def _predict_many(self, states):
            self.batch_count = len(states)
            return super()._predict_many(states)

    evaluator = RecordingEvaluator(cache_size=1)
    first, second = State.new(1, 1), State.new(1, 2)
    predictions = evaluator.evaluate_many([first, second, first])
    assert evaluator.batch_count == 2
    assert predictions[0] is predictions[2]
    assert len(predictions[1][0]) == second.board.num_edges
    assert all(not policy.flags.writeable for policy, _ in predictions)


def test_search_only_constructs_states_for_visited_moves(monkeypatch):
    original = State.play
    moves = []

    def counted(state, action):
        moves.append(action)
        return original(state, action)

    monkeypatch.setattr(State, "play", counted)
    MCTS(uniform, SearchConfig(simulations=32)).policy(State.new(5, 5))
    assert len(moves) <= 32


def test_solved_leaves_do_not_use_neural_values_and_keep_capture_perspective():
    state = State.new(1, 2)
    calls = []

    def wrong_value(position):
        calls.append(len(position.legal_actions))
        return np.ones(position.board.num_edges), -1.0

    config = SearchConfig(simulations=256, leaf_exact_threshold=5)
    search = MCTS(wrong_value, config)
    action = search.action(state)
    assert action in solve(state)[1]
    assert calls and min(calls) > 5
    before = len(search._solved)
    search.action(state)
    assert len(search._solved) == before
    assert all(-1 <= value <= 1 for value in search._solved.values())


def test_leaf_solver_covers_root_when_root_aid_is_disabled():
    def forbidden(state):
        raise AssertionError("Solved root should not call the network.")

    state = State.new(1, 1).play(0)
    policy, value = MCTS(forbidden, SearchConfig(leaf_exact_threshold=3)).policy(state)
    margin, optimal = solve(state)
    assert set(np.flatnonzero(policy)) == set(optimal)
    assert value == np.sign(margin)
