import numpy as np

from alphaboxes.game import State
from alphaboxes.opponents import solve
from alphaboxes.search import MCTS, SearchConfig, perspective


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
