import numpy as np

from alphaboxes.chains import chain_action, components
from alphaboxes.game import State
from alphaboxes.opponents import solve


def horizontal_chains(rows=3, cols=5):
    state = State.new(rows, cols)
    for action in range((rows + 1) * cols):
        state = state.play(action)
    return state


def test_chain_control_hands_back_two_boxes():
    state = horizontal_chains()
    first_vertical = (state.board.rows + 1) * state.board.cols
    for action in range(first_vertical, first_vertical + 4):
        state = state.play(action)
    assert len(state.legal_actions) == 14
    action = chain_action(state, np.random.default_rng(3))
    assert action == first_vertical + 5  # Give the last pair, retaining control.
    assert not state.captures(action)
    assert action in solve(state)[1]
    before = state.scores
    child = state.play(action).play(first_vertical + 4)
    assert child.scores[1 - state.player] == before[1 - state.player] + 2


def test_chain_control_takes_last_component():
    state = horizontal_chains(1, 5)
    for action in range(10, 14):
        state = state.play(action)
    action = chain_action(state, np.random.default_rng(3), exact_threshold=0)
    assert state.captures(action)


def test_loop_control_hands_back_four_boxes():
    state = State.new(4, 4)
    group_for_box = {b: (b // 4 // 2, b % 4 // 2) for b in range(16)}
    loop_edges = {
        e
        for e, boxes in enumerate(state.board.edge_boxes)
        if len(boxes) == 2 and group_for_box[boxes[0]] == group_for_box[boxes[1]]
    }
    for action in range(state.board.num_edges):
        if action not in loop_edges:
            state = state.play(action)
    assert len(components(state)) == 4
    state = state.play(min(loop_edges))
    assert len(state.legal_actions) == 15
    action = chain_action(state, np.random.default_rng(2))
    assert not state.captures(action)
    assert action in solve(state)[1]
    next_state = state.play(action)
    player = next_state.player
    before = next_state.scores[player]
    for _ in range(2):
        capture = next(a for a in next_state.legal_actions if len(next_state.captures(a)) == 2)
        next_state = next_state.play(capture)
        assert next_state.player == player
    assert next_state.scores[player] == before + 4


def test_chain_opponent_finishes_legal_games():
    rng = np.random.default_rng(91)
    for size in [(1, 2), (3, 3), (5, 5)]:
        state = State.new(*size)
        while not state.terminal:
            action = chain_action(state, rng)
            assert action in state.legal_actions
            state = state.play(action)
        assert sum(state.scores) == state.board.num_boxes
