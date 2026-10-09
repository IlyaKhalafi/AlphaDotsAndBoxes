from functools import cache

import numpy as np
import pytest

from alphaboxes.game import State, board
from alphaboxes.graph import encode, spaces
from alphaboxes.opponents import solve


def test_geometry():
    for rows, cols in [(1, 1), (2, 3), (7, 4)]:
        b = board(rows, cols)
        assert b.num_edges == (rows + 1) * cols + rows * (cols + 1)
        assert all(len(edges) == 4 for edges in b.boxes)
        assert sum(map(len, b.edge_boxes)) == 4 * rows * cols


def test_capture_keeps_turn_and_scores():
    s = State.new(1, 1)
    for action in [0, 1, 2]:
        s = s.play(action)
    assert s.player == 1
    s = s.play(3)
    assert s.player == 1
    assert s.scores == (0, 1)
    assert s.outcome(0) == -1
    with pytest.raises(ValueError):
        s.play(0)


def test_double_capture():
    s = State.new(1, 2)
    for action in [0, 1, 2, 3, 4, 6]:
        s = s.play(action)
    assert s.captures(5) == (0, 1)
    assert s.play(5).scores == (2, 0)


def test_illegal_and_immutable():
    initial = State.new()
    after = initial.play(0)
    assert initial.edges[0] == -1
    for action in [-1, 1000, 0]:
        with pytest.raises(ValueError):
            after.play(action)


def test_graph_and_padding():
    state = State.new(2, 3).play(0)
    obs = encode(state, 50)
    observation_space, _ = spaces(50)
    assert observation_space.contains(obs)
    assert obs["action_mask"].sum() == len(state.legal_actions)
    assert obs["adjacency"].sum() == state.board.num_boxes * 8
    assert np.array_equal(obs["adjacency"], obs["adjacency"].T)
    assert not obs["x"][state.board.num_nodes :].any()


def test_exact_oracle():
    value, moves = solve(State.new(1, 1))
    assert value == -1  # player drawing the fourth edge wins
    assert set(moves) == {0, 1, 2, 3}
    assert solve(State.new(1, 2))[0] == 0


def test_random_games_conserve_boxes():
    rng = np.random.default_rng(17)
    for size in [(1, 2), (2, 3), (4, 4), (5, 5)]:
        s = State.new(*size)
        for _ in range(s.board.num_edges):
            before = s
            action = int(rng.choice(s.legal_actions))
            captured = len(s.captures(action))
            s = s.play(action)
            assert sum(s.scores) - sum(before.scores) == captured
            assert (s.player == before.player) == (captured > 0)
        assert s.terminal
        assert sum(s.scores) == s.board.num_boxes


def test_bitset_oracle_matches_rules_engine_minimax():
    """An independent engine-based reference catches scoring/turn/cache mistakes."""

    @cache
    def reference(state):
        if state.terminal:
            scores = state.scores
            return scores[state.player] - scores[1 - state.player]
        values = []
        for action in state.legal_actions:
            child = state.play(action)
            # The engine keeps edge authors for UI; minimax needs only occupancy.
            child = State(
                child.board,
                tuple(0 if e >= 0 else -1 for e in child.edges),
                child.owners,
                child.player,
            )
            value = reference(child)
            values.append(value if state.player == child.player else -value)
        return max(values)

    rng = np.random.default_rng(71)
    for size in [(1, 1), (1, 2), (2, 2)]:
        state = State.new(*size)
        while not state.terminal:
            margin, optimal = solve(state)
            expected = reference(state) / state.board.num_boxes
            assert margin == expected
            for action in optimal:
                child = state.play(action)
                score = reference(child) / state.board.num_boxes
                assert (score if child.player == state.player else -score) == margin
            state = state.play(int(rng.choice(state.legal_actions)))
