from dataclasses import replace

import numpy as np
import pytest

from alphaboxes.endgame import DegreeTwoSolver, EndgameTimeout, position
from alphaboxes.game import State
from alphaboxes.opponents import solve


def occupied_state(rows, cols, occupied, player=0):
    state = State.new(rows, cols)
    edges = tuple(0 if occupied & (1 << e) else -1 for e in range(state.board.num_edges))
    owners = tuple(
        b % 2 if all(edges[e] >= 0 for e in box) else -1
        for b, box in enumerate(state.board.boxes)
    )
    return replace(state, edges=edges, owners=owners, player=player)


def test_all_recognized_two_by_two_positions_match_full_edge_oracle():
    solver = DegreeTwoSolver()
    recognized = 0
    for mask in range(1 << State.new(2, 2).board.num_edges):
        state = occupied_state(2, 2, mask, mask % 2)
        if position(state) is None or state.terminal:
            continue
        recognized += 1
        assert solver.solve(state) == solve(state)
    assert recognized > 100


def test_three_by_three_random_chains_match_exact_actions():
    rng, solver = np.random.default_rng(4101), DegreeTwoSolver()
    checked = 0
    for _ in range(500):
        state = State.new(3, 3)
        for action in rng.choice(state.board.num_edges, size=16, replace=False):
            state = state.play(int(action))
        if position(state) is not None:
            assert solver.solve(state) == solve(state)
            checked += 1
    assert checked >= 20


def test_long_chains_and_large_cycle_without_edge_subset_enumeration():
    solver = DegreeTwoSolver()
    state = State.new(3, 3)
    for edge, (r1, _, r2, _) in enumerate(state.board.edges):
        if r1 == r2:
            state = state.play(edge)
    assert position(state) == ((3, 2),) * 3
    assert solver.solve(state) == solve(state)
    # A 4-box cycle has no boundary cuts; opening it creates two capture fronts.
    state = State.new(2, 2)
    for edge, boxes in enumerate(state.board.edge_boxes):
        if len(boxes) == 1:
            state = state.play(edge)
    assert position(state) == ((4, -1),)
    assert solver.solve(state) == solve(state)


def test_deadline_does_not_cache_unfinished_results(monkeypatch):
    solver = DegreeTwoSolver()
    state = State.new(1, 3)
    for edge, (r1, _, r2, _) in enumerate(state.board.edges):
        if r1 == r2:
            state = state.play(edge)
    monkeypatch.setattr("alphaboxes.endgame.monotonic", lambda: 2.0)
    with pytest.raises(EndgameTimeout):
        solver.solve(state, deadline=1.0)
    assert not solver.cache
    assert solver.solve(state) == solve(state)
