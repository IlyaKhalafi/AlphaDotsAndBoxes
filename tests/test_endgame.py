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
        b % 2 if all(edges[e] >= 0 for e in box) else -1 for b, box in enumerate(state.board.boxes)
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
        fast = solver.fast_value(state)
        if fast is not None:
            assert fast == pytest.approx(solve(state)[0])
        for action in state.legal_actions:
            child = state.play(action)
            expected = solve(child)[0]
            if child.player != state.player:
                expected = -expected
            assert solver.action_value(state, action) == pytest.approx(expected)
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


def test_every_descendant_of_mixed_chain_and_loop_matches_edge_oracle():
    state = State.new(3, 3)
    chain = (2, 5, 8, 7, 6)
    connections = {(0, 1), (0, 3), (1, 4), (3, 4)}
    connections.update(tuple(sorted(pair)) for pair in zip(chain, chain[1:], strict=False))
    free = {
        edge for edge, boxes in enumerate(state.board.edge_boxes) if tuple(boxes) in connections
    }
    free.update((state.board.boxes[2][0], state.board.boxes[6][1]))
    assert len(free) == 10
    base = sum(1 << edge for edge in range(state.board.num_edges) if edge not in free)
    free = sorted(free)
    solver = DegreeTwoSolver()
    for subset in range(1 << len(free)):
        mask = base | sum(1 << edge for bit, edge in enumerate(free) if subset & (1 << bit))
        current = occupied_state(3, 3, mask, subset % 2)
        assert solver.solve(current) == solve(current)


def test_partially_captured_chain_can_require_a_two_box_handout():
    state = State.new(2, 3)
    for edge, (r1, _, r2, _) in enumerate(state.board.edges):
        if r1 == r2:
            state = state.play(edge)
    state = state.play(state.board.boxes[0][2])
    state = state.play(state.board.boxes[0][3])
    assert position(state) == ((2, 1), (3, 2))
    result = DegreeTwoSolver().solve(state)
    assert result == solve(state)
    assert result[0] == pytest.approx(1 / 3)
    assert all(not state.captures(action) for action in result[1])


def test_twenty_five_closed_loops_use_component_counts():
    state = State.new(10, 10)
    connections = set()
    for row in range(0, 10, 2):
        for col in range(0, 10, 2):
            a = row * 10 + col
            connections.update(((a, a + 1), (a, a + 10), (a + 1, a + 11), (a + 10, a + 11)))
    free = {
        edge for edge, boxes in enumerate(state.board.edge_boxes) if tuple(boxes) in connections
    }
    mask = sum(1 << edge for edge in range(state.board.num_edges) if edge not in free)
    state = occupied_state(10, 10, mask)
    assert position(state) == ((4, -1),) * 25
    solver = DegreeTwoSolver()
    value, actions = solver.solve(state)
    assert value == pytest.approx(-0.04)
    assert actions == state.legal_actions
    # One state per closed-loop count, plus opened loops: no edge-subset search.
    assert solver.nodes <= 50


def test_large_closed_chain_openings_skip_two_front_enumeration():
    state = State.new(10, 10)
    for edge, (r1, _, r2, _) in enumerate(state.board.edges):
        if r1 == r2:
            state = state.play(edge)
    solver = DegreeTwoSolver()
    expected, actions = solver.solve(state)
    assert expected == pytest.approx(-0.64)
    assert actions == state.legal_actions
    for action in actions:
        assert solver.action_value(state, action) == pytest.approx(expected)
    assert solver.nodes <= 25
