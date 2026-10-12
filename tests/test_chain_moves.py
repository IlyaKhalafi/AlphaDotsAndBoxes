from dataclasses import replace

import numpy as np

from alphaboxes.chain_moves import allowed_actions
from alphaboxes.game import State
from alphaboxes.opponents import solve


def test_every_two_by_two_reduction_preserves_an_exact_best_margin_move():
    layout = State.new(2, 2)
    reduced = 0
    for mask in range(1 << layout.board.num_edges):
        edges = tuple(0 if mask & (1 << e) else -1 for e in range(layout.board.num_edges))
        owners = tuple(
            b % 2 if all(edges[e] >= 0 for e in box) else -1
            for b, box in enumerate(layout.board.boxes)
        )
        state = replace(layout, edges=edges, owners=owners)
        if state.terminal:
            continue
        allowed = allowed_actions(state)
        assert allowed and set(allowed) <= set(state.legal_actions)
        if len(allowed) < len(state.legal_actions):
            reduced += 1
            assert set(allowed) & set(solve(state)[1])
    assert reduced > 100


def test_chain_reductions_at_junctions_preserve_exact_moves_on_larger_boards():
    rng = np.random.default_rng(1007)
    reductions = 0
    for size in ((3, 3), (4, 4), (5, 5)):
        for _ in range(48):
            state = State.new(*size)
            while len(state.legal_actions) > 13:
                state = state.play(int(rng.choice(state.legal_actions)))
            allowed = allowed_actions(state)
            if len(allowed) < len(state.legal_actions):
                reductions += 1
                assert set(allowed) & set(solve(state)[1])
    assert reductions > 10


def test_search_evaluates_both_handout_choices_even_with_a_tiny_prior():
    from alphaboxes.search import MCTS, SearchConfig

    state = State.new(2, 3)
    for edge, (r1, _, r2, _) in enumerate(state.board.edges):
        if r1 == r2:
            state = state.play(edge)
    state = state.play(state.board.boxes[0][2]).play(state.board.boxes[0][3])
    options = allowed_actions(state)
    assert len(options) == 2 and len(state.legal_actions) > 2

    def evaluator(position):
        policy = np.full(position.board.num_edges, 1e-12, np.float32)
        policy[options[0]] = 1
        return policy, 0.5

    observed = []
    planner = MCTS(
        evaluator, SearchConfig(simulations=2, chain_pruning=True), root_observer=observed.append
    )
    policy, _ = planner.policy(state)
    assert set(observed[0].children) == set(options)
    assert all(child.visits == 1 for child in observed[0].children.values())
    assert set(np.flatnonzero(policy)) == set(options)
