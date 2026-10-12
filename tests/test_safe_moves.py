import numpy as np
import pytest

from alphaboxes.game import State
from alphaboxes.opponents import safe_actions, tactical_action
from alphaboxes.search import MCTS, SearchConfig


def gift_position():
    state = State.new(1, 2)
    for action in state.board.boxes[0][:2]:
        state = state.play(action)
    return state


def test_safe_filter_excludes_gifts_but_preserves_capture_choices():
    state = gift_position()
    safe = safe_actions(state)
    assert safe and len(safe) < len(state.legal_actions)
    for action in safe:
        child = state.play(action)
        assert not any(child.captures(a) for a in child.legal_actions)
    risky = next(a for a in state.legal_actions if a not in safe)
    child = state.play(risky)
    assert any(child.captures(a) for a in child.legal_actions)
    assert safe_actions(child) == child.legal_actions


def test_search_filters_and_renormalizes_an_adversarial_gift_prior():
    state = gift_position()
    safe = safe_actions(state)
    risky = next(a for a in state.legal_actions if a not in safe)

    def evaluator(position):
        policy = np.zeros(position.board.num_edges)
        policy[risky] = 1
        return policy, 0.0

    ordinary = MCTS(evaluator, SearchConfig(simulations=8))
    filtered = MCTS(evaluator, SearchConfig(simulations=8, safe_moves=True))
    assert ordinary.policy(state)[0].argmax() == risky
    policy, _ = filtered.policy(state)
    assert policy[risky] == 0
    assert policy.argmax() in safe and np.isclose(policy.sum(), 1)


def test_heuristic_filter_cannot_create_proof_labels():
    with pytest.raises(ValueError, match="cannot prove"):
        SearchConfig(safe_moves=True, prove_results=True)


def test_shared_helper_preserves_seeded_tactical_baseline():
    trajectory = np.random.default_rng(74)
    old_rng, new_rng = np.random.default_rng(17), np.random.default_rng(17)
    for shape in ((1, 2), (3, 3), (5, 5), (10, 10)):
        state = State.new(*shape)
        while not state.terminal:
            legal = state.legal_actions
            captures = [len(state.captures(a)) for a in legal]
            if max(captures):
                choices = [
                    a for a, count in zip(legal, captures, strict=True) if count == max(captures)
                ]
            else:
                choices = [
                    a
                    for a in legal
                    if all(
                        sum(state.edges[e] < 0 for e in state.board.boxes[b]) != 2
                        for b in state.board.edge_boxes[a]
                    )
                ]
            assert tactical_action(state, new_rng) == int(old_rng.choice(choices or legal))
            state = state.play(int(trajectory.choice(legal)))


def test_empty_and_no_safe_positions_return_all_legal_moves():
    state = State.new(1, 1)
    state = state.play(0).play(1)
    assert safe_actions(state) == state.legal_actions
    state = state.play(2).play(3)
    assert safe_actions(state) == ()
