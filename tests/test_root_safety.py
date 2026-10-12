import numpy as np
import pytest

from alphaboxes.game import State
from alphaboxes.opponents import safe_actions
from alphaboxes.search import MCTS, SearchConfig


def position(rows, cols):
    state = State.new(rows, cols)
    for action in state.board.boxes[0][:2]:
        state = state.play(action)
    return state


def evaluator_for(risky):
    def evaluate(state):
        policy = np.zeros(state.board.num_edges)
        legal = state.legal_actions
        policy[list(legal)] = 1 / len(legal)
        if risky in legal and len(legal) > 1:
            policy[list(legal)] = 0.1 / (len(legal) - 1)
            policy[risky] = 0.9
        return policy, 0.0

    return evaluate


def test_root_bias_preserves_all_actions_and_internal_priors():
    state = position(4, 4)
    safe = safe_actions(state)
    risky = next(a for a in state.legal_actions if a not in safe)
    roots = []
    search = MCTS(
        evaluator_for(risky),
        SearchConfig(simulations=8, root_safety_scale=0.01),
        root_observer=roots.append,
    )
    policy, _ = search.policy(state)
    root = roots[-1]
    assert set(root.children) == set(state.legal_actions)
    assert len(safe) == 36 and len(state.legal_actions) == 38
    # Both gifting edges are downweighted, including the low-prior second edge.
    expected_risky = 0.009 / (0.009 + 0.001 / 37 + 3.6 / 37)
    assert root.children[risky].prior == pytest.approx(expected_risky)
    assert root.children[risky].prior > 0
    assert sum(child.prior for child in root.children.values()) == pytest.approx(1)
    assert np.isclose(policy.sum(), 1)
    assert policy.argmax() in state.legal_actions
    expanded = next(child for child in root.children.values() if child.children)
    expected, _ = evaluator_for(risky)(expanded.state)
    assert set(expanded.children) == set(expanded.state.legal_actions)
    for action, child in expanded.children.items():
        assert child.prior == pytest.approx(expected[action])


def test_late_sacrifice_priors_remain_unchanged():
    state = position(1, 2)
    safe = safe_actions(state)
    assert len(safe) < 8
    risky = next(a for a in state.legal_actions if a not in safe)
    roots = []
    search = MCTS(
        evaluator_for(risky),
        SearchConfig(simulations=4, root_safety_scale=0.01),
        root_observer=roots.append,
    )
    search.policy(state)
    assert roots[-1].children[risky].prior == pytest.approx(0.9)


def test_reused_root_does_not_accumulate_the_prior_penalty():
    state = position(4, 4)
    risky = next(a for a in state.legal_actions if a not in safe_actions(state))
    roots = []
    search = MCTS(
        evaluator_for(risky),
        SearchConfig(simulations=4, root_safety_scale=0.01, reuse_tree=True),
        root_observer=roots.append,
    )
    search.policy(state)
    search.policy(state)
    assert roots[0] is roots[1]
    assert roots[-1].children[risky].prior == pytest.approx(0.009 / (0.009 + 0.001 / 37 + 3.6 / 37))


def test_open_captures_preserve_original_priors():
    state = position(4, 4)
    risky = next(a for a in state.legal_actions if a not in safe_actions(state))
    state = state.play(risky)
    capture = next(a for a in state.legal_actions if state.captures(a))
    roots = []
    search = MCTS(
        evaluator_for(capture),
        SearchConfig(simulations=4, root_safety_scale=0.01),
        root_observer=roots.append,
    )
    search.policy(state)
    assert roots[-1].children[capture].prior == pytest.approx(0.9)


def test_cached_selector_preserves_the_biased_search_result():
    state = position(4, 4)
    risky = next(a for a in state.legal_actions if a not in safe_actions(state))
    results = [
        MCTS(
            evaluator_for(risky),
            SearchConfig(
                simulations=128, fpu_reduction=0.33, root_safety_scale=0.01, sparse_selection=cached
            ),
        ).policy(state)
        for cached in (False, True)
    ]
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


@pytest.mark.parametrize("scale", [0, -1, 1.01, float("nan"), float("inf")])
def test_invalid_root_safety_scale(scale):
    with pytest.raises(ValueError, match="safety scale"):
        SearchConfig(root_safety_scale=scale)
