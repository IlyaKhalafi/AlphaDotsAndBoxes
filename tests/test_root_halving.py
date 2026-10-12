import numpy as np
import pytest

from alphaboxes.game import State
from alphaboxes.root_halving import RootHalving
from alphaboxes.search import MCTS, Node, SearchConfig


def test_halving_spends_uneven_budget_and_retains_the_best_action():
    values = (-0.8, 0.1, 0.3, 0.9)
    children = {a: Node(None, prior=0.25, action=a) for a in range(4)}
    schedule = RootHalving(children, 37, 4, lambda child: values[child.action])
    for _ in range(37):
        child = schedule.select()
        child.visits += 1
    assert sum(child.visits for child in children.values()) == 37
    assert all(child.visits >= 4 for child in children.values())
    assert children[3].visits > children[0].visits
    assert schedule.best_action() == 3


def test_small_budget_returns_a_tested_action():
    children = {a: Node(None, prior=0.25, action=a) for a in range(4)}
    schedule = RootHalving(children, 3, 16, lambda child: -0.5)
    for _ in range(3):
        schedule.select().visits += 1
    assert sum(child.visits for child in children.values()) == 3
    assert children[schedule.best_action()].visits > 0


def test_one_action_and_a_cold_deadline_have_legal_choices():
    child = Node(None, prior=1.0, action=7)
    schedule = RootHalving({7: child}, 8, 16, lambda _: 0.0)
    assert schedule.best_action() == 7
    for _ in range(8):
        assert schedule.select() is child
        child.visits += 1
    assert child.visits == 8


def test_batched_halving_respects_capture_turns_and_full_simulation_budget():
    def uniform(state):
        policy = np.zeros(state.board.num_edges, np.float32)
        policy[list(state.legal_actions)] = 1 / len(state.legal_actions)
        return policy, 0.0

    state = State.new(1, 1).play(0).play(1)
    capture = state.play(2)
    observed = []
    search = MCTS(
        uniform,
        SearchConfig(simulations=32, root_candidates=16, fpu_reduction=0.33),
        root_observer=observed.append,
    )
    results = search.policies([state, capture])
    assert search.completed_simulations == 32
    assert len(observed) == 2
    assert all(root.visits == 32 for root in observed)
    for board, (policy, _) in zip((state, capture), results, strict=True):
        assert policy.argmax() in board.legal_actions
        assert np.count_nonzero(policy) == 1 and policy.sum() == 1
    assert results[0][1] < 0
    assert results[1][1] == 1
    with pytest.raises(ValueError, match="acting experiment"):
        search.policy(state, explore=True)


@pytest.mark.parametrize("options", [{"reuse_tree": True}, {"prove_results": True}])
def test_halving_rejects_unsupported_tree_modes(options):
    with pytest.raises(ValueError, match="fresh trees"):
        SearchConfig(root_candidates=16, **options)
