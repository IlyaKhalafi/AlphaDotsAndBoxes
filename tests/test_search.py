import numpy as np

from alphaboxes.game import State
from alphaboxes.opponents import solve
from alphaboxes.search import MCTS, CachedEvaluator, Node, SearchConfig, perspective


def uniform(state):
    return np.ones(state.board.num_edges), 0.0


def test_time_budget_stops_search_and_preserves_legal_fallback(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("alphaboxes.search.monotonic", lambda: clock[0])

    def slow(state):
        clock[0] += 0.1
        return np.arange(1, state.board.num_edges + 1), 0.4

    state = State.new(2, 2).play(11)
    for limit, count in ((0.05, 0), (0.25, 2)):
        clock[0] = 0
        search = MCTS(slow, SearchConfig(simulations=100, time_limit=limit))
        policy, value = search.policy(state)
        assert search.completed_simulations == count
        assert np.isclose(policy.sum(), 1) and policy[11] == 0
        assert -1 <= value <= 1
        if count == 0:
            assert value == 0.4 and policy.argmax() == 10


def test_repeated_expired_search_retains_value_with_tree_reuse(monkeypatch):
    clock = [0.0]

    def tick():
        clock[0] += 1
        return clock[0]

    monkeypatch.setattr("alphaboxes.search.monotonic", tick)
    search = MCTS(
        lambda state: (np.ones(state.board.num_edges), 0.7),
        SearchConfig(time_limit=0.01, reuse_tree=True),
    )
    state = State.new(2, 2)
    for _ in range(2):
        policy, value = search.policy(state)
        assert np.isclose(policy.sum(), 1) and value == 0.7
        assert search.completed_simulations == 0


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


def test_reuse_follows_moves_and_clears_after_weight_changes():
    state = State.new(2, 2)
    search = MCTS(uniform, SearchConfig(simulations=64, reuse_tree=True))
    policy, _ = search.policy(state)
    old = search._roots[0]
    action = int(np.argmax(policy))
    child = old.children[action]
    state = state.play(action)
    search.policy(state)
    assert search._roots[0] is child
    assert search.inherited_visits > 0
    search.clear_tree()
    search.policy(state)
    assert search._roots[0] is not child
    assert search.inherited_visits == 0


def test_reuse_handles_undo_and_independent_identical_games():
    state = State.new(2, 2)
    search = MCTS(uniform, SearchConfig(simulations=64, reuse_tree=True))
    policies = search.policies([state, state])
    assert search._roots[0] is not search._roots[1]
    old = search._roots.copy()
    action = int(np.argmax(policies[0][0]))
    search.policy(state.play(action))
    search.policy(state)
    assert search.inherited_visits == 0
    assert all(search._roots[0] is not node for node in old)
    search.policy(State.new(1, 2))
    assert search.inherited_visits == 0


def test_reuse_removes_exploration_noise_before_evaluation():
    state = State.new(2, 2)
    search = MCTS(uniform, SearchConfig(simulations=16, reuse_tree=True), seed=10)
    search.policy(state, explore=True)
    assert not np.allclose([c.prior for c in search._roots[0].children.values()], 1 / 12)
    search.policy(state, explore=False)
    assert np.allclose([c.prior for c in search._roots[0].children.values()], 1 / 12)


def test_materialized_children_preserve_capture_and_turn_perspectives():
    state = State.new(1, 2)
    for action in (0, 2, 4):
        state = state.play(action)
    search = MCTS(uniform, SearchConfig(simulations=64, reuse_tree=True))
    search.policy(state)
    visited = list(search._roots)
    captures = switches = 0
    while visited:
        node = visited.pop()
        for child in (node.children or {}).values():
            if child.state is not None:
                assert child.player == child.state.player
                assert child.state == node.state.play(child.action)
                captures += child.player == node.player
                switches += child.player != node.player
                visited.append(child)
            else:
                assert child.visits == 0
    assert captures and switches


def test_degree_two_root_returns_exact_actions_without_network():
    state = State.new(3, 3)
    for edge, (r1, _, r2, _) in enumerate(state.board.edges):
        if r1 == r2:
            state = state.play(edge)

    def forbidden(state):
        raise AssertionError("Recognized endgame should not use the network")

    search = MCTS(forbidden, SearchConfig(degree_two=True, prove_results=True))
    policy, value = search.policy(state)
    margin, actions = solve(state)
    assert set(np.flatnonzero(policy)) == set(actions)
    assert value == np.sign(margin) and search.endgame_hits == 1


def test_proof_backups_respect_captures_and_require_all_losing_alternatives():
    search = MCTS(uniform, SearchConfig(prove_results=True))
    root = Node(None, player=0)
    capture = Node(None, player=0, solved_value=1.0)
    root.children = {0: capture, 1: Node(None, player=-1)}
    search._prove([root, capture])
    assert root.solved_value == 1
    root = Node(None, player=0)
    losing = Node(None, player=1, solved_value=1.0)
    unknown = Node(None, player=-1)
    root.children = {0: losing, 1: unknown}
    search._prove([root, losing])
    assert root.solved_value is None
    assert search._select(root, 1.0) is unknown
    unknown.player, unknown.solved_value = 0, -1.0
    search._prove([root, unknown])
    assert root.solved_value == -1


def test_proof_search_finishes_small_game_before_simulation_budget():
    search = MCTS(uniform, SearchConfig(simulations=10000, prove_results=True))
    state = State.new(1, 1)
    policy, value = search.policy(state)
    assert search.completed_simulations < 10000
    assert policy.argmax() in solve(state)[1]
    assert value == -1 and search.proven_nodes > 0


def test_policy_mass_first_play_scales_with_explored_probability():
    root = Node(None, player=0, evaluation=0.0)
    visited = Node(None, prior=0.25, visits=1, value_sum=-0.2, player=0)
    unvisited = Node(None, prior=0.75, player=-1)
    root.children = {0: visited, 1: unvisited}
    constant = MCTS(uniform, SearchConfig(fpu_reduction=0.33))
    adaptive = MCTS(uniform, SearchConfig(fpu_reduction=0.33, fpu_mode="policy_mass"))
    assert constant._select(root, 0) is visited
    assert adaptive._select(root, 0) is unvisited


def test_branching_mode_preserves_constant_search_on_small_roots():
    state = State.new(4, 4)
    base = MCTS(uniform, SearchConfig(simulations=128, fpu_reduction=0.33))
    adaptive = MCTS(
        uniform, SearchConfig(simulations=128, fpu_reduction=0.33, fpu_mode="branching")
    )
    actual, value = adaptive.policy(state)
    expected, expected_value = base.policy(state)
    np.testing.assert_array_equal(actual, expected)
    assert value == expected_value


def test_first_play_can_use_the_backed_up_parent_mean():
    root = Node(None, player=0, evaluation=0.0, visits=1, value_sum=-0.2)
    visited = Node(None, prior=0.25, visits=1, value_sum=0.2, player=1)
    unvisited = Node(None, prior=0.75, player=-1)
    root.children = {0: visited, 1: unvisited}
    network = MCTS(uniform, SearchConfig(fpu_reduction=0.33, fpu_mode="policy_mass"))
    mean = MCTS(
        uniform, SearchConfig(fpu_reduction=0.33, fpu_mode="policy_mass", fpu_parent="mean")
    )
    assert network._select(root, 0) is unvisited
    assert mean._select(root, 0) is visited
    root.visits = 0
    assert mean._select(root, 0) is unvisited


def test_reuse_discards_value_only_solved_leaf_when_root_action_solve_times_out(monkeypatch):
    from alphaboxes.endgame import EndgameTimeout

    state = State.new(3, 3)
    for action, (r1, _, r2, _) in enumerate(state.board.edges):
        if r1 == r2:
            state = state.play(action)
    search = MCTS(uniform, SearchConfig(simulations=8, degree_two=True, reuse_tree=True))
    search._roots = [
        Node(state, player=state.player, visits=16, value_sum=-16, solved_value=-1, children=None)
    ]

    def timeout(*args):
        raise EndgameTimeout

    monkeypatch.setattr("alphaboxes.search.shared_solver.solve", timeout)
    policy, _ = search.policy(state, explore=True)
    assert search.inherited_visits == 0
    assert search._roots[0].children
    assert policy.argmax() in state.legal_actions
    assert np.isclose(policy.sum(), 1)
