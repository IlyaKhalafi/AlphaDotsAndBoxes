import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np

from alphaboxes.components import free_components, structural_features
from alphaboxes.endgame import DegreeTwoSolver
from alphaboxes.game import State
from alphaboxes.graph import encode, encode_batch, spaces
from alphaboxes.numpy_network import NumpyNetwork, weight_shapes


def fixture_module():
    path = Path(__file__).resolve().parents[1] / "scripts/inspect_representation.py"
    spec = importlib.util.spec_from_file_location("representation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_closed_chain_and_capture_fronts_are_explicit():
    state = State.new(1, 4)
    for edge, (r1, _, r2, _) in enumerate(state.board.edges):
        if r1 == r2:
            state = state.play(edge)
    (component,) = free_components(state)
    assert component.closed_chain and not component.closed_loop
    assert component.boundaries == 2 and len(component.boxes) == 4
    nodes, context = structural_features(state)
    np.testing.assert_array_equal(nodes, np.tile([1, 1, 0, 0], (4, 1)))
    assert context[0] == 1 and context[1] == 0.25 and context[10] == 1
    state = state.play(state.board.boxes[0][2])
    (component,) = free_components(state)
    assert component.capture_fronts == 1 and not component.closed_chain
    nodes, context = structural_features(state)
    assert np.all(nodes[:, 3] == 0.5) and context[8] == 0.25


def test_structured_batch_padding_spaces_and_player_perspective():
    states = [State.new(2, 3), State.new(1, 4).play(0)]
    capacity = max(s.board.num_nodes for s in states) + 3
    batch = encode_batch(states, capacity, structure=True)
    without_adjacency = encode_batch(states, capacity, structure=True, include_adjacency=False)
    assert set(without_adjacency) == set(batch) - {"adjacency"}
    for index, state in enumerate(states):
        single = encode(state, capacity, structure=True)
        assert spaces(capacity, structure=True)[0].contains(single)
        for key in single:
            np.testing.assert_array_equal(single[key], batch[key][index])
        for key in without_adjacency:
            np.testing.assert_array_equal(without_adjacency[key][index], single[key])
        assert not single["x"][state.board.num_nodes :].any()
    state, _ = fixture_module().loop_fixture(False)
    _, original = structural_features(state)
    _, flipped = structural_features(replace(state, player=1 - state.player))
    np.testing.assert_array_equal(original[:12], flipped[:12])
    assert original[12] == -flipped[12] == 0.5


def test_reachable_opposite_outcomes_alias_locally_but_have_distinct_component_context():
    fixtures = fixture_module()
    first, opening = fixtures.loop_fixture(False)
    second, other_opening = fixtures.loop_fixture(True)
    for state, actions in ((first, opening), (second, other_opening)):
        replay = State.new(14, 14)
        for action in actions:
            replay = replay.play(action)
        assert replay == state and len(state.legal_actions) == 192
    solver = DegreeTwoSolver()
    assert solver.value(first) > 0 and solver.value(second) < 0
    # This information failure also occurs with arbitrary local-network weights.
    rng = np.random.default_rng(6112)
    weights = {
        name: np.asarray(rng.normal(0, 0.1, shape), np.float32)
        for name, shape in weight_shapes(8, 6).items()
    }
    network = NumpyNetwork(weights, 8, 6)
    old_values = [network.forward(encode(state))[1] for state in (first, second)]
    assert abs(old_values[0] - old_values[1]) < 1e-5
    contexts = [encode(state, structure=True)["context"] for state in (first, second)]
    assert not np.array_equal(contexts[0], contexts[1])
