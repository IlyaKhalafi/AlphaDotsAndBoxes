import numpy as np
import pytest

from alphaboxes.game import State
from alphaboxes.graph import adjacency, encode, neighbor_indices
from alphaboxes.numpy_network import NumpyNetwork, grid_messages, silu, weight_shapes


@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (8, 1), (3, 5), (10, 10), (20, 30)])
def test_grid_messages_preserve_ordered_incidence_sums(shape):
    state = State.new(*shape)
    rng = np.random.default_rng(92)
    hidden = rng.normal(size=(state.board.num_nodes, 96)).astype(np.float32)
    original = hidden.copy()
    padded = np.concatenate((hidden, np.zeros((1, 96), np.float32)))
    indices = neighbor_indices(*shape)
    expected = padded[indices[:, 0]] + padded[indices[:, 1]]
    expected += padded[indices[:, 2]]
    expected += padded[indices[:, 3]]
    actual = np.full_like(hidden, np.nan)
    grid_messages(hidden, *shape, actual)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(hidden, original)
    hidden.fill(-0.0)
    padded[:-1] = hidden
    expected = padded[indices[:, 0]] + padded[indices[:, 1]]
    expected += padded[indices[:, 2]]
    expected += padded[indices[:, 3]]
    grid_messages(hidden, *shape, actual)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    with pytest.raises(ValueError, match="C-contiguous"):
        grid_messages(hidden, *shape, np.empty_like(hidden, order="F"))


def test_silu_reuses_only_explicitly_owned_input():
    values = np.array([-100, -80, -1, -0.0, 0, 1, 80, 100], dtype=np.float32)
    original = values.copy()
    expected = silu(values)
    np.testing.assert_array_equal(values, original)
    actual = silu(values, inplace=True)
    assert actual is values
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(1, 1), (2, 7), (4, 4), (5, 5), (10, 10), (12, 12)])
def test_sparse_incidence_matches_dense_with_padding(shape):
    state = State.new(*shape)
    capacity = state.board.num_nodes + 7
    indices = neighbor_indices(*shape, capacity)
    rng = np.random.default_rng(87)
    hidden = rng.normal(size=(capacity, 96)).astype(np.float32)
    padded = np.concatenate((hidden, np.zeros((1, 96), np.float32)))
    actual = padded[indices].sum(axis=1)
    expected = np.zeros_like(hidden)
    n = state.board.num_nodes
    expected[:n] = adjacency(*shape) @ hidden[:n]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert not indices.flags.writeable


def test_sparse_features_do_not_materialize_dense_geometry(monkeypatch):
    state = State.new(20, 30)
    rng = np.random.default_rng(19)
    for _ in range(80):
        state = state.play(int(rng.choice(state.legal_actions)))
    expected = encode(state, state.board.num_nodes + 7)

    def fail(*args):
        raise AssertionError("Sparse encoding attempted dense geometry.")

    monkeypatch.setattr("alphaboxes.graph.adjacency", fail)
    monkeypatch.setattr("alphaboxes.graph.encoding_geometry", fail)
    actual = encode(state, state.board.num_nodes + 7, include_adjacency=False)
    assert "adjacency" not in actual
    for name, value in actual.items():
        np.testing.assert_array_equal(value, expected[name])


@pytest.mark.parametrize("structure", [False, True])
@pytest.mark.parametrize("shape", [(1, 2), (3, 5), (5, 5), (10, 10), (12, 12)])
def test_sparse_network_matches_dense_on_legal_positions_and_padding(shape, structure):
    rng = np.random.default_rng(91)
    weights = {
        name: np.asarray(rng.standard_normal(shape) * 0.04, dtype=np.float32)
        for name, shape in weight_shapes(48, 4, structure).items()
    }
    for name in weights:
        if name.endswith("norm.weight"):
            weights[name] += 1
    network = NumpyNetwork(weights, 48, 4, structure)
    state = State.new(*shape)
    capacity = state.board.num_nodes + 7
    indices = neighbor_indices(*shape, capacity)
    while not state.terminal:
        if len(state.legal_actions) % 9 == 0 or len(state.legal_actions) <= 4:
            dense = encode(state, capacity, structure=structure)
            sparse = encode(state, capacity, structure=structure, include_adjacency=False)
            assert "adjacency" not in sparse
            expected, expected_value = network.forward(dense)
            actual, value = network.forward(sparse, neighbors=indices)
            np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5)
            assert value == pytest.approx(expected_value, abs=3e-6)
            assert np.all(actual[sparse["action_mask"] == 0] == -1e9)
        state = state.play(int(rng.choice(state.legal_actions)))


@pytest.mark.parametrize("structure", [False, True])
@pytest.mark.parametrize("shape", [(1, 1), (1, 7), (8, 1), (3, 5), (10, 10), (12, 12)])
def test_grid_network_preserves_gather_predictions_and_observations(shape, structure):
    rng = np.random.default_rng(93)
    weights = {
        name: np.asarray(rng.standard_normal(shape) * 0.1, dtype=np.float32)
        for name, shape in weight_shapes(48, 6, structure).items()
    }
    for name in weights:
        if name.endswith("norm.weight"):
            weights[name] += 1
    network = NumpyNetwork(weights, 48, 6, structure)
    state = State.new(*shape)
    for moves in (0, state.board.num_edges // 2, state.board.num_edges - 1):
        while state.board.num_edges - len(state.legal_actions) < moves:
            state = state.play(int(rng.choice(state.legal_actions)))
        obs = encode(state, structure=structure, include_adjacency=False)
        original = {name: value.copy() for name, value in obs.items()}
        expected, expected_value = network.forward(obs, neighbors=neighbor_indices(*shape))
        actual, value = network.forward(obs, board_shape=shape)
        np.testing.assert_array_equal(actual, expected)
        assert value == expected_value
        for name in obs:
            np.testing.assert_array_equal(obs[name], original[name])
    with pytest.raises(ValueError, match="unpadded"):
        network.forward(
            encode(state, state.board.num_nodes + 7, structure=structure), board_shape=shape
        )
    with pytest.raises(ValueError, match="unpadded"):
        network.forward(obs, board_shape=(shape[0] + 1, shape[1]))
    with pytest.raises(ValueError, match="not both"):
        network.forward(obs, board_shape=shape, neighbors=neighbor_indices(*shape))
