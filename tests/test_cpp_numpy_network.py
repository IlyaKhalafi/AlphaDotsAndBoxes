import numpy as np
import pytest

from alphaboxes import numpy_network
from alphaboxes.game import State
from alphaboxes.graph import encode, neighbor_indices
from alphaboxes.numpy_network import NumpyNetwork, weight_shapes

pytestmark = pytest.mark.skipif(
    numpy_network._kernels is None, reason="Optional C++ kernels absent"
)


@pytest.mark.parametrize("structure", [False, True])
@pytest.mark.parametrize("shape", [(1, 7), (3, 5), (5, 5), (10, 10), (12, 12)])
def test_native_network_matches_numpy_on_positions_and_padding(shape, structure):
    rng = np.random.default_rng(62004)
    weights = {
        name: np.asarray(rng.standard_normal(shape) * 0.08, dtype=np.float32)
        for name, shape in weight_shapes(96, 6, structure).items()
    }
    for name in weights:
        if name.endswith("norm.weight"):
            weights[name] += 1
    reference = NumpyNetwork(weights, 96, 6, structure, native=False)
    candidate = NumpyNetwork(weights, 96, 6, structure, native=True)
    state = State.new(*shape)
    for moves in (0, state.board.num_edges // 2, state.board.num_edges - 1):
        while state.board.num_edges - len(state.legal_actions) < moves:
            state = state.play(int(rng.choice(state.legal_actions)))
        for padded in (False, True):
            capacity = state.board.num_nodes + (7 if padded else 0)
            obs = encode(state, capacity, structure=structure)
            options = (
                {"neighbors": neighbor_indices(*shape, capacity)}
                if padded
                else {"board_shape": shape}
            )
            before = {name: value.copy() for name, value in obs.items()}
            expected, expected_value = reference.forward(obs, **options)
            actual, value = candidate.forward(obs, **options)
            np.testing.assert_array_equal(actual, expected)
            assert value == expected_value
            for name in obs:
                np.testing.assert_array_equal(obs[name], before[name])


def test_native_network_falls_back_if_extension_unavailable(monkeypatch):
    rng = np.random.default_rng(62005)
    weights = {
        name: np.asarray(rng.standard_normal(shape) * 0.04, dtype=np.float32)
        for name, shape in weight_shapes(16, 2).items()
    }
    network = NumpyNetwork(weights, 16, 2, native=True)
    expected = NumpyNetwork(weights, 16, 2, native=False).forward(encode(State.new(3, 4)))
    monkeypatch.setattr(numpy_network, "_kernels", None)
    actual = network.forward(encode(State.new(3, 4)))
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1]
    assert not NumpyNetwork(weights, 16, 2).native
    with pytest.raises(RuntimeError, match="unavailable"):
        NumpyNetwork(weights, 16, 2, native=True)
