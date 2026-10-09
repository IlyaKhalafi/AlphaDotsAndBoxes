import numpy as np
import pytest

from alphaboxes.game import State
from alphaboxes.graph import encode_batch


def test_mixed_shapes_keep_order_turn_features_and_padding():
    near_capture = State.new(1, 1).play(0).play(1).play(2)
    captured = near_capture.play(3)
    observation = encode_batch([near_capture, State.new(1, 2), captured], capacity=12)
    np.testing.assert_array_equal(observation["context"], [[0, 0.25, 1], [0, 1, 1], [1, 0, 0]])
    np.testing.assert_array_equal(observation["action_mask"][0, :4], [0, 0, 0, 1])
    np.testing.assert_array_equal(observation["x"][0, :4, 11], [0.5] * 4)
    assert observation["x"][0, 4, 6] == 1  # one remaining side
    assert observation["x"][2, 4, 3] == 1  # capturer keeps the turn
    assert observation["x"][2, 4, 5] == 1  # no remaining sides
    assert observation["x"][1, 5, 10] == 1  # shared edge touches both boxes
    for index, nodes in [(0, 5), (1, 9), (2, 5)]:
        assert not observation["x"][index, nodes:].any()
        assert not observation["node_mask"][index, nodes:].any()
        assert not observation["adjacency"][index, nodes:].any()
        assert not observation["adjacency"][index, :, nodes:].any()
    assert all(value.dtype == np.float32 for value in observation.values())


def test_invalid_batch_capacity_and_empty_batch():
    with pytest.raises(ValueError, match="capacity"):
        encode_batch([State.new(1, 2)], capacity=4)
    with pytest.raises(ValueError, match="empty"):
        encode_batch([])
