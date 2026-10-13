import numpy as np
import pytest

from alphaboxes.game import State
from alphaboxes.numpy_network import grid_messages

kernels = pytest.importorskip("alphaboxes._kernels")


def test_native_activation_keeps_numpy_exponential_and_float_bits():
    rng = np.random.default_rng(62006)
    raw = (rng.normal(size=(13, 192)) * 50).astype(np.float32)
    raw[0, :8] = [-100, -80, -1e-40, -0.0, 0.0, 1e-40, 80, 100]
    bias = rng.normal(size=192).astype(np.float32)
    bias[:8] = 0
    biased = raw + bias
    expected_denominator = np.clip(-biased, -80, 80)
    values, denominator = raw.copy(), np.empty_like(raw)
    kernels.activation_prepare(values, bias, denominator)
    np.testing.assert_array_equal(values.view(np.uint32), biased.view(np.uint32))
    np.testing.assert_array_equal(denominator.view(np.uint32), expected_denominator.view(np.uint32))
    np.exp(denominator, out=denominator)
    kernels.activation_finish(values, denominator)
    expected = biased / (np.exp(np.clip(-biased, -80, 80)) + np.float32(1))
    np.testing.assert_array_equal(values.view(np.uint32), expected.view(np.uint32))
    with pytest.raises(ValueError, match="overlap"):
        kernels.activation_prepare(raw, bias, raw)
    with pytest.raises(ValueError, match="overlap"):
        kernels.activation_finish(raw, raw)


@pytest.mark.parametrize("width", [1, 7, 8, 11, 48, 96, 128, 192, 257, 384])
def test_native_normalization_matches_float32_reductions(width):
    rng = np.random.default_rng(62002)
    raw = rng.normal(size=(17, width)).astype(np.float32)
    raw[0] = 0
    residual = rng.normal(size=raw.shape).astype(np.float32)
    bias, scale, shift = rng.normal(size=(3, width)).astype(np.float32)
    mask = rng.integers(0, 2, size=17).astype(np.float32)
    expected = raw + bias
    expected += residual
    expected -= expected.mean(axis=-1, keepdims=True)
    denominator = (expected * expected).mean(axis=-1, keepdims=True) + np.float32(1e-5)
    np.sqrt(denominator, out=denominator)
    expected /= denominator
    expected *= scale
    expected += shift
    expected *= mask[:, None]
    original = [a.copy() for a in (bias, residual, scale, shift, mask)]
    actual = raw.copy()
    kernels.normalize(actual, bias, residual, scale, shift, mask)
    np.testing.assert_array_equal(actual, expected)
    for array, before in zip((bias, residual, scale, shift, mask), original, strict=True):
        np.testing.assert_array_equal(array, before)


@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (8, 1), (3, 5), (10, 10), (20, 30)])
def test_native_aggregate_matches_ordered_messages(shape):
    state = State.new(*shape)
    rng = np.random.default_rng(62003)
    hidden = rng.normal(size=(state.board.num_nodes, 96)).astype(np.float32)
    hidden[0] = -0.0
    epsilon = np.float32(0.17)
    messages = np.empty_like(hidden)
    grid_messages(hidden, *shape, messages)
    expected = (np.float32(1) + epsilon) * hidden
    expected += messages
    actual = np.full_like(hidden, np.nan)
    kernels.aggregate(hidden, actual, *shape, epsilon)
    np.testing.assert_array_equal(actual, expected)
    with pytest.raises(ValueError, match="overlap"):
        kernels.aggregate(hidden, hidden, *shape, epsilon)
    with pytest.raises(ValueError, match="shape"):
        kernels.aggregate(hidden, actual[:-1], *shape, epsilon)
    with pytest.raises(ValueError):
        kernels.aggregate(hidden, actual, 0, shape[1], epsilon)


def test_native_normalization_rejects_invalid_buffers():
    raw = np.ones((4, 16), np.float32)
    vectors = np.ones((3, 16), np.float32)
    mask = np.ones(4, np.float32)
    with pytest.raises(ValueError, match="overlap"):
        kernels.normalize(raw, vectors[0], raw, vectors[1], vectors[2], mask)
    with pytest.raises(ValueError, match="widths"):
        kernels.normalize(raw, vectors[0, :-1], raw.copy(), vectors[1], vectors[2], mask)
    with pytest.raises(ValueError, match="shapes"):
        kernels.normalize(raw, vectors[0], raw.copy(), vectors[1], vectors[2], mask[:-1])
    with pytest.raises(ValueError):
        kernels.normalize(raw[:, ::-1], vectors[0], raw.copy(), vectors[1], vectors[2], mask)
    with pytest.raises(ValueError, match="overlap"):
        kernels.normalize(raw, raw[0], raw.copy(), vectors[1], vectors[2], mask)


@pytest.mark.parametrize("width", [1, 7, 8, 11, 96, 192])
def test_native_normalization_preserves_signed_zero(width):
    raw = np.full((3, width), -0.0, np.float32)
    bias = np.full(width, -0.0, np.float32)
    residual = raw.copy()
    scale = np.ones(width, np.float32)
    shift = bias.copy()
    mask = np.ones(3, np.float32)
    expected = raw + bias + residual
    expected -= expected.mean(axis=-1, keepdims=True)
    expected /= np.sqrt((expected * expected).mean(axis=-1, keepdims=True) + np.float32(1e-5))
    expected *= scale
    expected += shift
    kernels.normalize(raw, bias, residual, scale, shift, mask)
    np.testing.assert_array_equal(raw.view(np.uint32), expected.view(np.uint32))
