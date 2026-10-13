import numpy as np

from alphaboxes.game import State
from alphaboxes.numpy_network import NumpyEvaluator, NumpyNetwork, weight_shapes
from alphaboxes.search import MCTS, SearchConfig


class CountedEvaluator(NumpyEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.predictions = 0

    def _predict(self, state):
        self.predictions += 1
        return super()._predict(state)


def test_larger_cache_preserves_search_and_reuses_continuations():
    rng = np.random.default_rng(23)
    weights = {
        name: np.asarray(rng.standard_normal(shape) * 0.05, np.float32)
        for name, shape in weight_shapes(16, 2).items()
    }
    network = NumpyNetwork(weights, 16, 2)
    small, large = (
        CountedEvaluator(network, cache_size=8),
        CountedEvaluator(network, cache_size=1024),
    )
    state = State.new(4, 4)
    config = SearchConfig(simulations=128, fpu_reduction=0.33, sparse_selection=True)
    for _ in range(3):
        outputs = [MCTS(evaluator, config).policy(state) for evaluator in (small, large)]
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        assert outputs[0][1] == outputs[1][1]
        state = state.play(int(outputs[0][0].argmax()))
    assert large.predictions < small.predictions
    assert len(small._cache) <= 8
    large.clear_cache()
    assert not large._cache
