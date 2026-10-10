import copy

import numpy as np
import pytest
import torch

from alphaboxes.game import State
from alphaboxes.network import module_spec
from alphaboxes.search import NeuralEvaluator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_graph_handles_shapes_partial_batches_and_weight_updates():
    torch.set_num_threads(1)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(1024**3 / total)
    module = module_spec(width=16, depth=2).build().to("cuda").eval()
    eager = NeuralEvaluator(module, cache_size=0)
    captured = NeuralEvaluator(module, cache_size=0, cuda_batch_size=8, capacity=85)
    states = [State.new(*size).play(0) for size in ((1, 1), (2, 3), (5, 5))]
    for group in (states, states[:1], states * 2, states[1:]):
        expected, actual = eager.evaluate_many(group), captured.evaluate_many(group)
        for (policy, value), (reference, reference_value) in zip(actual, expected, strict=True):
            np.testing.assert_allclose(policy, reference, atol=1e-6)
            assert value == pytest.approx(reference_value, abs=1e-6)
    before = captured(states[0])[1]
    weights = copy.deepcopy(module.get_state())
    weights["value.2.bias"] += 0.5
    module.set_state(weights)
    after = captured(states[0])[1]
    assert abs(after - before) > 0.01
    assert after == pytest.approx(eager(states[0])[1], abs=1e-6)
