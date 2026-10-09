from unittest.mock import patch

import numpy as np

from alphaboxes.game import State
from alphaboxes.network import module_spec
from alphaboxes.search import NeuralEvaluator


def test_cache_uses_relative_ownership_and_invalidates():
    model = module_spec(width=16, depth=2).build()
    evaluator = NeuralEvaluator(model)
    state = State.new(1, 2).play(0)
    mirrored = State(
        state.board,
        tuple(1 if owner == 0 else owner for owner in state.edges),
        state.owners,
        1 - state.player,
    )
    with patch.object(model, "forward_inference", wraps=model.forward_inference) as forward:
        a = evaluator(state)
        b = evaluator(mirrored)
        assert forward.call_count == 1
        assert np.array_equal(a[0], b[0]) and a[1] == b[1]
        evaluator.clear_cache()
        evaluator(state)
        assert forward.call_count == 2


def test_cache_eviction_and_disabled_mode():
    model = module_spec(width=16, depth=2).build()
    state = State.new(1, 2)
    for capacity, expected in [(0, 3), (1, 3), (2, 2)]:
        evaluator = NeuralEvaluator(model, capacity)
        with patch.object(model, "forward_inference", wraps=model.forward_inference) as forward:
            evaluator(state)
            evaluator(state.play(0))
            evaluator(state)
            assert forward.call_count == expected
