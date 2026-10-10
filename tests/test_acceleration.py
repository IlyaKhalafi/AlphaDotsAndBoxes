import numpy as np
import pytest

import alphaboxes.opponents as opponents
from alphaboxes.game import State

accelerator = pytest.importorskip("alphaboxes._endgame")


def test_compiled_solver_matches_python_across_sizes_and_extra_turns(monkeypatch):
    rng = np.random.default_rng(901)
    compiled = accelerator.solve_remaining
    for size in ((1, 1), (1, 2), (2, 3), (4, 4), (5, 5), (6, 6), (2, 7)):
        for _ in range(3):
            state = State.new(*size)
            while len(state.legal_actions) > 8:
                state = state.play(int(rng.choice(state.legal_actions)))
            monkeypatch.setattr(opponents, "solve_remaining", None)
            expected = opponents.solve(state)
            monkeypatch.setattr(opponents, "solve_remaining", compiled)
            assert opponents.solve(state) == expected


def test_compiled_solver_rejects_invalid_buffers():
    masks = np.array([1], dtype=np.uint32)
    with pytest.raises(ValueError, match="Neighbor index"):
        accelerator.solve_remaining(masks, np.array([[2, -1]], dtype=np.int32))
    with pytest.raises(ValueError, match="incident edge"):
        accelerator.solve_remaining(
            np.array([0], dtype=np.uint32), np.array([[0, -1]], dtype=np.int32)
        )
    with pytest.raises(ValueError, match="at most 18"):
        accelerator.solve_remaining(masks, np.zeros((19, 2), dtype=np.int32))
