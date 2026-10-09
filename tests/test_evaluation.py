import pytest

from alphaboxes.evaluation import wilson
from alphaboxes.game import State
from alphaboxes.opponents import endgame_action, solve


def test_wilson_includes_observed_win_rate():
    for wins in [0, 5, 10]:
        lower, upper = wilson(wins, 10)
        assert 0 <= lower <= wins / 10 <= upper <= 1
    assert wilson(10, 10)[0] < 1


def test_endgame_baseline_is_optimal():
    import numpy as np

    state = State.new(2, 2)
    assert endgame_action(state, np.random.default_rng(6)) in solve(state)[1]


@pytest.mark.parametrize("rows,cols", [(0, 2), (-1, 3)])
def test_invalid_board_size(rows, cols):
    with pytest.raises(ValueError):
        State.new(rows, cols)


def test_evaluation_receipt_uses_loaded_snapshot(tmp_path, monkeypatch):
    import hashlib

    from alphaboxes import evaluation
    from alphaboxes.checkpoint import load_agent, save_agent
    from alphaboxes.network import module_spec

    checkpoint = tmp_path / "agent.pt"
    module = module_spec(width=16, depth=2).build()
    save_agent(checkpoint, module.state_dict(), 16, 2, {"games_total": 0})
    expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def load_then_move(path):
        loaded = load_agent(path)
        path.rename(tmp_path / "moved.pt")
        return loaded

    monkeypatch.setattr(evaluation, "load_agent", load_then_move)
    receipt = evaluation.evaluate(
        checkpoint, tmp_path / "result.json", [(1, 1)], games=2, simulations=2
    )
    assert receipt["checkpoint_sha256"] == expected_hash
    assert not checkpoint.exists()
