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
    from alphaboxes.checkpoint import load_evaluator, save_agent
    from alphaboxes.network import module_spec

    checkpoint = tmp_path / "agent.pt"
    module = module_spec(width=16, depth=2).build()
    save_agent(checkpoint, module.state_dict(), 16, 2, {"games_total": 0})
    expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def load_then_move(path):
        loaded = load_evaluator(path)
        path.rename(tmp_path / "moved.pt")
        return loaded

    monkeypatch.setattr(evaluation, "load_evaluator", load_then_move)
    receipt = evaluation.evaluate(
        checkpoint, tmp_path / "result.json", [(1, 1)], games=2, simulations=2
    )
    assert receipt["checkpoint_sha256"] == expected_hash
    assert not checkpoint.exists()


def test_selected_opponents_and_invalid_combination(tmp_path):
    from alphaboxes.checkpoint import save_agent
    from alphaboxes.evaluation import evaluate
    from alphaboxes.network import module_spec

    checkpoint = tmp_path / "agent.pt"
    module = module_spec(width=16, depth=2).build()
    save_agent(checkpoint, module.state_dict(), 16, 2, {})
    receipt = evaluate(
        checkpoint,
        tmp_path / "result.json",
        [(1, 1)],
        games=2,
        simulations=2,
        opponent_names=["chain_control"],
    )
    assert [row["opponent"] for row in receipt["results"]] == ["chain_control"]
    assert receipt["opponents"] == ["chain_control"]
    with pytest.raises(ValueError, match="supported opponents"):
        evaluate(
            checkpoint,
            tmp_path / "invalid.json",
            [(1, 1)],
            games=2,
            opponent_names=["tactical_endgame"],
        )


def test_checkpoint_duel_balances_seats_and_records_replay(tmp_path):
    from alphaboxes.checkpoint import save_agent
    from alphaboxes.evaluation import compare_agents
    from alphaboxes.network import module_spec

    checkpoint = tmp_path / "agent.pt"
    module = module_spec(width=16, depth=2).build()
    save_agent(checkpoint, module.state_dict(), 16, 2, {})
    receipt = compare_agents(
        checkpoint,
        checkpoint,
        tmp_path / "duel.json",
        [(1, 1)],
        games=2,
        simulations=2,
        exact_threshold=4,
        opening_moves=0,
    )
    row = receipt["results"][0]
    assert row["wins"] == row["losses"] == 1
    assert row["score_rate"] == 0.5
    for seat, moves, margin in zip(
        row["agent_seats"], row["moves"], row["box_margins"], strict=True
    ):
        state = State.new(1, 1)
        for action in moves:
            state = state.play(action)
        assert state.terminal
        assert state.scores[seat] - state.scores[1 - seat] == margin
