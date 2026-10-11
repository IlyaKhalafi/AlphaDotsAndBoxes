import hashlib
import json

import numpy as np
import pytest

from alphaboxes.game import State
from alphaboxes.opponents import solve
from alphaboxes.teaching import collect_positions, oracle_target


def test_oracle_targets_include_claimed_boxes_and_extra_turns():
    rng = np.random.default_rng(18)
    for _ in range(12):
        state = State.new(2, 2)
        while len(state.legal_actions) > 5:
            state = state.play(int(rng.choice(state.legal_actions)))
        policy, value = oracle_target(state)
        margin, actions = solve(state)
        assert value == np.sign(margin)
        assert set(np.flatnonzero(policy)) == set(actions)
        assert policy.sum() == pytest.approx(1)
        for action in actions:
            child = state.play(action)
            child_margin = solve(child)[0]
            assert (child_margin if child.player == state.player else -child_margin) == margin


def test_source_histories_are_deduplicated_and_must_be_complete():
    history = list(range(State.new(3, 3).board.num_edges))
    receipt = {"results": [{"size": [3, 3], "moves": [history, history]}]}
    exact, anchors = collect_positions([receipt, receipt], seed=10)
    assert len(exact) == 4
    assert len(anchors) == 8
    assert all(13 <= len(s.legal_actions) <= 16 for s in exact)
    assert all(len(s.legal_actions) > 16 for s in anchors)
    with pytest.raises(ValueError, match="complete"):
        collect_positions([{"results": [{"size": [3, 3], "moves": [history[:-1]]}]}], 10)


def test_teacher_dataset_and_actual_rllib_update(tmp_path):
    from alphaboxes.checkpoint import save_agent
    from alphaboxes.network import module_spec
    from alphaboxes.numpy_network import export_agent
    from alphaboxes.teaching import build_dataset, fine_tune, load_dataset, teaching_examples

    checkpoint = tmp_path / "initial.pt"
    module = module_spec(width=16, depth=2).build()
    save_agent(checkpoint, module.state_dict(), 16, 2, {})
    numpy_checkpoint = tmp_path / "initial.npz"
    export_agent(checkpoint, numpy_checkpoint)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "seed": 10,
                "results": [
                    {
                        "size": [3, 3],
                        "moves": [list(range(State.new(3, 3).board.num_edges))],
                    }
                ],
            }
        )
    )
    path = tmp_path / "teaching.npz"
    metadata = build_dataset(numpy_checkpoint, [source], path)
    dataset = load_dataset(path)
    assert metadata["sources"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    indices = [0, dataset.exact_count]
    critic_only = teaching_examples(dataset, indices, False)
    both = teaching_examples(dataset, indices, True)
    assert np.allclose(critic_only[0].policy, dataset.anchor_policy[0])
    assert np.allclose(both[0].policy, dataset.oracle_policy[0])
    assert critic_only[0].value == dataset.oracle_value[0]
    assert critic_only[1].value == dataset.anchor_value[dataset.exact_count]
    output = tmp_path / "fine-tune"
    result = fine_tune(checkpoint, path, output, steps=1, batch_size=8, device="cpu")
    assert result.exists()
    metrics = json.loads((output / "metrics.jsonl").read_text())
    assert np.isfinite(metrics["policy_loss"])
    assert np.isfinite(metrics["value_loss"])
    assert result.read_bytes() != checkpoint.read_bytes()
    other = tmp_path / "other.pt"
    save_agent(other, module.state_dict(), 16, 2, {"different": True})
    with pytest.raises(ValueError, match="different starting checkpoint"):
        fine_tune(other, path, tmp_path / "invalid", steps=1, device="cpu")
