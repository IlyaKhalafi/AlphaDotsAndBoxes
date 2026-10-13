import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from alphaboxes.distillation import build_replay_dataset, distill, frozen_examples, split_positions
from alphaboxes.game import State
from alphaboxes.search import CachedEvaluator
from alphaboxes.teaching import TeachingDataset


def test_replay_targets_balance_boards_and_ignore_duplicate_and_terminal_states(tmp_path):
    import torch

    from alphaboxes.checkpoint import load_evaluator, save_agent
    from alphaboxes.network import module_spec
    from alphaboxes.numpy_network import export_agent
    from alphaboxes.selfplay import Example
    from alphaboxes.teaching import load_dataset

    module = module_spec(width=16, depth=2).build()
    teacher = tmp_path / "teacher.pt"
    save_agent(teacher, module.state_dict(), 16, 2, {})
    numpy_teacher = tmp_path / "teacher.npz"
    export_agent(teacher, numpy_teacher)
    states = []
    for shape in [(2, 2), (3, 5)]:
        state = State.new(*shape)
        states.extend([state, state, state.play(0), state.play(1)])
        while not state.terminal:
            state = state.play(state.legal_actions[0])
        states.append(state)
    replay = tmp_path / "replay.pt"
    torch.save(
        {
            "iteration": 3,
            "config": {"seed": 75},
            "replay": [Example(s, np.zeros(s.board.num_edges), 0) for s in states],
        },
        replay,
    )
    path = tmp_path / "targets.npz"
    metadata = build_replay_dataset(numpy_teacher, replay, path, positions_per_board=2)
    dataset = load_dataset(path)
    assert len(dataset.states) == 4 and dataset.exact_count == 0
    assert len({CachedEvaluator._key(s) for s in dataset.states}) == 4
    assert [(s.board.rows, s.board.cols) for s in dataset.states] == [(2, 2)] * 2 + [(3, 5)] * 2
    assert metadata["sources"][0]["sha256"] == hashlib.sha256(replay.read_bytes()).hexdigest()
    assert not any(s.terminal for s in dataset.states)
    evaluator, _ = load_evaluator(numpy_teacher)
    for i, state in enumerate(dataset.states):
        policy, value = evaluator(state)
        np.testing.assert_array_equal(dataset.anchor_policy[i, : len(policy)], policy)
        assert dataset.anchor_value[i] == pytest.approx(value)
    with pytest.raises(ValueError, match="new dataset"):
        build_replay_dataset(numpy_teacher, replay, path)


def test_equivalent_positions_cannot_cross_validation_split():
    initial = State.new(2, 2)
    other_player = State(initial.board, initial.edges, initial.owners, 1)
    states = [initial, other_player, initial.play(0), initial.play(1)]
    policies = np.zeros((4, initial.board.num_edges), np.float32)
    values = np.zeros(4, np.float32)
    dataset = TeachingDataset(states, policies, values, policies, values, 0, {})
    assert CachedEvaluator._key(initial) == CachedEvaluator._key(other_player)
    for seed in range(20):
        training, validation = split_positions(dataset, seed, fraction=0.5)
        assert set(training) | set(validation) == set(range(4))
        assert not {CachedEvaluator._key(states[i]) for i in training} & {
            CachedEvaluator._key(states[i]) for i in validation
        }


def test_frozen_distillation_targets_and_rllib_compression(tmp_path, monkeypatch):
    from alphaboxes.checkpoint import load_agent, load_evaluator, save_agent
    from alphaboxes.network import module_spec
    from alphaboxes.numpy_network import export_agent
    from alphaboxes.search import NeuralEvaluator
    from alphaboxes.teaching import build_dataset, load_dataset

    teacher = tmp_path / "teacher.pt"
    module = module_spec(width=16, depth=2).build()
    save_agent(teacher, module.state_dict(), 16, 2, {})
    numpy_teacher = tmp_path / "teacher.npz"
    export_agent(teacher, numpy_teacher)
    source = tmp_path / "games.json"
    source.write_text(
        json.dumps(
            {
                "seed": 1,
                "results": [
                    {
                        "size": [3, 3],
                        "moves": [list(range(State.new(3, 3).board.num_edges))],
                    }
                ],
            }
        )
    )
    dataset_path = tmp_path / "targets.npz"
    build_dataset(numpy_teacher, [source], dataset_path)
    dataset = load_dataset(dataset_path)
    training, validation = split_positions(dataset, 1)
    assert not set(training) & set(validation)
    assert set(training) | set(validation) == set(range(len(dataset.states)))
    assert np.array_equal(validation, split_positions(dataset, 1)[1])
    example = frozen_examples(dataset, [0])[0]
    assert example.value == dataset.anchor_value[0]
    assert np.array_equal(example.policy, dataset.anchor_policy[0])
    # Exact targets deliberately disagree; compression must ignore both.
    dataset.oracle_value[0] = 100
    dataset.oracle_policy[0] = 0
    assert frozen_examples(dataset, [0])[0].value == example.value
    assert np.array_equal(frozen_examples(dataset, [0])[0].policy, example.policy)
    output = tmp_path / "student"
    script = Path(__file__).resolve().parents[1] / "scripts/distill.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--teacher",
            str(numpy_teacher),
            "--dataset",
            str(dataset_path),
            "--output",
            str(output),
            "--width",
            "8",
            "--depth",
            "2",
            "--steps",
            "2",
            "--batch-size",
            "4",
            "--validate-every",
            "1",
            "--device",
            "cpu",
        ],
    )
    runpy.run_path(str(script), run_name="__main__")
    result = output / "best.npz"
    numpy_student, _ = load_evaluator(result)
    torch_student, _ = load_agent(output / "best.pt")
    assert numpy_student.network.width == 8
    torch_evaluator = NeuralEvaluator(torch_student)
    for state in dataset.states[:3]:
        numpy_policy, numpy_value = numpy_student(state)
        torch_policy, torch_value = torch_evaluator(state)
        assert np.allclose(numpy_policy, torch_policy, atol=2e-6)
        assert numpy_value == pytest.approx(torch_value, abs=2e-6)
    selection = json.loads((output / "selection.json").read_text())
    assert selection["selected_step"] in (1, 2)
    assert selection["checkpoint_sha256"] == hashlib.sha256(result.read_bytes()).hexdigest()
    assert np.isfinite(selection["validation_score"])
    with pytest.raises(ValueError, match="already exists"):
        distill(numpy_teacher, dataset_path, output, steps=1, device="cpu")
    other = tmp_path / "other.npz"
    export_agent(output / "best.pt", other)
    with pytest.raises(ValueError, match="different teacher"):
        distill(other, dataset_path, tmp_path / "invalid", steps=1, device="cpu")
