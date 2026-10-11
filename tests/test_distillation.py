import hashlib
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from alphaboxes.distillation import distill, frozen_examples, split_positions
from alphaboxes.game import State
from alphaboxes.search import CachedEvaluator
from alphaboxes.teaching import TeachingDataset


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
