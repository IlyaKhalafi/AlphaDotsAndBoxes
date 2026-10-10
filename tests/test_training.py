import json
from dataclasses import replace

import pytest
import torch

from alphaboxes.checkpoint import load_agent, save_agent
from alphaboxes.network import module_spec
from alphaboxes.search import SearchConfig
from alphaboxes.training import TrainConfig, train


def test_rllib_training_checkpoint_and_resume(tmp_path):
    config = TrainConfig(
        sizes=((1, 1), (1, 2)),
        iterations=2,
        games_per_iteration=2,
        workers=0,
        updates_per_iteration=2,
        batch_size=8,
        width=16,
        depth=2,
        checkpoint_every=10,
    )
    path = train(config, tmp_path)
    model, metadata = load_agent(path)
    assert metadata["games_total"] == 4
    before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    assert not (tmp_path / "agent-00001.pt").exists()
    first_record, second_record = [
        json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
    ]
    # Simulate interruption after the atomic checkpoint, before its metrics append.
    (tmp_path / "metrics.jsonl").write_text(json.dumps(first_record) + "\n")
    train(replace(config, iterations=3), tmp_path, tmp_path / "resume.pt")
    after, metadata = load_agent(path)
    assert metadata["games_total"] == 6
    assert any(not torch.equal(before[key], tensor) for key, tensor in after.state_dict().items())
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert records[0] == first_record
    assert records[1] == second_record
    assert [row["iteration"] for row in records] == [1, 2, 3]
    assert records[-1]["elapsed_seconds"] > records[0]["elapsed_seconds"]


def test_resuming_same_run_retains_used_time_budget(tmp_path):
    config = TrainConfig(
        sizes=((1, 1),),
        iterations=1,
        games_per_iteration=1,
        workers=0,
        updates_per_iteration=1,
        batch_size=4,
        width=16,
        depth=2,
        checkpoint_every=1,
        search=SearchConfig(simulations=2),
    )
    train(config, tmp_path)
    resume = tmp_path / "resume.pt"
    state = torch.load(resume, map_location="cpu", weights_only=False)
    state["elapsed_seconds"] = 100.0
    torch.save(state, resume)
    train(replace(config, iterations=10, max_seconds=50), tmp_path, resume)
    record = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[-1])
    assert record["iteration"] == 2
    assert record["stop_reason"] == "time_budget"
    assert record["elapsed_seconds"] >= 100


def test_warm_start_on_new_boards_and_time_budget(tmp_path):
    initial = module_spec(width=16, depth=2).build()
    checkpoint = tmp_path / "initial.pt"
    save_agent(checkpoint, initial.state_dict(), 16, 2, {"games_total": 12, "positions_total": 80})
    _, source = load_agent(checkpoint)
    config = TrainConfig(
        sizes=((2, 3),),
        iterations=10,
        games_per_iteration=1,
        workers=0,
        updates_per_iteration=1,
        batch_size=4,
        width=16,
        depth=2,
        learning_rate=0,
        checkpoint_every=10,
        max_seconds=1e-6,
        search=SearchConfig(simulations=2),
    )
    output = tmp_path / "larger"
    path = train(config, output, initial_checkpoint=checkpoint)
    model, metadata = load_agent(path)
    assert metadata["iteration"] == 1
    assert metadata["sizes"] == [[2, 3]]
    assert metadata["initialization"]["checkpoint_sha256"] == source["checkpoint_sha256"]
    assert metadata["initialization"]["training_games"] == 12
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in initial.state_dict().items()
    )
    assert (output / "resume.pt").exists()
    train(replace(config, iterations=2, max_seconds=None), output, output / "resume.pt")
    _, resumed = load_agent(path)
    assert resumed["games_total"] == 2
    assert resumed["initialization"] == metadata["initialization"]


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_training_time_budget(seconds):
    with pytest.raises(ValueError, match="max_seconds"):
        TrainConfig(max_seconds=seconds)
