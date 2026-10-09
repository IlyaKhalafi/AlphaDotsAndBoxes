from dataclasses import replace

import torch

from alphaboxes.checkpoint import load_agent
from alphaboxes.training import TrainConfig, train


def test_rllib_training_checkpoint_and_resume(tmp_path):
    config = TrainConfig(
        sizes=((1, 1), (1, 2)),
        iterations=1,
        games_per_iteration=2,
        workers=0,
        updates_per_iteration=2,
        batch_size=8,
        width=16,
        depth=2,
        checkpoint_every=1,
    )
    path = train(config, tmp_path)
    model, metadata = load_agent(path)
    assert metadata["games_total"] == 2
    before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
    train(replace(config, iterations=2), tmp_path, tmp_path / "resume.pt")
    after, metadata = load_agent(path)
    assert metadata["games_total"] == 4
    assert any(not torch.equal(before[key], tensor) for key, tensor in after.state_dict().items())
