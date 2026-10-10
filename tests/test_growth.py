import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from alphaboxes.checkpoint import load_agent, save_agent
from alphaboxes.game import State
from alphaboxes.graph import encode
from alphaboxes.growth import widen_agent
from alphaboxes.network import module_spec, tensor_observations
from alphaboxes.numpy_network import export_agent, load_network
from alphaboxes.search import SearchConfig
from alphaboxes.training import TrainConfig, train


@pytest.mark.parametrize("factor", [2, 3])
def test_widening_preserves_predictions_and_breaks_gradient_symmetry(tmp_path, factor):
    torch.set_num_threads(1)
    torch.manual_seed(52)
    model = module_spec(width=16, depth=2).build().eval()
    source, output = tmp_path / "source.pt", tmp_path / "wide.pt"
    save_agent(source, model.state_dict(), 16, 2, {"games_total": 7})
    receipt = widen_agent(source, output, factor=factor)
    widened, metadata = load_agent(output)
    assert metadata["games_total"] == 7
    assert receipt["parameters"] == sum(p.numel() for p in widened.parameters())
    assert receipt["parameters"] > receipt["source_parameters"] * factor
    export_agent(output, output.with_suffix(".npz"))
    numpy_model, _ = load_network(output.with_suffix(".npz"))
    for size in [(1, 1), (2, 3), (5, 5), (2, 7)]:
        state = State.new(*size).play(0)
        obs = encode(state, state.board.num_nodes + 7)
        tensors = tensor_observations([obs])
        a, b = model.predict(tensors), widened.predict(tensors)
        for key in (Columns.ACTION_DIST_INPUTS, Columns.VF_PREDS):
            torch.testing.assert_close(a[key], b[key], atol=3e-6, rtol=3e-5)
        logits, value = numpy_model.forward(obs)
        np.testing.assert_allclose(
            logits, b[Columns.ACTION_DIST_INPUTS][0].numpy(), atol=3e-6, rtol=3e-5
        )
        assert value == pytest.approx(float(b[Columns.VF_PREDS][0]), abs=3e-6)
    widened.train()
    prediction = widened.forward_train({Columns.OBS: tensors})
    loss = prediction[Columns.VF_PREDS].sum() + prediction[Columns.ACTION_DIST_INPUTS][0, 1]
    loss.backward()
    gradient = widened.encoder[0].weight.grad
    assert not torch.allclose(gradient[0], gradient[1], atol=1e-9, rtol=1e-5)


def test_widened_checkpoint_can_train_with_rllib(tmp_path):
    model = module_spec(width=16, depth=2).build()
    source, wide = tmp_path / "source.pt", tmp_path / "wide.pt"
    save_agent(source, model.state_dict(), 16, 2, {"games_total": 7})
    expansion = widen_agent(source, wide)
    before, _ = load_agent(wide)
    config = TrainConfig(
        sizes=((1, 2),),
        iterations=1,
        games_per_iteration=1,
        workers=0,
        updates_per_iteration=1,
        batch_size=4,
        width=48,
        depth=2,
        search=SearchConfig(simulations=2),
    )
    path = train(config, tmp_path / "training", initial_checkpoint=wide)
    after, metadata = load_agent(path)
    assert metadata["initialization"]["expansion"]["parameters"] == expansion["parameters"]
    assert any(
        not torch.equal(value, after.state_dict()[key])
        for key, value in before.state_dict().items()
    )


@pytest.mark.parametrize("factor,noise", [(1, 0.01), (2.5, 0.01), (3, -1), (3, float("nan"))])
def test_invalid_expansion_is_rejected(tmp_path, factor, noise):
    with pytest.raises(ValueError):
        widen_agent(tmp_path / "source.pt", tmp_path / "wide.pt", factor, noise)
