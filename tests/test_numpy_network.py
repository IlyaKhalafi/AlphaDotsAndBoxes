import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from alphaboxes.checkpoint import save_agent
from alphaboxes.game import State
from alphaboxes.graph import encode
from alphaboxes.network import module_spec, tensor_observations
from alphaboxes.numpy_network import NumpyEvaluator, NumpyNetwork, export_agent, load_network


@pytest.mark.parametrize("size", [(1, 2), (3, 3), (4, 4), (3, 5), (5, 5), (2, 7)])
def test_numpy_matches_torch_on_live_games_and_padding(tmp_path, size):
    torch.manual_seed(41)
    torch.set_num_threads(1)
    module = module_spec(width=96, depth=6).build().eval()
    checkpoint = tmp_path / "agent.pt"
    output = tmp_path / "agent.npz"
    save_agent(checkpoint, module.state_dict(), 96, 6, {"games_total": 12})
    metadata = export_agent(checkpoint, output)
    network, loaded = load_network(output)
    assert metadata == loaded
    assert loaded["games_total"] == 12
    assert loaded["source_checkpoint_sha256"] != loaded["checkpoint_sha256"]
    rng = np.random.default_rng(67)
    state = State.new(*size)
    while not state.terminal:
        obs = encode(state, capacity=state.board.num_nodes + 5)
        with torch.inference_mode():
            expected = module.forward_inference({Columns.OBS: tensor_observations([obs])})
        logits, value = network.forward(obs)
        np.testing.assert_allclose(
            logits, expected[Columns.ACTION_DIST_INPUTS][0].numpy(), atol=3e-6, rtol=3e-5
        )
        assert value == pytest.approx(float(expected[Columns.VF_PREDS][0]), abs=3e-6)
        policy, _ = NumpyEvaluator(network)(state)
        assert np.isclose(policy.sum(), 1)
        assert np.all(policy[np.asarray(state.edges) >= 0] == 0)
        state = state.play(int(rng.choice(state.legal_actions)))


def test_checkpoint_rejects_invalid_parameters():
    with pytest.raises(ValueError, match="parameters"):
        NumpyNetwork({}, width=16, depth=2)
