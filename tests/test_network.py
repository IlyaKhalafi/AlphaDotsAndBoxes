import numpy as np
import torch
from ray.rllib.core.columns import Columns

from alphaboxes.game import State
from alphaboxes.graph import encode
from alphaboxes.network import module_spec, tensor_observations


def test_variable_sizes_and_padding_invariance():
    torch.set_num_threads(1)
    model = module_spec(width=16, depth=2).build().eval()
    for size in [(1, 1), (2, 3), (5, 7)]:
        state = State.new(*size).play(0)
        with torch.inference_mode():
            a = model.forward_inference({Columns.OBS: tensor_observations([encode(state)])})
            b = model.forward_inference(
                {Columns.OBS: tensor_observations([encode(state, state.board.num_nodes + 8)])}
            )
        assert torch.allclose(a[Columns.VF_PREDS], b[Columns.VF_PREDS], atol=1e-6)
        assert torch.allclose(
            a[Columns.ACTION_DIST_INPUTS],
            b[Columns.ACTION_DIST_INPUTS][:, : state.board.num_nodes],
            atol=1e-6,
        )
        assert a[Columns.ACTION_DIST_INPUTS][0, 0] < -1e8


def test_node_permutation_equivariance():
    model = module_spec(width=16, depth=2).build().eval()
    obs = encode(State.new(2, 2).play(0))
    permutation = np.random.default_rng(5).permutation(len(obs["x"]))
    shuffled = {
        key: value if key == "context" else value[permutation] for key, value in obs.items()
    }
    shuffled["adjacency"] = obs["adjacency"][permutation][:, permutation]
    with torch.inference_mode():
        a = model.forward_inference({Columns.OBS: tensor_observations([obs])})
        b = model.forward_inference({Columns.OBS: tensor_observations([shuffled])})
    assert torch.allclose(a[Columns.VF_PREDS], b[Columns.VF_PREDS], atol=1e-5)
    assert torch.allclose(
        a[Columns.ACTION_DIST_INPUTS][:, permutation], b[Columns.ACTION_DIST_INPUTS], atol=1e-5
    )
