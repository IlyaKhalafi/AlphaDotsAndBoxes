"""Residual graph isomorphism network exposed as an RLlib TorchRLModule."""

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from torch import nn

from alphaboxes.graph import FEATURES, spaces


class GINBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.epsilon = nn.Parameter(torch.zeros(()))
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width)
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor):
        messages = torch.bmm(adjacency, h)
        return self.norm(h + self.mlp((1 + self.epsilon) * h + messages)) * mask


class GraphModule(TorchRLModule):
    """Shared policy/value network. Padded nodes never enter graph readouts."""

    def setup(self):
        width = int(self.model_config.get("width", 96))
        depth = int(self.model_config.get("depth", 6))
        if width < 8 or depth < 1:
            raise ValueError("Network requires width >= 8 and depth >= 1.")
        self.encoder = nn.Sequential(nn.Linear(FEATURES + 3, width), nn.SiLU())
        self.blocks = nn.ModuleList([GINBlock(width) for _ in range(depth)])
        # Global mean and maximum provide context beyond the local receptive field.
        self.policy = nn.Sequential(nn.Linear(width * 3 + 3, width), nn.SiLU(), nn.Linear(width, 1))
        self.value = nn.Sequential(
            nn.Linear(width * 2 + 3, width), nn.SiLU(), nn.Linear(width, 1), nn.Tanh()
        )

    def _forward(self, batch, **kwargs):
        obs = batch[Columns.OBS]
        x = obs["x"].float()
        mask = obs["node_mask"].float().unsqueeze(-1)
        context = obs["context"].float()
        broadcast = context.unsqueeze(1).expand(-1, x.shape[1], -1)
        h = self.encoder(torch.cat((x, broadcast), dim=-1)) * mask
        adjacency = obs["adjacency"].float()
        for block in self.blocks:
            h = block(h, adjacency, mask)
        mean = h.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        maximum = h.masked_fill(mask == 0, -1e9).amax(dim=1)
        readout = torch.cat((mean, maximum, context), dim=-1)
        shared = readout.unsqueeze(1).expand(-1, x.shape[1], -1)
        logits = self.policy(torch.cat((h, shared), dim=-1)).squeeze(-1)
        logits = logits.masked_fill(obs["action_mask"] == 0, -1e9)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS: self.value(readout).squeeze(-1),
        }


def module_spec(capacity: int = 1, width: int = 96, depth: int = 6) -> RLModuleSpec:
    observation_space, action_space = spaces(capacity)
    return RLModuleSpec(
        module_class=GraphModule,
        observation_space=observation_space,
        action_space=action_space,
        model_config={"width": width, "depth": depth},
    )


def tensor_observations(observations: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(np.stack([obs[key] for obs in observations]))
        for key in observations[0]
    }
