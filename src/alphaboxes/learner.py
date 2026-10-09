"""AlphaZero policy cross-entropy and outcome regression in RLlib's learner."""

import torch
import torch.nn.functional as F
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.torch.torch_learner import TorchLearner


class AlphaZeroLearner(TorchLearner):
    def configure_optimizers_for_module(self, module_id, config=None):
        parameters = self.get_parameters(self.module[module_id])
        optimizer = torch.optim.AdamW(parameters, weight_decay=1e-4)
        self.register_optimizer(
            module_id=module_id,
            optimizer=optimizer,
            params=parameters,
            lr_or_lr_schedule=config.lr,
        )

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        log_policy = F.log_softmax(fwd_out[Columns.ACTION_DIST_INPUTS], dim=-1)
        policy_loss = -(batch["policy_target"] * log_policy).sum(dim=-1).mean()
        value_loss = F.mse_loss(fwd_out[Columns.VF_PREDS], batch["value_target"])
        self.metrics.log_dict(
            {"policy_loss": policy_loss.detach(), "value_loss": value_loss.detach()},
            key=module_id,
            window=1,
        )
        return policy_loss + value_loss
