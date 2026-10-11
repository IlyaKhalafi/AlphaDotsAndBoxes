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
        with torch.no_grad():
            target = batch["policy_target"]
            target_entropy = -(target * target.clamp_min(1e-12).log()).sum(dim=-1).mean()
            policy_entropy = -(log_policy.exp() * log_policy).sum(dim=-1).mean()
            policy_kl = policy_loss.detach() - target_entropy
            value_mae = (fwd_out[Columns.VF_PREDS] - batch["value_target"]).abs().mean()
        self.metrics.log_dict(
            {
                "policy_loss": policy_loss.detach(),
                "value_loss": value_loss.detach(),
                "target_entropy": target_entropy,
                "policy_entropy": policy_entropy,
                "policy_kl": policy_kl,
                "value_mae": value_mae,
            },
            key=module_id,
            window=1,
        )
        return policy_loss + value_loss
