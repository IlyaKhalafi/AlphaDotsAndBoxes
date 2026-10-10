"""Widen trained GINs while preserving their policy and value at initialization."""

import math
from pathlib import Path

import torch

from alphaboxes.checkpoint import load_agent, save_agent
from alphaboxes.graph import FEATURES


def widen_agent(
    source: Path, output: Path, factor: int = 3, noise: float = 0.01, seed: int = 45
) -> dict:
    """Replicate channels and split outgoing weights; retain the network depth.

    Zero-sum perturbations to outgoing weights break replica symmetry without
    changing predictions in exact arithmetic. Optimizer state is not transferred.
    """
    if factor < 2 or not isinstance(factor, int):
        raise ValueError("Widening requires an integer factor of at least two.")
    if not math.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative.")
    if output.suffix != ".pt" or source.resolve() == output.resolve():
        raise ValueError("Write widened weights to a separate .pt checkpoint.")
    module, metadata = load_agent(source)
    width, depth = module.model_config["width"], module.model_config["depth"]
    original = module.state_dict()
    weights = {}
    generator = torch.Generator().manual_seed(seed)

    def linear(name, segments, expand_output=True):
        weight = original[f"{name}.weight"]
        out_indices = torch.arange(weight.shape[0]).repeat_interleave(
            factor if expand_output else 1
        )
        inputs, counts = [], []
        offset = 0
        for size, expand in segments:
            copies = factor if expand else 1
            inputs.append(torch.arange(offset, offset + size).repeat_interleave(copies))
            counts.extend([copies] * (size * copies))
            offset += size
        expanded = weight[out_indices][:, torch.cat(inputs)] / torch.tensor(counts)
        offset = 0
        for size, expand in segments:
            copies = factor if expand else 1
            if expand and noise:
                perturbation = torch.randn(len(out_indices), size, copies, generator=generator)
                perturbation -= perturbation.mean(dim=-1, keepdim=True)
                perturbation *= noise * weight.std(unbiased=False) / copies
                expanded[:, offset : offset + size * copies] += perturbation.flatten(1)
            offset += size * copies
        weights[f"{name}.weight"] = expanded
        weights[f"{name}.bias"] = original[f"{name}.bias"][out_indices].clone()

    linear("encoder.0", [(FEATURES + 3, False)])
    for index in range(depth):
        prefix = f"blocks.{index}"
        linear(f"{prefix}.mlp.0", [(width, True)])
        linear(f"{prefix}.mlp.2", [(width * 2, True)])
        weights[f"{prefix}.epsilon"] = original[f"{prefix}.epsilon"].clone()
        for name in ("weight", "bias"):
            key = f"{prefix}.norm.{name}"
            weights[key] = original[key].repeat_interleave(factor)
    linear("policy.0", [(width * 3, True), (3, False)])
    linear("policy.2", [(width, True)], expand_output=False)
    linear("value.0", [(width * 2, True), (3, False)])
    linear("value.2", [(width, True)], expand_output=False)
    expansion = {
        "method": "channel_replication_with_zero_sum_outgoing_noise",
        "source_checkpoint_sha256": metadata["checkpoint_sha256"],
        "source_width": width,
        "width": width * factor,
        "depth": depth,
        "factor": factor,
        "noise": noise,
        "seed": seed,
        "source_parameters": sum(value.numel() for value in original.values()),
        "parameters": sum(value.numel() for value in weights.values()),
    }
    metadata = {key: value for key, value in metadata.items() if key != "checkpoint_sha256"}
    metadata["expansion"] = expansion
    save_agent(output, weights, width * factor, depth, metadata)
    return {**expansion, "checkpoint_sha256": load_agent(output)[1]["checkpoint_sha256"]}
