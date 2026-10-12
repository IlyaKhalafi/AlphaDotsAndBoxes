"""Widen trained GINs while preserving their policy and value at initialization."""

import math
from pathlib import Path

import torch

from alphaboxes.checkpoint import load_agent, save_agent
from alphaboxes.graph import CONTEXT, FEATURES, dimensions


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
    structure = module.structure
    features, context = dimensions(structure)
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

    linear("encoder.0", [(features + context, False)])
    for index in range(depth):
        prefix = f"blocks.{index}"
        linear(f"{prefix}.mlp.0", [(width, True)])
        linear(f"{prefix}.mlp.2", [(width * 2, True)])
        weights[f"{prefix}.epsilon"] = original[f"{prefix}.epsilon"].clone()
        for name in ("weight", "bias"):
            key = f"{prefix}.norm.{name}"
            weights[key] = original[key].repeat_interleave(factor)
    linear("policy.0", [(width * 3, True), (context, False)])
    linear("policy.2", [(width, True)], expand_output=False)
    linear("value.0", [(width * 2, True), (context, False)])
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
    save_agent(output, weights, width * factor, depth, metadata, structure=structure)
    return {**expansion, "checkpoint_sha256": load_agent(output)[1]["checkpoint_sha256"]}


def enable_structure(source: Path, output: Path) -> dict:
    """Add component inputs with zero weights; preserve all existing predictions."""
    from alphaboxes.network import module_spec

    if output.suffix != ".pt" or source.resolve() == output.resolve():
        raise ValueError("Write structured weights to a separate .pt checkpoint.")
    original, metadata = load_agent(source)
    if original.structure:
        raise ValueError("The checkpoint already has component inputs.")
    width, depth = original.model_config["width"], original.model_config["depth"]
    module = module_spec(width=width, depth=depth, structure=True).build()
    weights = module.state_dict()
    features, _ = dimensions(True)
    for name, value in original.state_dict().items():
        target = weights[name]
        if target.shape == value.shape:
            target.copy_(value)
        elif name == "encoder.0.weight":
            target.zero_()
            target[:, :FEATURES].copy_(value[:, :FEATURES])
            target[:, features : features + CONTEXT].copy_(value[:, FEATURES:])
        elif name in {"policy.0.weight", "value.0.weight"}:
            target.zero_()
            target[:, : value.shape[1]].copy_(value)
        else:
            raise ValueError(f"Unexpected parameter expansion: {name}")
    initialization = {
        "method": "zero_new_component_input_columns",
        "source_checkpoint_sha256": metadata["checkpoint_sha256"],
        "source_parameters": sum(p.numel() for p in original.parameters()),
        "parameters": sum(p.numel() for p in module.parameters()),
    }
    metadata = {key: value for key, value in metadata.items() if key != "checkpoint_sha256"}
    metadata["structure_initialization"] = initialization
    save_agent(output, weights, width, depth, metadata, structure=True)
    return {**initialization, "checkpoint_sha256": load_agent(output)[1]["checkpoint_sha256"]}
