"""Portable, inference-only checkpoints separate from resumable learner state."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphaboxes.network import GraphModule


def save_agent(path: Path, weights: dict, width: int, depth: int, metadata: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_config": {"width": width, "depth": depth},
        "state_dict": {key: torch.as_tensor(value).cpu() for key, value in weights.items()},
        "metadata": metadata,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_agent(path: Path) -> tuple[GraphModule, dict]:
    import torch

    from alphaboxes.network import module_spec

    # Hash the same snapshot that is loaded, even if a training job replaces the file.
    snapshot = path.read_bytes()
    payload = torch.load(io.BytesIO(snapshot), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported agent checkpoint version.")
    module = module_spec(**payload["model_config"]).build()
    module.load_state_dict(payload["state_dict"], strict=True)
    metadata = {**payload["metadata"], "checkpoint_sha256": hashlib.sha256(snapshot).hexdigest()}
    return module.eval(), metadata


def load_evaluator(path: Path):
    """Load deployment weights without requiring a training framework."""
    if path.suffix == ".npz":
        from alphaboxes.numpy_network import NumpyEvaluator, load_network

        network, metadata = load_network(path)
        return NumpyEvaluator(network), metadata
    try:
        import torch
    except ImportError as error:
        raise ValueError(
            "Use a .npz checkpoint for deployment. Convert .pt weights with "
            "adb export in an installation with the train extra."
        ) from error
    from alphaboxes.search import NeuralEvaluator

    torch.set_num_threads(1)
    module, metadata = load_agent(path)
    return NeuralEvaluator(module), metadata
