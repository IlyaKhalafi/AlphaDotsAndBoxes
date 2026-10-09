"""NumPy inference for the trained residual GIN, without PyTorch or RLlib."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np

from alphaboxes.game import State
from alphaboxes.graph import FEATURES, encode
from alphaboxes.search import CachedEvaluator


def weight_shapes(width: int, depth: int) -> dict[str, tuple[int, ...]]:
    shapes = {}
    layers = {
        "encoder.0": (FEATURES + 3, width),
        "policy.0": (width * 3 + 3, width),
        "policy.2": (width, 1),
        "value.0": (width * 2 + 3, width),
        "value.2": (width, 1),
    }
    for index in range(depth):
        prefix = f"blocks.{index}"
        layers[f"{prefix}.mlp.0"] = (width, width * 2)
        layers[f"{prefix}.mlp.2"] = (width * 2, width)
        shapes[f"{prefix}.epsilon"] = ()
        shapes[f"{prefix}.norm.weight"] = (width,)
        shapes[f"{prefix}.norm.bias"] = (width,)
    for name, (inputs, outputs) in layers.items():
        shapes[f"{name}.weight"] = (outputs, inputs)
        shapes[f"{name}.bias"] = (outputs,)
    return shapes


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(np.clip(-x, -80, 80)))


class NumpyNetwork:
    def __init__(self, weights: dict[str, np.ndarray], width: int, depth: int):
        if width < 8 or depth < 1:
            raise ValueError("Network requires width >= 8 and depth >= 1.")
        shapes = weight_shapes(width, depth)
        if weights.keys() != shapes.keys():
            raise ValueError("Checkpoint parameters do not match the GIN architecture.")
        for name, shape in shapes.items():
            value = weights[name]
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"Invalid parameter: {name}")
            value.flags.writeable = False
        self.weights = weights
        self.width, self.depth = width, depth

    def linear(self, x: np.ndarray, name: str) -> np.ndarray:
        return x @ self.weights[f"{name}.weight"].T + self.weights[f"{name}.bias"]

    def forward(self, obs: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        """Return masked logits and value for one graph, including optional padding."""
        x, context = obs["x"], obs["context"]
        mask = obs["node_mask"][:, None]
        broadcast = np.broadcast_to(context, (len(x), 3))
        h = silu(self.linear(np.concatenate((x, broadcast), axis=-1), "encoder.0")) * mask
        for index in range(self.depth):
            prefix = f"blocks.{index}"
            messages = obs["adjacency"] @ h
            aggregate = (1 + self.weights[f"{prefix}.epsilon"]) * h + messages
            residual = h + self.linear(
                silu(self.linear(aggregate, f"{prefix}.mlp.0")), f"{prefix}.mlp.2"
            )
            centered = residual - residual.mean(axis=-1, keepdims=True)
            normalized = centered / np.sqrt(
                np.mean(centered * centered, axis=-1, keepdims=True) + np.float32(1e-5)
            )
            h = (
                normalized * self.weights[f"{prefix}.norm.weight"]
                + self.weights[f"{prefix}.norm.bias"]
            ) * mask
        mean = h.sum(axis=0) / max(mask.sum(), 1)
        maximum = np.where(mask != 0, h, np.float32(-1e9)).max(axis=0)
        readout = np.concatenate((mean, maximum, context))
        shared = np.broadcast_to(readout, (len(x), len(readout)))
        logits = self.linear(
            silu(self.linear(np.concatenate((h, shared), axis=-1), "policy.0")), "policy.2"
        ).ravel()
        logits = np.where(obs["action_mask"] != 0, logits, np.float32(-1e9))
        value = np.tanh(self.linear(silu(self.linear(readout, "value.0")), "value.2"))[0]
        return logits, float(value)


class NumpyEvaluator(CachedEvaluator):
    def __init__(self, network: NumpyNetwork, cache_size: int = 4096):
        super().__init__(cache_size)
        self.network = network

    def _predict(self, state: State) -> tuple[np.ndarray, float]:
        logits, value = self.network.forward(encode(state))
        logits = logits[: state.board.num_edges]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return probabilities, value


def load_network(path: Path) -> tuple[NumpyNetwork, dict]:
    snapshot = path.read_bytes()
    with np.load(io.BytesIO(snapshot), allow_pickle=False) as archive:
        payload = json.loads(archive["_metadata"].tobytes().decode("utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("Unsupported NumPy checkpoint version.")
        weights = {name: archive[name] for name in archive.files if name != "_metadata"}
    network = NumpyNetwork(weights, **payload["model_config"])
    metadata = {
        **payload["metadata"],
        "checkpoint_sha256": hashlib.sha256(snapshot).hexdigest(),
    }
    return network, metadata


def export_agent(source: Path, output: Path) -> dict:
    """Convert trusted training weights; only this conversion needs the train extra."""
    from alphaboxes.checkpoint import load_agent

    if output.suffix != ".npz":
        raise ValueError("Deployment checkpoints must use the .npz extension.")
    module, metadata = load_agent(source)
    weights = {
        name: value.detach().cpu().numpy().copy() for name, value in module.state_dict().items()
    }
    config = {"width": module.model_config["width"], "depth": module.model_config["depth"]}
    NumpyNetwork(weights, **config)  # Validate the conversion before writing.
    metadata = {**metadata, "source_checkpoint_sha256": metadata["checkpoint_sha256"]}
    metadata.pop("checkpoint_sha256")
    payload = {"format_version": 1, "model_config": config, "metadata": metadata}
    encoded = np.frombuffer(json.dumps(payload, sort_keys=True).encode("utf-8"), dtype=np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, _metadata=encoded, **weights)
    temporary.replace(output)
    return load_network(output)[1]
