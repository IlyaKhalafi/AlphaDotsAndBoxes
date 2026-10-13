"""NumPy inference for the trained residual GIN, without PyTorch or RLlib."""

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

from alphaboxes.game import State
from alphaboxes.graph import dimensions, encode
from alphaboxes.search import CachedEvaluator

try:
    if os.environ.get("ALPHABOXES_PURE_NUMPY") == "1":
        raise ImportError
    from alphaboxes import _kernels

    if getattr(_kernels, "API_VERSION", None) != 1:
        raise ImportError("Unsupported compiled kernel interface.")
except ImportError:
    _kernels = None


def weight_shapes(width: int, depth: int, structure: bool = False) -> dict[str, tuple[int, ...]]:
    shapes = {}
    features, context = dimensions(structure)
    layers = {
        "encoder.0": (features + context, width),
        "policy.0": (width * 3 + context, width),
        "policy.2": (width, 1),
        "value.0": (width * 2 + context, width),
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


def silu(x: np.ndarray, *, inplace: bool = False) -> np.ndarray:
    denominator = -x
    np.clip(denominator, -80, 80, out=denominator)
    np.exp(denominator, out=denominator)
    denominator += np.float32(1)
    return np.divide(x, denominator, out=x if inplace else None)


def grid_messages(h: np.ndarray, rows: int, cols: int, out: np.ndarray) -> None:
    """Sum ordered neighbors into a C-contiguous output using rectangular views."""
    if not out.flags.c_contiguous:
        raise ValueError("Grid message output must be C-contiguous.")
    horizontal_count = (rows + 1) * cols
    edge_count = horizontal_count + rows * (cols + 1)
    horizontal = h[:horizontal_count].reshape(rows + 1, cols, -1)
    vertical = h[horizontal_count:edge_count].reshape(rows, cols + 1, -1)
    boxes = h[edge_count:].reshape(rows, cols, -1)
    box_messages = out[edge_count:].reshape(rows, cols, -1)
    np.add(horizontal[:-1], horizontal[1:], out=box_messages)
    box_messages += vertical[:, :-1]
    box_messages += vertical[:, 1:]
    horizontal_messages = out[:horizontal_count].reshape(rows + 1, cols, -1)
    horizontal_messages[0] = boxes[0]
    horizontal_messages[-1] = boxes[-1]
    np.add(boxes[:-1], boxes[1:], out=horizontal_messages[1:-1])
    vertical_messages = out[horizontal_count:edge_count].reshape(rows, cols + 1, -1)
    vertical_messages[:, 0] = boxes[:, 0]
    vertical_messages[:, -1] = boxes[:, -1]
    np.add(boxes[:, :-1], boxes[:, 1:], out=vertical_messages[:, 1:-1])
    # Edge nodes also have zero sentinel neighbors in the gather path. Keep
    # their signed-zero behavior, including interior edges with two neighbors.
    np.add(out[:edge_count], np.float32(0), out=out[:edge_count])


class NumpyNetwork:
    def __init__(
        self,
        weights: dict[str, np.ndarray],
        width: int,
        depth: int,
        structure: bool = False,
        *,
        native: bool | None = None,
    ):
        if width < 8 or depth < 1:
            raise ValueError("Network requires width >= 8 and depth >= 1.")
        shapes = weight_shapes(width, depth, structure)
        if weights.keys() != shapes.keys():
            raise ValueError("Checkpoint parameters do not match the GIN architecture.")
        for name, shape in shapes.items():
            value = weights[name]
            if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"Invalid parameter: {name}")
            value.flags.writeable = False
        self.weights = weights
        self.width, self.depth = width, depth
        self.structure = structure
        if native and _kernels is None:
            raise RuntimeError("C++ kernels are unavailable; build optional acceleration first.")
        self.native = _kernels is not None if native is None else native

    def linear(self, x: np.ndarray, name: str) -> np.ndarray:
        result = x @ self.weights[f"{name}.weight"].T
        result += self.weights[f"{name}.bias"]
        return result

    @staticmethod
    def _activate(values: np.ndarray, bias: np.ndarray, kernels) -> np.ndarray:
        if not values.size:
            return values
        if kernels is None:
            values += bias
            return silu(values, inplace=True)
        matrix = values.reshape(-1, values.shape[-1])
        denominator = np.empty_like(matrix)
        kernels.activation_prepare(matrix, bias, denominator)
        np.exp(denominator, out=denominator)
        kernels.activation_finish(matrix, denominator)
        return values

    def _linear_activation(self, x: np.ndarray, name: str, kernels) -> np.ndarray:
        values = x @ self.weights[f"{name}.weight"].T
        return self._activate(values, self.weights[f"{name}.bias"], kernels)

    def forward(
        self,
        obs: dict[str, np.ndarray],
        *,
        neighbors: np.ndarray | None = None,
        board_shape: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Return masked logits and value for one graph, including optional padding."""
        x, context = obs["x"], obs["context"]
        features, contexts = dimensions(self.structure)
        if x.shape[-1] != features or context.shape != (contexts,):
            raise ValueError("Observation features do not match the checkpoint encoding.")
        mask = obs["node_mask"][:, None]
        kernels = (
            _kernels
            if self.native and x.dtype == np.float32 and context.dtype == np.float32
            else None
        )
        node_mask = np.ascontiguousarray(obs["node_mask"], dtype=np.float32)
        full = bool(np.all(mask == 1))
        broadcast = np.broadcast_to(context, (len(x), len(context)))
        h = self._linear_activation(np.concatenate((x, broadcast), axis=-1), "encoder.0", kernels)
        if not full:
            h *= mask
        if board_shape is not None:
            if neighbors is not None:
                raise ValueError("Choose grid messages or neighbor indices, not both.")
            rows, cols = board_shape
            if min(rows, cols) < 1 or len(x) != 3 * rows * cols + rows + cols or not full:
                raise ValueError("Grid messages require an unpadded board observation.")
            messages = np.empty(h.shape, dtype=np.float32)
        elif neighbors is not None:
            if neighbors.shape != (len(x), 4):
                raise ValueError("Sparse neighbor shape does not match the observation.")
            padded = np.zeros((len(x) + 1, self.width), dtype=np.float32)
        for index in range(self.depth):
            prefix = f"blocks.{index}"
            if board_shape is not None and kernels is not None:
                kernels.aggregate(h, messages, *board_shape, self.weights[f"{prefix}.epsilon"])
                aggregate = messages
            elif board_shape is not None:
                grid_messages(h, *board_shape, messages)
            elif neighbors is None:
                messages = obs["adjacency"] @ h
            else:
                padded[:-1] = h
                messages = padded[neighbors[:, 0]] + padded[neighbors[:, 1]]
                messages += padded[neighbors[:, 2]]
                messages += padded[neighbors[:, 3]]
            if board_shape is None or kernels is None:
                aggregate = (np.float32(1) + self.weights[f"{prefix}.epsilon"]) * h
                aggregate += messages
            hidden = self._linear_activation(aggregate, f"{prefix}.mlp.0", kernels)
            if kernels is not None:
                residual = hidden @ self.weights[f"{prefix}.mlp.2.weight"].T
                kernels.normalize(
                    residual,
                    self.weights[f"{prefix}.mlp.2.bias"],
                    h,
                    self.weights[f"{prefix}.norm.weight"],
                    self.weights[f"{prefix}.norm.bias"],
                    node_mask,
                )
                h = residual
                continue
            residual = self.linear(hidden, f"{prefix}.mlp.2")
            residual += h
            residual -= residual.mean(axis=-1, keepdims=True)
            centered = residual
            denominator = np.mean(centered * centered, axis=-1, keepdims=True) + np.float32(1e-5)
            np.sqrt(denominator, out=denominator)
            centered /= denominator
            centered *= self.weights[f"{prefix}.norm.weight"]
            centered += self.weights[f"{prefix}.norm.bias"]
            if not full:
                centered *= mask
            h = centered
        mean = h.sum(axis=0) / max(mask.sum(), np.float32(1))
        maximum = h.max(axis=0) if full else np.where(mask != 0, h, np.float32(-1e9)).max(axis=0)
        readout = np.concatenate((mean, maximum, context))
        if neighbors is None and board_shape is None:
            shared = np.broadcast_to(readout, (len(x), len(readout)))
            logits = self.linear(
                self._linear_activation(np.concatenate((h, shared), axis=-1), "policy.0", kernels),
                "policy.2",
            ).ravel()
            logits = np.where(obs["action_mask"] != 0, logits, np.float32(-1e9))
        else:
            # The global readout is identical for every node. Project it once,
            # and evaluate the policy head only at legal edge nodes.
            actions = np.flatnonzero(obs["action_mask"])
            weight = self.weights["policy.0.weight"]
            shared = readout @ weight[:, self.width :].T + self.weights["policy.0.bias"]
            hidden = self._activate(h[actions] @ weight[:, : self.width].T, shared, kernels)
            logits = np.full(len(x), np.float32(-1e9), dtype=np.float32)
            logits[actions] = self.linear(hidden, "policy.2").ravel()
        value = np.tanh(
            self.linear(self._linear_activation(readout, "value.0", kernels), "value.2")
        )[0]
        return logits, float(value)


class NumpyEvaluator(CachedEvaluator):
    def __init__(self, network: NumpyNetwork, cache_size: int = 4096):
        super().__init__(cache_size)
        self.network = network

    def _predict(self, state: State) -> tuple[np.ndarray, float]:
        sparse = state.board.num_nodes >= 256
        logits, value = self.network.forward(
            encode(state, structure=self.network.structure, include_adjacency=not sparse),
            board_shape=(state.board.rows, state.board.cols) if sparse else None,
        )
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
    if module.structure:
        config["structure"] = True
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
