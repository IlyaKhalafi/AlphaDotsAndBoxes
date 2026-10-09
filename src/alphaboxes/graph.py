"""Edge/box incidence graphs, with no coordinates or size-dependent weights."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from alphaboxes.game import State, board

if TYPE_CHECKING:
    import gymnasium as gym

FEATURES = 12


@lru_cache(maxsize=128)
def adjacency(rows: int, cols: int) -> np.ndarray:
    geometry = board(rows, cols)
    result = np.zeros((geometry.num_nodes, geometry.num_nodes), dtype=np.float32)
    for b, edges in enumerate(geometry.boxes):
        for e in edges:
            result[e, geometry.num_edges + b] = 1
            result[geometry.num_edges + b, e] = 1
    result.flags.writeable = False
    return result


def encode(state: State, capacity: int | None = None) -> dict[str, np.ndarray]:
    return {key: value[0] for key, value in encode_batch([state], capacity).items()}


def encode_batch(states: Sequence[State], capacity: int | None = None) -> dict[str, np.ndarray]:
    """Vectorize each board shape, keeping batch order and zero-filled padding."""
    if not states:
        raise ValueError("Cannot encode an empty batch.")
    required = max(state.board.num_nodes for state in states)
    capacity = required if capacity is None else capacity
    if capacity < required:
        raise ValueError("Observation capacity is smaller than this board.")
    count = len(states)
    result = {
        "x": np.zeros((count, capacity, FEATURES), dtype=np.float32),
        "adjacency": np.zeros((count, capacity, capacity), dtype=np.float32),
        "action_mask": np.zeros((count, capacity), dtype=np.float32),
        "node_mask": np.zeros((count, capacity), dtype=np.float32),
        "context": np.zeros((count, 3), dtype=np.float32),
    }
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, state in enumerate(states):
        groups[(state.board.rows, state.board.cols)].append(index)
    for shape, indices in groups.items():
        geometry = board(*shape)
        n, e = geometry.num_nodes, geometry.num_edges
        edges = np.array([states[i].edges for i in indices], dtype=np.int8)
        owners = np.array([states[i].owners for i in indices], dtype=np.int8)
        players = np.array([states[i].player for i in indices], dtype=np.int8)[:, None]
        free = edges < 0
        remaining = free[:, np.asarray(geometry.boxes)].sum(axis=-1)
        ours, theirs = owners == players, owners == 1 - players
        x = np.zeros((len(indices), n, FEATURES), dtype=np.float32)
        x[:, :e, 0] = 1
        x[:, e:, 1] = 1
        x[:, :e, 2] = ~free
        x[:, e:, 3], x[:, e:, 4] = ours, theirs
        x[np.arange(len(indices))[:, None], e + np.arange(geometry.num_boxes), 5 + remaining] = 1
        adj = adjacency(*shape)
        incidence = adj[:e, e:n]
        x[:, :e, 10] = incidence.sum(axis=1) / 2
        x[:, :e, 11] = (remaining == 1).astype(np.float32) @ incidence.T / 2
        result["x"][indices, :n] = x
        result["adjacency"][indices, :n, :n] = adj
        result["action_mask"][indices, :e] = free
        result["node_mask"][indices, :n] = 1
        result["context"][indices, 0] = (ours.sum(axis=1) - theirs.sum(axis=1)) / geometry.num_boxes
        result["context"][indices, 1] = free.sum(axis=1) / e
        result["context"][indices, 2] = (owners < 0).sum(axis=1) / geometry.num_boxes
    return result


def spaces(capacity: int) -> tuple[gym.spaces.Dict, gym.spaces.Discrete]:
    import gymnasium as gym

    observation = gym.spaces.Dict(
        {
            "x": gym.spaces.Box(0, 1, (capacity, FEATURES), np.float32),
            "adjacency": gym.spaces.Box(0, 1, (capacity, capacity), np.float32),
            "action_mask": gym.spaces.Box(0, 1, (capacity,), np.float32),
            "node_mask": gym.spaces.Box(0, 1, (capacity,), np.float32),
            "context": gym.spaces.Box(-1, 1, (3,), np.float32),
        }
    )
    return observation, gym.spaces.Discrete(capacity)
