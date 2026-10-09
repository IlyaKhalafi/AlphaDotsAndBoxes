"""Edge/box incidence graphs, with no coordinates or size-dependent weights."""

from functools import lru_cache

import gymnasium as gym
import numpy as np

from alphaboxes.game import State, board

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
    geometry = state.board
    n, e = geometry.num_nodes, geometry.num_edges
    capacity = n if capacity is None else capacity
    if capacity < n:
        raise ValueError("Observation capacity is smaller than this board.")
    x = np.zeros((capacity, FEATURES), dtype=np.float32)
    x[:e, 0] = 1  # edge node
    x[e:n, 1] = 1  # box node
    x[:e, 2] = np.asarray(state.edges) >= 0
    remaining = np.array([sum(state.edges[j] < 0 for j in edges) for edges in geometry.boxes])
    x[e:n, 3] = np.asarray(state.owners) == state.player
    x[e:n, 4] = np.asarray(state.owners) == 1 - state.player
    x[e + np.arange(geometry.num_boxes), 5 + remaining] = 1  # free sides: 0..4
    for edge, neighbors in enumerate(geometry.edge_boxes):
        x[edge, 10] = len(neighbors) / 2
        x[edge, 11] = np.count_nonzero(remaining[list(neighbors)] == 1) / 2
    scores = state.scores
    context = np.array(
        [
            (scores[state.player] - scores[1 - state.player]) / geometry.num_boxes,
            len(state.legal_actions) / e,
            sum(owner < 0 for owner in state.owners) / geometry.num_boxes,
        ],
        dtype=np.float32,
    )
    adj = np.zeros((capacity, capacity), dtype=np.float32)
    adj[:n, :n] = adjacency(geometry.rows, geometry.cols)
    mask = np.zeros(capacity, dtype=np.float32)
    mask[:e] = np.asarray(state.edges) < 0
    node_mask = np.zeros(capacity, dtype=np.float32)
    node_mask[:n] = 1
    return {
        "x": x,
        "adjacency": adj,
        "action_mask": mask,
        "node_mask": node_mask,
        "context": context,
    }


def spaces(capacity: int) -> tuple[gym.spaces.Dict, gym.spaces.Discrete]:
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
