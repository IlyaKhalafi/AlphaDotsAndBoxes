"""PUCT search with perspective-aware backups and optional exact endgames."""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from ray.rllib.core.columns import Columns

from alphaboxes.game import State
from alphaboxes.graph import encode
from alphaboxes.network import GraphModule, tensor_observations
from alphaboxes.opponents import solve


class Evaluator(Protocol):
    def __call__(self, state: State) -> tuple[np.ndarray, float]: ...


class NeuralEvaluator:
    def __init__(self, module: GraphModule):
        self.module = module.eval()

    def __call__(self, state: State) -> tuple[np.ndarray, float]:
        obs = tensor_observations([encode(state)])
        device = next(self.module.parameters()).device
        obs = {key: value.to(device) for key, value in obs.items()}
        with torch.inference_mode():
            result = self.module.forward_inference({Columns.OBS: obs})
            probabilities = result[Columns.ACTION_DIST_INPUTS].softmax(dim=-1)[0]
        return probabilities[: state.board.num_edges].cpu().numpy(), float(
            result[Columns.VF_PREDS][0]
        )


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 64
    cpuct: float = 1.5
    noise_fraction: float = 0.25
    dirichlet_alpha: float = 0.3
    exact_threshold: int = 0

    def __post_init__(self):
        if self.simulations < 1 or self.cpuct <= 0 or self.dirichlet_alpha <= 0:
            raise ValueError("Search budgets and exploration constants must be positive.")
        if not 0 <= self.noise_fraction <= 1 or not 0 <= self.exact_threshold <= 18:
            raise ValueError("Invalid noise fraction or exact endgame threshold (0..18).")


@dataclass
class Node:
    state: State
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def perspective(value: float, from_player: int, to_player: int) -> float:
    return value if from_player == to_player else -value


class MCTS:
    def __init__(self, evaluator: Evaluator, config: SearchConfig | None = None, seed: int = 0):
        self.evaluator = evaluator
        self.config = config or SearchConfig()
        self.rng = np.random.default_rng(seed)

    def _expand(self, node: Node) -> float:
        if node.state.terminal:
            return node.state.outcome(node.state.player)
        legal = node.state.legal_actions
        probabilities, value = self.evaluator(node.state)
        priors = np.maximum(probabilities[list(legal)], 0).astype(np.float64)
        priors = priors / priors.sum() if priors.sum() > 0 else np.full(len(legal), 1 / len(legal))
        node.children = {
            action: Node(node.state.play(action), float(prior))
            for action, prior in zip(legal, priors, strict=True)
        }
        return float(value)

    def policy(self, state: State, *, explore: bool = False) -> tuple[np.ndarray, float]:
        if state.terminal:
            raise ValueError("Cannot search a finished game.")
        result = np.zeros(state.board.num_edges, dtype=np.float32)
        if self.config.exact_threshold and len(state.legal_actions) <= self.config.exact_threshold:
            margin, actions = solve(state)
            result[list(actions)] = 1 / len(actions)
            return result, float(np.sign(margin))
        root = Node(state)
        self._expand(root)
        if explore:
            # Scale concentration with branching factor as board sizes change.
            alpha = self.config.dirichlet_alpha * 24 / len(root.children)
            noise = self.rng.dirichlet(np.full(len(root.children), alpha))
            fraction = self.config.noise_fraction
            for child, sample in zip(root.children.values(), noise, strict=True):
                child.prior = (1 - fraction) * child.prior + fraction * float(sample)
        for _ in range(self.config.simulations):
            node = root
            path = [node]
            while node.children:
                scale = self.config.cpuct * np.sqrt(node.visits + 1)
                node = max(
                    node.children.values(),
                    key=lambda child: (
                        perspective(child.value, child.state.player, node.state.player)
                        + scale * child.prior / (1 + child.visits)
                    ),
                )
                path.append(node)
            value = self._expand(node)
            leaf_player = node.state.player
            for ancestor in path:
                ancestor.visits += 1
                ancestor.value_sum += perspective(value, leaf_player, ancestor.state.player)
        total = sum(child.visits for child in root.children.values())
        for action, child in root.children.items():
            result[action] = child.visits / total
        return result, root.value

    def action(self, state: State) -> int:
        probabilities, _ = self.policy(state)
        return int(np.argmax(probabilities))
