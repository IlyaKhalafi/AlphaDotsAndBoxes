"""PUCT search with perspective-aware backups and optional exact endgames."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from alphaboxes.game import State
from alphaboxes.graph import encode, encode_batch
from alphaboxes.opponents import solve

try:
    from alphaboxes._search import choose_child
except ImportError:
    choose_child = None

if TYPE_CHECKING:
    from alphaboxes.network import GraphModule


class Evaluator(Protocol):
    def __call__(self, state: State) -> tuple[np.ndarray, float]: ...


class CachedEvaluator:
    def __init__(self, cache_size: int = 4096):
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple, tuple[np.ndarray, float]] = OrderedDict()

    def clear_cache(self) -> None:
        """Call after replacing weights; values are specific to one checkpoint."""
        self._cache.clear()

    @staticmethod
    def _key(state: State) -> tuple:
        # Edge authors do not affect play; box authors are relative to the active player.
        return (
            state.board.rows,
            state.board.cols,
            sum(1 << i for i, owner in enumerate(state.edges) if owner >= 0),
            sum(1 << i for i, owner in enumerate(state.owners) if owner == state.player),
            sum(1 << i for i, owner in enumerate(state.owners) if owner == 1 - state.player),
        )

    def _remember(self, key: tuple, prediction: tuple[np.ndarray, float]) -> None:
        prediction[0].flags.writeable = False
        if self.cache_size > 0:
            self._cache[key] = prediction
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def __call__(self, state: State) -> tuple[np.ndarray, float]:
        key = self._key(state)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        prediction = self._predict(state)
        self._remember(key, prediction)
        return prediction

    def evaluate_many(self, states: list[State]) -> list[tuple[np.ndarray, float]]:
        """Batch distinct cache misses, preserving input order across evictions."""
        keys = [self._key(state) for state in states]
        found, missing = {}, {}
        for key, state in zip(keys, states, strict=True):
            if key in self._cache:
                found[key] = self._cache[key]
                self._cache.move_to_end(key)
            else:
                missing[key] = state
        for key, prediction in zip(
            missing, self._predict_many(list(missing.values())), strict=True
        ):
            self._remember(key, prediction)
            found[key] = prediction
        return [found[key] for key in keys]

    def _predict(self, state: State) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def _predict_many(self, states: list[State]) -> list[tuple[np.ndarray, float]]:
        return [self._predict(state) for state in states]


class NeuralEvaluator(CachedEvaluator):
    """Training-side PyTorch adapter; importing search never imports PyTorch."""

    def __init__(
        self,
        module: GraphModule,
        cache_size: int = 4096,
        cuda_batch_size: int | None = None,
        capacity: int | None = None,
    ):
        super().__init__(cache_size)
        self.module = module.eval()
        self.cuda_inference = None
        if cuda_batch_size is not None:
            from alphaboxes.cuda_inference import CudaInference

            if next(module.parameters()).device.type != "cuda" or capacity is None:
                raise ValueError("CUDA graph prediction requires a GPU module and graph capacity.")
            self.cuda_inference = CudaInference(module, cuda_batch_size, capacity)

    def _predict(self, state: State) -> tuple[np.ndarray, float]:
        return self._predict_many([state])[0]

    def _predict_many(self, states: list[State]) -> list[tuple[np.ndarray, float]]:
        if not states:
            return []
        if self.cuda_inference is not None:
            return self.cuda_inference.predict(states)
        import torch
        from ray.rllib.core.columns import Columns

        from alphaboxes.network import tensor_observations

        obs = (
            tensor_observations([encode(states[0])])
            if len(states) == 1
            else {key: torch.from_numpy(value) for key, value in encode_batch(states).items()}
        )
        device = next(self.module.parameters()).device
        obs = {key: value.to(device) for key, value in obs.items()}
        result = self.module.predict(obs)
        probabilities = result[Columns.ACTION_DIST_INPUTS].softmax(dim=-1).cpu().numpy()
        values = result[Columns.VF_PREDS].cpu().numpy()
        return [
            (probabilities[i, : state.board.num_edges], float(values[i]))
            for i, state in enumerate(states)
        ]


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 64
    cpuct: float = 1.5
    noise_fraction: float = 0.25
    dirichlet_alpha: float = 0.3
    exact_threshold: int = 0
    leaf_exact_threshold: int = 0

    def __post_init__(self):
        if self.simulations < 1 or self.cpuct <= 0 or self.dirichlet_alpha <= 0:
            raise ValueError("Search budgets and exploration constants must be positive.")
        if (
            not 0 <= self.noise_fraction <= 1
            or not 0 <= self.exact_threshold <= 18
            or not 0 <= self.leaf_exact_threshold <= 18
        ):
            raise ValueError("Invalid noise fraction or exact endgame threshold (0..18).")


@dataclass(slots=True)
class Node:
    state: State | None
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, Node] | None = None
    player: int = 0
    action: int = -1
    solved_value: float | None = None

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
        self._solved: OrderedDict[tuple, float] = OrderedDict()

    def _expand(self, node: Node, prediction=None) -> float:
        if node.state.terminal:
            return node.state.outcome(node.state.player)
        legal = node.state.legal_actions
        probabilities, value = prediction if prediction is not None else self.evaluator(node.state)
        priors = np.maximum(probabilities[list(legal)], 0).astype(np.float64)
        priors = priors / priors.sum() if priors.sum() > 0 else np.full(len(legal), 1 / len(legal))
        node.children = {
            action: Node(
                None,
                float(prior),
                player=node.state.player if node.state.captures(action) else 1 - node.state.player,
                action=action,
            )
            for action, prior in zip(legal, priors, strict=True)
        }
        return float(value)

    def policy(self, state: State, *, explore: bool = False) -> tuple[np.ndarray, float]:
        return self.policies([state], explore=explore)[0]

    def _expand_many(self, nodes: list[Node]) -> list[float]:
        values: list[float | None] = [None] * len(nodes)
        pending = []
        for index, node in enumerate(nodes):
            if node.state.terminal:
                values[index] = node.state.outcome(node.player)
            elif node.solved_value is not None:
                values[index] = node.solved_value
            elif (
                self.config.leaf_exact_threshold
                and len(node.state.legal_actions) <= self.config.leaf_exact_threshold
            ):
                key = CachedEvaluator._key(node.state)
                if key not in self._solved:
                    margin, _ = solve(node.state)
                    self._solved[key] = float(np.sign(margin))
                    if len(self._solved) > 4096:
                        self._solved.popitem(last=False)
                self._solved.move_to_end(key)
                node.solved_value = self._solved[key]
                values[index] = node.solved_value
            else:
                pending.append(index)
        nonterminal = [nodes[index].state for index in pending]
        batch = getattr(self.evaluator, "evaluate_many", None)
        predictions = (
            batch(nonterminal) if batch else [self.evaluator(state) for state in nonterminal]
        )
        for index, prediction in zip(pending, predictions, strict=True):
            values[index] = self._expand(nodes[index], prediction)
        return values

    def policies(
        self, states: list[State], *, explore: bool = False
    ) -> list[tuple[np.ndarray, float]]:
        """Search independent games together, with one leaf per tree per simulation."""
        if any(state.terminal for state in states):
            raise ValueError("Cannot search a finished game.")
        results = [(np.zeros(state.board.num_edges, dtype=np.float32), 0.0) for state in states]
        roots = []
        for index, state in enumerate(states):
            if max(self.config.exact_threshold, self.config.leaf_exact_threshold) and len(
                state.legal_actions
            ) <= max(self.config.exact_threshold, self.config.leaf_exact_threshold):
                margin, actions = solve(state)
                results[index][0][list(actions)] = 1 / len(actions)
                results[index] = results[index][0], float(np.sign(margin))
            else:
                roots.append((index, Node(state, player=state.player)))
        self._expand_many([root for _, root in roots])
        if not roots:
            return results
        for _, root in roots if explore else ():
            # Scale concentration with branching factor as board sizes change.
            alpha = self.config.dirichlet_alpha * 24 / len(root.children)
            noise = self.rng.dirichlet(np.full(len(root.children), alpha))
            fraction = self.config.noise_fraction
            for child, sample in zip(root.children.values(), noise, strict=True):
                child.prior = (1 - fraction) * child.prior + fraction * float(sample)
        for _ in range(self.config.simulations):
            paths = []
            for _, root in roots:
                node, path = root, [root]
                while node.children:
                    scale = self.config.cpuct * np.sqrt(node.visits + 1)
                    child = (
                        choose_child(node.children, node.player, float(scale))
                        if choose_child
                        else max(
                            node.children.values(),
                            key=lambda child: (
                                perspective(child.value, child.player, node.player)
                                + scale * child.prior / (1 + child.visits)
                            ),
                        )
                    )
                    if child.state is None:
                        child.state = node.state.play(child.action)
                    node = child
                    path.append(node)
                paths.append(path)
            values = self._expand_many([path[-1] for path in paths])
            for path, value in zip(paths, values, strict=True):
                for ancestor in path:
                    ancestor.visits += 1
                    ancestor.value_sum += perspective(value, path[-1].player, ancestor.player)
        for index, root in roots:
            total = sum(child.visits for child in root.children.values())
            for action, child in root.children.items():
                results[index][0][action] = child.visits / total
            results[index] = results[index][0], root.value
        return results

    def action(self, state: State) -> int:
        probabilities, _ = self.policy(state)
        return int(np.argmax(probabilities))
