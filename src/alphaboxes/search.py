"""PUCT search with perspective-aware backups and optional exact endgames."""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import TYPE_CHECKING, Protocol

import numpy as np

from alphaboxes.chain_moves import allowed_actions
from alphaboxes.endgame import EndgameTimeout, shared_solver
from alphaboxes.game import State
from alphaboxes.graph import encode, encode_batch
from alphaboxes.opponents import safe_actions, solve
from alphaboxes.root_halving import RootHalving
from alphaboxes.selection import ChildSelection

try:
    if os.environ.get("ALPHABOXES_PURE_NUMPY") == "1":
        raise ImportError
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
            tensor_observations([encode(states[0], structure=self.module.structure)])
            if len(states) == 1
            else {
                key: torch.from_numpy(value)
                for key, value in encode_batch(states, structure=self.module.structure).items()
            }
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
    reuse_tree: bool = False
    time_limit: float | None = None
    fpu_reduction: float = 0.0
    fpu_floor: float | None = None
    fpu_mode: str = "constant"
    fpu_parent: str = "network"
    root_candidates: int = 0
    degree_two: bool = False
    prove_results: bool = False
    chain_pruning: bool = False
    sparse_selection: bool = False
    safe_moves: bool = False

    def __post_init__(self):
        if self.time_limit is not None and (
            not np.isfinite(self.time_limit) or self.time_limit <= 0
        ):
            raise ValueError("Thinking time must be finite and positive.")
        if self.simulations < 1 or self.cpuct <= 0 or self.dirichlet_alpha <= 0:
            raise ValueError("Search budgets and exploration constants must be positive.")
        if not np.isfinite(self.fpu_reduction) or self.fpu_reduction < 0:
            raise ValueError("First-play reduction must be finite and nonnegative.")
        if self.fpu_floor is not None and (
            not np.isfinite(self.fpu_floor) or not -1 <= self.fpu_floor <= 1
        ):
            raise ValueError("First-play floor must be finite and within [-1, 1].")
        if self.fpu_mode not in {"constant", "policy_mass", "branching"}:
            raise ValueError("First-play mode must be constant, policy_mass or branching.")
        if self.fpu_parent not in {"network", "mean"}:
            raise ValueError("First-play parent must be network or mean.")
        if not isinstance(self.root_candidates, int) or not 0 <= self.root_candidates <= 512:
            raise ValueError("Root candidates must be an integer in 0..512.")
        if self.root_candidates and (self.reuse_tree or self.prove_results):
            raise ValueError("Root halving requires fresh trees without proof propagation.")
        if self.safe_moves and self.prove_results:
            raise ValueError("Safe-move filtering is heuristic and cannot prove results.")
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
    evaluation: float = 0.0
    chain_restricted: bool = False
    selection: ChildSelection | None = None

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def perspective(value: float, from_player: int, to_player: int) -> float:
    return value if from_player == to_player else -value


class MCTS:
    def __init__(
        self,
        evaluator: Evaluator,
        config: SearchConfig | None = None,
        seed: int = 0,
        *,
        root_observer: Callable[[Node], None] | None = None,
    ):
        self.evaluator = evaluator
        self.root_observer = root_observer
        self.config = config or SearchConfig()
        self.rng = np.random.default_rng(seed)
        self._solved: OrderedDict[tuple, float] = OrderedDict()
        self._roots: list[Node] = []
        self.inherited_visits = 0
        self.completed_simulations = 0
        self.endgame_hits = 0
        self.proven_nodes = 0
        self._deadline = None

    def clear_tree(self) -> None:
        """Discard visits when weights change; exact game-rule values remain valid."""
        self._roots.clear()

    @staticmethod
    def _descendant(root: Node, state: State) -> Node | None:
        if root.state.board != state.board:
            return None
        wanted = CachedEvaluator._key(state)
        pending = [root]
        while pending:
            node = pending.pop()
            if node.state is None:
                continue
            key = CachedEvaluator._key(node.state)
            if key[2] & ~wanted[2]:
                continue
            if key == wanted and node.player == state.player:
                # Bound repeated searches without a move, such as repeated UI hints.
                # A value-only solved leaf has no action strategy. Search it as
                # a fresh root if the root oracle cannot finish an action solve.
                return node if node.visits < 8192 and node.children is not None else None
            if node.children:
                pending.extend(node.children.values())
        return None

    def _expand(self, node: Node, prediction=None) -> float:
        if node.state.terminal:
            return node.state.outcome(node.state.player)
        legal = node.state.legal_actions
        if self.config.chain_pruning:
            reduced = allowed_actions(node.state)
            node.chain_restricted = len(reduced) < len(legal)
            legal = reduced
        if self.config.safe_moves:
            safe = set(safe_actions(node.state))
            legal = tuple(action for action in legal if action in safe)
        probabilities, value = prediction if prediction is not None else self.evaluator(node.state)
        priors = np.maximum(probabilities[list(legal)], 0).astype(np.float64)
        priors = priors / priors.sum() if priors.sum() > 0 else np.full(len(legal), 1 / len(legal))
        node.children = {
            action: Node(
                None,
                float(prior),
                # Unvisited values are zero regardless of perspective. Determine
                # the next player only when this edge is actually traversed.
                player=-1,
                action=action,
            )
            for action, prior in zip(legal, priors, strict=True)
        }
        node.evaluation = float(value)
        return node.evaluation

    def policy(self, state: State, *, explore: bool = False) -> tuple[np.ndarray, float]:
        return self.policies([state], explore=explore)[0]

    def _expand_many(self, nodes: list[Node], *, allow_solve: bool = True) -> list[float]:
        values: list[float | None] = [None] * len(nodes)
        pending = []
        for index, node in enumerate(nodes):
            if node.state.terminal:
                values[index] = node.state.outcome(node.player)
                node.solved_value = values[index]
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
                margin = None
                if self.config.degree_two and allow_solve:
                    deadline = monotonic() + 0.003
                    if self._deadline is not None:
                        deadline = min(deadline, self._deadline)
                    try:
                        margin = shared_solver.value(node.state, deadline)
                    except EndgameTimeout:
                        pass
                if margin is None:
                    pending.append(index)
                else:
                    self.endgame_hits += 1
                    node.solved_value = float(np.sign(margin))
                    values[index] = node.solved_value
        nonterminal = [nodes[index].state for index in pending]
        batch = getattr(self.evaluator, "evaluate_many", None)
        predictions = (
            batch(nonterminal) if batch else [self.evaluator(state) for state in nonterminal]
        )
        for index, prediction in zip(pending, predictions, strict=True):
            values[index] = self._expand(nodes[index], prediction)
        return values

    def _prove(self, path: list[Node]) -> None:
        """Propagate exact outcomes, respecting captures that retain the turn."""
        for node in reversed(path):
            if node.solved_value is not None or not node.children:
                continue
            known = [
                perspective(child.solved_value, child.player, node.player)
                for child in node.children.values()
                if child.solved_value is not None
            ]
            if 1.0 in known or len(known) == len(node.children):
                node.solved_value = max(known)
                self.proven_nodes += 1

    def _select(self, node: Node, scale: float) -> Node:
        children = node.children.values()
        if node.chain_restricted and len(node.children) == 2 and not self.config.prove_results:
            unvisited = [child for child in children if not child.visits]
            if unvisited:
                return max(unvisited, key=lambda child: child.prior)
        reduction = self.config.fpu_reduction
        if self.config.fpu_mode == "policy_mass":
            visited_mass = sum(child.prior for child in children if child.visits)
            reduction *= np.sqrt(visited_mass)
        elif self.config.fpu_mode == "branching":
            reduction *= min(1.0, np.sqrt(54 / len(node.children)))
        parent_value = (
            node.value if self.config.fpu_parent == "mean" and node.visits else node.evaluation
        )
        first_play = parent_value - reduction
        if self.config.fpu_floor is not None:
            first_play = max(self.config.fpu_floor, first_play)
        if (
            self.config.sparse_selection
            and self.config.fpu_mode == "constant"
            and not self.config.prove_results
            and len(node.children) >= 16
            and isfinite(scale)
            and scale > 0
            and isfinite(first_play)
        ):
            if (
                node.selection is None
                or node.selection.children is not node.children
                or node.selection.size != len(node.children)
            ):
                node.selection = ChildSelection(node.children)
            return node.selection.select(
                node.player, scale, first_play if self.config.fpu_reduction else 0.0
            )
        if self.config.prove_results:
            unresolved = [child for child in children if child.solved_value is None]
            if unresolved:
                children = unresolved
        return max(
            children,
            key=lambda child: (
                (
                    perspective(child.value, child.player, node.player)
                    if child.visits or not self.config.fpu_reduction
                    else first_play
                )
                + scale * child.prior / (1 + child.visits)
            ),
        )

    def policies(
        self, states: list[State], *, explore: bool = False
    ) -> list[tuple[np.ndarray, float]]:
        """Search independent games together, with one leaf per tree per simulation."""
        if any(state.terminal for state in states):
            raise ValueError("Cannot search a finished game.")
        if self.config.root_candidates and explore:
            raise ValueError(
                "Root halving is an acting experiment; self-play uses standard search."
            )
        deadline = monotonic() + self.config.time_limit if self.config.time_limit else None
        self._deadline = deadline
        self.completed_simulations = 0
        self.endgame_hits = self.proven_nodes = 0
        results = [(np.zeros(state.board.num_edges, dtype=np.float32), 0.0) for state in states]
        roots = []
        reused = set()
        for index, state in enumerate(states):
            if self.config.degree_two:
                exact = None
                root_deadline = monotonic() + 0.05
                if deadline is not None:
                    root_deadline = min(root_deadline, deadline)
                try:
                    exact = shared_solver.solve(state, root_deadline)
                except EndgameTimeout:
                    pass
                if exact is not None:
                    margin, actions = exact
                    results[index][0][list(actions)] = 1 / len(actions)
                    results[index] = results[index][0], float(np.sign(margin))
                    self.endgame_hits += 1
                    continue
            if max(self.config.exact_threshold, self.config.leaf_exact_threshold) and len(
                state.legal_actions
            ) <= max(self.config.exact_threshold, self.config.leaf_exact_threshold):
                margin, actions = solve(state)
                results[index][0][list(actions)] = 1 / len(actions)
                results[index] = results[index][0], float(np.sign(margin))
            else:
                root = None
                if self.config.reuse_tree:
                    for previous, old in enumerate(self._roots):
                        if previous not in reused:
                            root = self._descendant(old, state)
                            if root is not None:
                                reused.add(previous)
                                break
                if root is None:
                    root = Node(state, player=state.player)
                else:
                    root.state = state
                roots.append((index, root))
        self.inherited_visits = sum(root.visits for _, root in roots)
        self._roots = [root for _, root in roots] if self.config.reuse_tree else []
        fresh = [root for _, root in roots if root.children is None]
        self._expand_many(fresh, allow_solve=False)
        if not roots:
            return results
        for _, root in roots:
            root.selection = None
            if self.config.reuse_tree and root.visits:
                # A repeated exploratory search must start from network priors,
                # rather than mixing new noise into yesterday's noisy priors.
                probabilities, _ = self.evaluator(root.state)
                priors = np.maximum(probabilities[list(root.children)], 0).astype(np.float64)
                priors = (
                    priors / priors.sum()
                    if priors.sum() > 0
                    else np.full(len(priors), 1 / len(priors))
                )
                for child, prior in zip(root.children.values(), priors, strict=True):
                    child.prior = float(prior)
            if not explore:
                continue
            # Scale concentration with branching factor as board sizes change.
            alpha = self.config.dirichlet_alpha * 24 / len(root.children)
            noise = self.rng.dirichlet(np.full(len(root.children), alpha))
            fraction = self.config.noise_fraction
            for child, sample in zip(root.children.values(), noise, strict=True):
                child.prior = (1 - fraction) * child.prior + fraction * float(sample)
        halving = (
            {
                id(root): RootHalving(
                    root.children,
                    self.config.simulations,
                    self.config.root_candidates,
                    lambda child, player=root.player: perspective(
                        child.value, child.player, player
                    ),
                )
                for _, root in roots
            }
            if self.config.root_candidates
            else {}
        )
        for _ in range(self.config.simulations):
            if deadline is not None and monotonic() >= deadline:
                break
            paths = []
            for _, root in roots:
                if self.config.prove_results and root.solved_value is not None:
                    continue
                node, path = root, [root]
                while node.children:
                    scale = self.config.cpuct * np.sqrt(node.visits + 1)
                    if node is root and halving:
                        child = halving[id(root)].select()
                    else:
                        child = (
                            choose_child(node.children, node.player, float(scale))
                            if choose_child
                            and not self.config.fpu_reduction
                            and not self.config.prove_results
                            and not (node.chain_restricted and len(node.children) == 2)
                            else self._select(node, scale)
                        )
                    if child.state is None:
                        child.state = node.state.play(child.action)
                        child.player = child.state.player
                    node = child
                    path.append(node)
                paths.append(path)
            if not paths:
                break
            values = self._expand_many([path[-1] for path in paths])
            for path, value in zip(paths, values, strict=True):
                for ancestor in path:
                    ancestor.visits += 1
                    ancestor.value_sum += perspective(value, path[-1].player, ancestor.player)
                if self.config.prove_results:
                    self._prove(path)
            self.completed_simulations += 1
        for index, root in roots:
            eligible = root.children
            if halving:
                best = halving[id(root)].best_action()
                eligible = {best: root.children[best]}
            if self.config.prove_results and root.solved_value is not None:
                eligible = {
                    action: child
                    for action, child in root.children.items()
                    if child.solved_value is not None
                    and perspective(child.solved_value, child.player, root.player)
                    == root.solved_value
                }
            elif self.config.prove_results:
                safe = {
                    action: child
                    for action, child in root.children.items()
                    if child.solved_value is None
                    or perspective(child.solved_value, child.player, root.player) >= 0
                }
                eligible = safe or eligible
            total = sum(child.visits for child in eligible.values())
            prior_total = sum(child.prior for child in eligible.values())
            for action, child in eligible.items():
                results[index][0][action] = child.visits / total if total else child.prior
                if not total:
                    results[index][0][action] /= prior_total
            value = root.value if root.visits else root.evaluation
            if self.config.prove_results and root.solved_value is not None:
                value = root.solved_value
            results[index] = results[index][0], value
            if self.root_observer is not None:
                self.root_observer(root)
        return results

    def action(self, state: State) -> int:
        probabilities, _ = self.policy(state)
        return int(np.argmax(probabilities))
