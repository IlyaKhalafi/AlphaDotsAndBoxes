"""Time-bounded iterative-deepening alpha-beta search with a GIN evaluator.

All legal moves remain available, including non-capturing chain handouts. Depth
counts drawn edges; negation occurs only when the active player changes.
"""

from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic

import numpy as np

from alphaboxes.game import State
from alphaboxes.opponents import solve
from alphaboxes.search import CachedEvaluator, Evaluator


@dataclass(frozen=True)
class EngineConfig:
    time_limit: float = 1.0
    max_depth: int = 64
    table_size: int = 16384
    exact_threshold: int = 12

    def __post_init__(self):
        if not np.isfinite(self.time_limit) or self.time_limit <= 0:
            raise ValueError("Thinking time must be finite and positive.")
        if self.max_depth < 1 or self.table_size < 0 or not 0 <= self.exact_threshold <= 18:
            raise ValueError("Invalid engine depth, table size, or endgame threshold.")


@dataclass(frozen=True)
class Entry:
    depth: int
    value: float
    bound: str
    action: int


class SearchExpired(Exception):
    """Unwind the incomplete iteration without replacing its completed result."""


class AlphaBeta:
    def __init__(self, evaluator: Evaluator, config: EngineConfig | None = None):
        self.evaluator = evaluator
        self.config = config or EngineConfig()
        self.table: OrderedDict[tuple, Entry] = OrderedDict()
        self.depth = self.nodes = self.cutoffs = self.table_hits = 0
        self.deadline = 0.0

    def clear_tree(self):
        """Call when replacing weights: bounds depend on the evaluator."""
        self.table.clear()

    def _check_time(self):
        if monotonic() >= self.deadline:
            raise SearchExpired

    def _search(self, state: State, depth: int, alpha: float, beta: float) -> tuple[float, int]:
        self._check_time()
        self.nodes += 1
        if state.terminal:
            return state.outcome(state.player), -1
        if depth == 0:
            _, value = self.evaluator(state)
            self._check_time()
            return float(value), -1
        key = CachedEvaluator._key(state)
        entry = self.table.get(key)
        if entry is not None:
            self.table.move_to_end(key)
            if entry.depth >= depth:
                self.table_hits += 1
                if entry.bound == "exact":
                    return entry.value, entry.action
                if entry.bound == "lower":
                    alpha = max(alpha, entry.value)
                else:
                    beta = min(beta, entry.value)
                if alpha >= beta:
                    return entry.value, entry.action
        # Classify against the effective window, including cached bounds.
        original_alpha, original_beta = alpha, beta
        policy, _ = self.evaluator(state)
        self._check_time()
        actions = sorted(
            state.legal_actions,
            key=lambda action: (entry is not None and action == entry.action, policy[action]),
            reverse=True,
        )
        best, best_action = -float("inf"), actions[0]
        for action in actions:
            child = state.play(action)
            if child.player == state.player:
                value, _ = self._search(child, depth - 1, alpha, beta)
            else:
                value, _ = self._search(child, depth - 1, -beta, -alpha)
                value = -value
            if value > best:
                best, best_action = value, action
            alpha = max(alpha, best)
            if alpha >= beta:
                self.cutoffs += 1
                break
        bound = "upper" if best <= original_alpha else "lower" if best >= original_beta else "exact"
        if self.config.table_size:
            self.table[key] = Entry(depth, best, bound, best_action)
            self.table.move_to_end(key)
            if len(self.table) > self.config.table_size:
                self.table.popitem(last=False)
        return best, best_action

    def policy(self, state: State) -> tuple[np.ndarray, float]:
        if state.terminal:
            raise ValueError("Cannot search a finished game.")
        self.depth = self.nodes = self.cutoffs = self.table_hits = 0
        self.deadline = monotonic() + self.config.time_limit
        result = np.zeros(state.board.num_edges, dtype=np.float32)
        if self.config.exact_threshold and len(state.legal_actions) <= self.config.exact_threshold:
            margin, actions = solve(state)
            result[list(actions)] = 1 / len(actions)
            self.depth = len(state.legal_actions)
            return result, float(np.sign(margin))
        # Always obtain a legal fallback. A single network evaluation (or the
        # small root endgame solver above) can overrun this soft time budget.
        prior, value = self.evaluator(state)
        action = max(state.legal_actions, key=lambda move: prior[move])
        for depth in range(1, min(self.config.max_depth, len(state.legal_actions)) + 1):
            try:
                next_value, next_action = self._search(state, depth, -float("inf"), float("inf"))
            except SearchExpired:
                break
            value, action, self.depth = next_value, next_action, depth
        result[action] = 1
        return result, float(value)

    def action(self, state: State) -> int:
        return int(np.argmax(self.policy(state)[0]))
