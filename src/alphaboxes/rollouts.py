"""Ground graph-network values with seeded legal playouts and exact long endgames."""

import hashlib
from functools import lru_cache

import numpy as np

from alphaboxes.endgame import DegreeTwoSolver
from alphaboxes.game import State, board
from alphaboxes.search import CachedEvaluator


@lru_cache(maxsize=128)
def geometry(rows: int, cols: int):
    layout = board(rows, cols)
    boxes = np.asarray(layout.boxes, dtype=np.intp)
    neighbors = np.full((layout.num_edges, 2), -1, dtype=np.intp)
    for edge, incident in enumerate(layout.edge_boxes):
        neighbors[edge, : len(incident)] = incident
    valid = neighbors >= 0
    indices = np.maximum(neighbors, 0)
    for array in (boxes, neighbors, valid, indices):
        array.flags.writeable = False
    return boxes, neighbors, valid, indices


def playout(state: State, rng, oracle=None, on_move=None):
    """Return WDL from the initial player's view, steps and exact-finish status.

    Mutable arrays are private to this episode. Captures keep the player active;
    replay tests compare every transition with the immutable rules engine.
    """
    boxes, neighbors, valid, indices = geometry(state.board.rows, state.board.cols)
    edges = np.asarray(state.edges, dtype=np.int8).copy()
    owners = np.asarray(state.owners, dtype=np.int8).copy()
    remaining = (edges[boxes] < 0).sum(axis=1).astype(np.int8)
    player, steps = state.player, 0
    while np.any(edges < 0):
        if oracle is not None and remaining.max() <= 2:
            position = State(state.board, tuple(map(int, edges)), tuple(map(int, owners)), player)
            value = oracle.fast_value(position)
            if value is not None:
                outcome = float(np.sign(value))
                return outcome if player == state.player else -outcome, steps, True
        legal = edges < 0
        incident = remaining[indices]
        captures = ((incident == 1) & valid).sum(axis=1)
        maximum = captures[legal].max()
        if maximum:
            choices = legal & (captures == maximum)
        else:
            safe = legal & np.all((incident > 2) | ~valid, axis=1)
            choices = safe if safe.any() else legal
        action = int(rng.choice(np.flatnonzero(choices)))
        adjacent = neighbors[action, valid[action]]
        edges[action] = player
        remaining[adjacent] -= 1
        captured = adjacent[remaining[adjacent] == 0]
        owners[captured] = player
        if on_move is not None:
            on_move(action)
        if not len(captured):
            player = 1 - player
        steps += 1
    margin = np.count_nonzero(owners == state.player) - np.count_nonzero(owners == 1 - state.player)
    return float(np.sign(margin)), steps, False


class RolloutEvaluator(CachedEvaluator):
    """Keep learned priors and blend their value with complete-game estimates."""

    def __init__(self, neural, rollouts=2, weight=1.0, seed=1729, cache_size=4096):
        super().__init__(cache_size)
        if not isinstance(rollouts, int) or rollouts < 1:
            raise ValueError("Use a positive integer rollout count.")
        if not np.isfinite(weight) or not 0 <= weight <= 1 or seed < 0:
            raise ValueError("Use a weight in [0, 1] and a nonnegative seed.")
        self.neural, self.rollouts, self.weight, self.seed = neural, rollouts, weight, seed
        self.oracle = DegreeTwoSolver(cache_size=10000)
        self.episodes = self.moves = self.oracle_finishes = 0

    def clear_cache(self):
        super().clear_cache()
        self.neural.clear_cache()

    def statistics(self):
        return {
            "rollout_episodes": self.episodes,
            "rollout_moves": self.moves,
            "rollout_oracle_finishes": self.oracle_finishes,
        }

    def _predict(self, state):
        policy, value = self.neural(state)
        digest = hashlib.blake2b(repr(self._key(state)).encode(), digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little") ^ self.seed)
        outcomes = []
        for _ in range(self.rollouts):
            outcome, steps, exact = playout(state, rng, self.oracle)
            outcomes.append(outcome)
            self.episodes += 1
            self.moves += steps
            self.oracle_finishes += exact
        return policy, (1 - self.weight) * value + self.weight * float(np.mean(outcomes))
