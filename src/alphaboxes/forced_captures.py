"""Evaluate mandatory captures at the next genuine decision position."""

import numpy as np

from alphaboxes.chain_moves import allowed_actions
from alphaboxes.game import State
from alphaboxes.search import CachedEvaluator


class ForcedCaptureEvaluator(CachedEvaluator):
    """Retain neural choices, extending values through single-choice captures.

    The conservative chain reduction determines mandatory moves. Extension
    stops before every capture-versus-handout decision; no random or heuristic
    continuation is used. Enable the same reduction in the enclosing search.
    """

    def __init__(self, evaluator, cache_size: int = 4096):
        super().__init__(cache_size)
        self.evaluator = evaluator
        self.positions = self.moves = self.terminal_finishes = 0

    def clear_cache(self):
        super().clear_cache()
        self.evaluator.clear_cache()

    def statistics(self):
        return {
            "forced_capture_positions": self.positions,
            "forced_capture_moves": self.moves,
            "forced_capture_terminal_finishes": self.terminal_finishes,
        }

    def _predict(self, state: State) -> tuple[np.ndarray, float]:
        choices = allowed_actions(state)
        if len(choices) != 1:
            return self.evaluator(state)
        first = choices[0]
        continuation = state
        steps = 0
        while not continuation.terminal and len(choices) == 1:
            action = choices[0]
            if not continuation.captures(action):
                break
            continuation = continuation.play(action)
            steps += 1
            if not continuation.terminal:
                choices = allowed_actions(continuation)
        if not steps:
            return self.evaluator(state)
        self.positions += 1
        self.moves += steps
        policy = np.zeros(state.board.num_edges, np.float32)
        policy[first] = 1
        if continuation.terminal:
            self.terminal_finishes += 1
            value = continuation.outcome(state.player)
        else:
            _, value = self.evaluator(continuation)
            if continuation.player != state.player:
                value = -value
        return policy, float(value)
