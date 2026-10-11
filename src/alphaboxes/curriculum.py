"""Bounded archives of reachable self-play positions for optional midgame starts."""

from collections import defaultdict, deque
from collections.abc import Iterable

import numpy as np

from alphaboxes.game import State


class StartPositionArchive:
    def __init__(
        self, fraction: float = 0, min_moves: int = 12, max_moves: int = 28, capacity: int = 4096
    ):
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("start_fraction must be finite and between zero and one.")
        if (
            any(not isinstance(value, int) for value in (min_moves, max_moves, capacity))
            or min_moves < 1
            or max_moves < min_moves
            or capacity < 1
        ):
            raise ValueError("Start moves and archive capacity must be positive and ordered.")
        self.fraction = fraction
        self.min_moves, self.max_moves = min_moves, max_moves
        self.positions: dict[tuple[int, int], deque[State]] = defaultdict(
            lambda: deque(maxlen=capacity)
        )
        self.reset_statistics()

    def reset_statistics(self) -> None:
        self.games = self.midgames = self.start_moves = 0

    def observe(self, states: Iterable[State]) -> None:
        if not self.fraction:
            return
        for state in states:
            moves = state.board.num_edges - len(state.legal_actions)
            if not state.terminal and self.min_moves <= moves <= self.max_moves:
                self.positions[state.board.rows, state.board.cols].append(state)

    def choose(self, size: tuple[int, int], rng: np.random.Generator) -> State:
        self.games += 1
        state = State.new(*size)
        # Preserve the empty-start RNG stream when this option is disabled.
        if not self.fraction or rng.random() >= self.fraction:
            return state
        archived = self.positions.get(size)
        if archived:
            state = archived[int(rng.integers(len(archived)))]
        else:
            upper = min(self.max_moves, state.board.num_edges - 1)
            if upper < self.min_moves:
                return state
            for _ in range(int(rng.integers(self.min_moves, upper + 1))):
                state = state.play(int(rng.choice(state.legal_actions)))
        self.midgames += 1
        self.start_moves += state.board.num_edges - len(state.legal_actions)
        return state

    def statistics(self) -> dict[str, int]:
        return {
            "sampled_games": self.games,
            "midgame_starts": self.midgames,
            "initial_moves_total": self.start_moves,
        }
