"""Bounded archives of reachable self-play positions for optional midgame starts."""

from collections import defaultdict, deque
from collections.abc import Iterable

import numpy as np

from alphaboxes.game import State


def sample_endgame(size: tuple[int, int], rng: np.random.Generator) -> State:
    """Reach a varied collection of row chains and 4/6-box loops by legal play.

    Some whole components are completed in a shuffled prefix, producing varied
    current scores. Every unfinished component keeps two free sides per box.
    """
    state = State.new(*size)
    rows, cols = size
    groups = []
    row = 0
    while row < rows:
        if cols < 2 or row + 1 == rows or rng.random() < 0.35:
            groups.append(
                {state.board.boxes[row * cols + c][2] for c in range(cols)}
                | {state.board.boxes[row * cols + cols - 1][3]}
            )
            row += 1
        else:
            col = 0
            while col < cols:
                remaining = cols - col
                choices = [
                    width for width in (2, 3) if remaining - width >= 0 and remaining - width != 1
                ]
                width = int(rng.choice(choices))
                free = {
                    state.board.boxes[(row + r) * cols + c][3]
                    for r in (0, 1)
                    for c in range(col, col + width - 1)
                }
                free.update(
                    (
                        state.board.boxes[row * cols + col][1],
                        state.board.boxes[row * cols + col + width - 1][1],
                    )
                )
                groups.append(free)
                col += width
            row += 2
    completed = rng.random(len(groups)) < rng.uniform(0, 0.65)
    if completed.all():
        completed[int(rng.integers(len(groups)))] = False
    free = set().union(*(group for group, done in zip(groups, completed, strict=True) if not done))
    prefix = [edge for edge in range(state.board.num_edges) if edge not in free]
    for action in rng.permutation(prefix):
        state = state.play(int(action))
    return state


class StartPositionArchive:
    def __init__(
        self,
        fraction: float = 0,
        min_moves: int = 12,
        max_moves: int = 28,
        capacity: int = 4096,
        endgame_fraction: float = 0,
    ):
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("start_fraction must be finite and between zero and one.")
        if not np.isfinite(endgame_fraction) or not 0 <= endgame_fraction <= 1:
            raise ValueError("endgame_fraction must be finite and between zero and one.")
        if (
            any(not isinstance(value, int) for value in (min_moves, max_moves, capacity))
            or min_moves < 1
            or max_moves < min_moves
            or capacity < 1
        ):
            raise ValueError("Start moves and archive capacity must be positive and ordered.")
        self.fraction = fraction
        self.endgame_fraction = endgame_fraction
        self.min_moves, self.max_moves = min_moves, max_moves
        self.positions: dict[tuple[int, int], deque[State]] = defaultdict(
            lambda: deque(maxlen=capacity)
        )
        self.reset_statistics()

    def reset_statistics(self) -> None:
        self.games = self.midgames = self.endgames = self.start_moves = 0

    def observe(self, states: Iterable[State]) -> None:
        if not self.fraction:
            return
        for state in states:
            moves = state.board.num_edges - len(state.legal_actions)
            if not state.terminal and self.min_moves <= moves <= self.max_moves:
                self.positions[state.board.rows, state.board.cols].append(state)

    def choose(self, size: tuple[int, int], rng: np.random.Generator) -> State:
        self.games += 1
        if self.endgame_fraction and rng.random() < self.endgame_fraction:
            state = sample_endgame(size, rng)
            self.endgames += 1
            self.start_moves += state.board.num_edges - len(state.legal_actions)
            return state
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
            "endgame_starts": self.endgames,
            "initial_moves_total": self.start_moves,
        }
