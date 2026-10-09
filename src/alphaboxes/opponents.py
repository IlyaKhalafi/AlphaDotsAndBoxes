"""Reproducible baselines and an exact late-game oracle."""

from functools import cache

import numpy as np

from alphaboxes.game import State


def random_action(state: State, rng: np.random.Generator) -> int:
    return int(rng.choice(state.legal_actions))


def tactical_action(state: State, rng: np.random.Generator) -> int:
    """Capture most boxes, otherwise avoid offering a third side when possible."""
    legal = state.legal_actions
    captures = [len(state.captures(a)) for a in legal]
    if max(captures) > 0:
        return int(
            rng.choice(
                [a for a, count in zip(legal, captures, strict=True) if count == max(captures)]
            )
        )
    safe = [
        a
        for a in legal
        if all(
            sum(state.edges[e] < 0 for e in state.board.boxes[b]) != 2
            for b in state.board.edge_boxes[a]
        )
    ]
    return int(rng.choice(safe or legal))


def solve(state: State) -> tuple[float, tuple[int, ...]]:
    """Exact minimax score margin; switch signs only when the player changes.

    Cache ignores historical edge ownership: only occupancy and the current score
    affect the remaining game. Intended for small boards / few remaining edges.
    """
    geometry = state.board
    full = (1 << geometry.num_edges) - 1
    box_masks = tuple(sum(1 << e for e in edges) for edges in geometry.boxes)

    @cache
    def future(occupied: int) -> int:
        if occupied == full:
            return 0
        best = -geometry.num_boxes - 1
        for a in range(geometry.num_edges):
            bit = 1 << a
            if occupied & bit:
                continue
            next_occupied = occupied | bit
            captured = sum(
                (next_occupied & box_masks[b]) == box_masks[b] for b in geometry.edge_boxes[a]
            )
            continuation = future(next_occupied)
            best = max(best, captured + continuation if captured else -continuation)
        return best

    occupied = sum(1 << e for e, owner in enumerate(state.edges) if owner >= 0)
    scored = state.scores[state.player] - state.scores[1 - state.player]
    action_scores = {}
    for action in state.legal_actions:
        captured = len(state.captures(action))
        continuation = future(occupied | (1 << action))
        action_scores[action] = captured + continuation if captured else -continuation
    if not action_scores:
        return scored / geometry.num_boxes, ()
    best = max(action_scores.values())
    return (scored + best) / geometry.num_boxes, tuple(
        a for a, value in action_scores.items() if value == best
    )
