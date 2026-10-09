"""An independent chain-control benchmark, with exact small endgames.

This is a heuristic opponent, not a claim of perfect play. In a reduced endgame
it compares taking a component against handing back two chain boxes or four loop
boxes. Outside that structure it uses immediate tactics. It never trains the GNN.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from alphaboxes.game import State
from alphaboxes.opponents import solve, tactical_action


@dataclass(frozen=True)
class Component:
    boxes: frozenset[int]
    boundaries: tuple[int, ...]
    active: bool

    @property
    def handout(self) -> int:
        return 2 if self.boundaries else 4


def components(state: State) -> tuple[Component, ...] | None:
    """Recognize unclaimed-box paths/cycles once every box has ≤2 open sides."""
    remaining = {
        b: tuple(e for e in state.board.boxes[b] if state.edges[e] < 0)
        for b, owner in enumerate(state.owners)
        if owner < 0
    }
    if any(len(edges) > 2 for edges in remaining.values()):
        return None
    unseen = set(remaining)
    result = []
    while unseen:
        pending = [min(unseen)]
        members, boundaries = set(), set()
        while pending:
            b = pending.pop()
            if b in members:
                continue
            members.add(b)
            unseen.discard(b)
            for edge in remaining[b]:
                neighbors = [c for c in state.board.edge_boxes[edge] if c != b]
                if not neighbors:
                    boundaries.add(edge)
                pending.extend(c for c in neighbors if c in unseen)
        result.append(
            Component(
                frozenset(members),
                tuple(sorted(boundaries)),
                any(len(remaining[b]) == 1 for b in members),
            )
        )
    return tuple(result)


@lru_cache(maxsize=32_768)
def _opening_margin(unopened: tuple[tuple[int, int], ...]) -> int:
    """Reduced component-game margin for the player who must open a component.

    This abstracts normal take-all/keep-control play. Short chains, premature
    sacrifices, and unusual interleavings can violate its assumptions.
    """
    if not unopened:
        return 0
    values = []
    for index, (length, handout) in enumerate(unopened):
        rest = unopened[:index] + unopened[index + 1 :]
        future = _opening_margin(rest)
        take_all = -length - future
        keep_control = -length + 2 * handout + future
        values.append(min(take_all, keep_control) if length >= handout else take_all)
    return max(values)


def chain_action(state: State, rng: np.random.Generator, *, exact_threshold: int = 12) -> int:
    if exact_threshold and len(state.legal_actions) <= exact_threshold:
        return int(rng.choice(solve(state)[1]))
    groups = components(state)
    if not groups:
        return tactical_action(state, rng)
    active = [group for group in groups if group.active]
    if len(active) > 1:
        return tactical_action(state, rng)
    if active:
        current = active[0]
        rest = tuple(
            sorted((len(group.boxes), group.handout) for group in groups if group is not current)
        )
        future = _opening_margin(rest)
        length, handout = len(current.boxes), current.handout
        take_all = length + future
        keep_control = length - 2 * handout - future
        if rest and length >= handout and keep_control >= take_all:
            if length == handout:
                # A double-cross leaves pairs of boxes for the other player.
                give_back = []
                for action in state.legal_actions:
                    touched = state.board.edge_boxes[action]
                    if not set(touched).issubset(current.boxes) or state.captures(action):
                        continue
                    if all(
                        sum(state.edges[e] < 0 and e != action for e in state.board.boxes[b]) == 1
                        for b in current.boxes
                    ):
                        give_back.append(action)
                if give_back:
                    return int(rng.choice(give_back))
        captures = [
            action
            for action in state.legal_actions
            if state.captures(action) and set(state.captures(action)).issubset(current.boxes)
        ]
        return int(rng.choice(captures))
    candidates = []
    for index, group in enumerate(groups):
        rest = tuple(
            sorted(
                (len(other.boxes), other.handout) for j, other in enumerate(groups) if j != index
            )
        )
        future = _opening_margin(rest)
        length, handout = len(group.boxes), group.handout
        value = -length - future
        if length >= handout:
            value = min(value, -length + 2 * handout + future)
        actions = group.boundaries or tuple(
            action
            for action in state.legal_actions
            if set(state.board.edge_boxes[action]).issubset(group.boxes)
        )
        candidates.extend((value, action) for action in actions)
    best = max(value for value, _ in candidates)
    return int(rng.choice([action for value, action in candidates if value == best]))
