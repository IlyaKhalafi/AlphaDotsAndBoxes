"""Exact degree-two endgames, reduced to multisets of paths and cycles.

Every legal cut is represented, including noncapturing sacrifices. Unlike a
take-all/handout heuristic this also handles short chains and multiple open
chains. Other positions are deliberately left to the neural search.
"""

from collections import OrderedDict
from time import monotonic

from alphaboxes.game import State

Position = tuple[tuple[int, int], ...]


class EndgameTimeout(TimeoutError):
    """No exact result was completed within the caller's deadline."""


def position(state: State) -> Position | None:
    """Return (box count, boundary count), with -1 denoting a closed cycle."""
    free = {
        box: tuple(edge for edge in state.board.boxes[box] if state.edges[edge] < 0)
        for box, owner in enumerate(state.owners)
        if owner < 0
    }
    if any(not edges or len(edges) > 2 for edges in free.values()):
        return None
    unseen = set(free)
    groups = []
    while unseen:
        pending = [unseen.pop()]
        members, boundaries, active = set(), 0, False
        while pending:
            box = pending.pop()
            members.add(box)
            active |= len(free[box]) == 1
            for edge in free[box]:
                neighbors = state.board.edge_boxes[edge]
                if len(neighbors) == 1:
                    boundaries += 1
                for other in neighbors:
                    if other in unseen:
                        unseen.remove(other)
                        pending.append(other)
        groups.append((len(members), boundaries if active or boundaries else -1))
    return tuple(sorted(groups))


def cuts(group: tuple[int, int]):
    """Yield (captured boxes, surviving components) for distinct edge cuts."""
    length, boundaries = group
    if boundaries == -1:
        yield 0, ((length, 0),)
        return
    if boundaries:
        yield (1, ()) if length == 1 and boundaries == 1 else (
            0, ((length, boundaries - 1),)
        )
    for left in range(1, length):
        pieces, captured = [], 0
        for size, boundary in ((left, int(boundaries > 0)),
                               (length - left, int(boundaries == 2))):
            if size == 1 and not boundary:
                captured += 1
            else:
                pieces.append((size, boundary))
        yield captured, tuple(pieces)


class DegreeTwoSolver:
    """Memoized exact margin with cooperative cancellation and bounded memory."""

    def __init__(self, cache_size: int = 100_000):
        self.cache_size = cache_size
        self.cache: OrderedDict[Position, int] = OrderedDict()
        self.nodes = 0

    def _future(self, groups: Position, deadline: float | None) -> int:
        if deadline is not None and monotonic() >= deadline:
            raise EndgameTimeout
        if not groups:
            return 0
        if groups in self.cache:
            self.cache.move_to_end(groups)
            return self.cache[groups]
        self.nodes += 1
        best = -sum(length for length, _ in groups)
        previous = None
        successors = set()
        for index, group in enumerate(groups):
            if group == previous:
                continue
            previous = group
            rest = groups[:index] + groups[index + 1 :]
            for captured, pieces in cuts(group):
                successors.add((captured, tuple(sorted(rest + pieces))))
        # Captures first often reach a tight bound before noncapturing cuts.
        for captured, child in sorted(successors, reverse=True):
            continuation = self._future(child, deadline)
            value = captured + continuation if captured else -continuation
            best = max(best, value)
            if best == sum(length for length, _ in groups):
                break
        if self.cache_size > 0:
            self.cache[groups] = best
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return best

    def value(self, state: State, deadline: float | None = None) -> float | None:
        groups = position(state)
        if groups is None:
            return None
        future = self._future(groups, deadline)
        ours, theirs = state.scores[state.player], state.scores[1 - state.player]
        return (ours - theirs + future) / state.board.num_boxes

    def solve(self, state: State, deadline: float | None = None):
        """Return exact margin and all optimal actions, or None outside scope."""
        value = self.value(state, deadline)
        if value is None:
            return None
        if state.terminal:
            return value, ()
        actions = []
        for action in state.legal_actions:
            child = state.play(action)
            child_value = self.value(child, deadline)
            if child.player != state.player:
                child_value = -child_value
            if abs(child_value - value) < 1e-10:
                actions.append(action)
        return value, tuple(actions)
