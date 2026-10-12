"""Exact degree-two endgames, reduced to multisets of paths and cycles.

Every legal cut is represented, including noncapturing sacrifices. Unlike a
take-all/handout heuristic this also handles short chains and multiple open
chains. Other positions are deliberately left to the neural search.
"""

from collections import OrderedDict
from time import monotonic

from alphaboxes.components import free_components
from alphaboxes.game import State

Position = tuple[tuple[int, int], ...]


class EndgameTimeout(TimeoutError):
    """No exact result was completed within the caller's deadline."""


def position(state: State) -> Position | None:
    """Return (box count, boundary count), with -1 denoting a closed cycle."""
    if any(
        owner < 0 and not 1 <= sum(state.edges[e] < 0 for e in box) <= 2
        for owner, box in zip(state.owners, state.board.boxes, strict=True)
    ):
        return None
    components = free_components(state)
    if any(c.min_free == 0 or c.max_free > 2 for c in components):
        return None
    return tuple(
        sorted(
            (len(c.boxes), c.boundaries if c.capture_fronts or c.boundaries else -1)
            for c in components
        )
    )


def cuts(group: tuple[int, int]):
    """Yield (captured boxes, surviving components) for distinct edge cuts."""
    length, boundaries = group
    if boundaries == -1:
        yield 0, ((length, 0),)
        return
    if boundaries:
        yield (1, ()) if length == 1 and boundaries == 1 else (0, ((length, boundaries - 1),))
    for left in range(1, length):
        pieces, captured = [], 0
        for size, boundary in ((left, int(boundaries > 0)), (length - left, int(boundaries == 2))):
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

    def _loony_future(self, groups: Position, deadline: float | None) -> int | None:
        """Collapse complete long chains/loops into take-all or handout turns.

        Closed chains need at least three boxes; closed loops need an even
        length of at least four. One partially captured component is supported.
        Short closed chains and multiple capture fronts use the general solver.
        See Buzzard and Ciere, Playing Simple Loony Dots-and-Boxes Endgames
        Optimally (2014), for the two-box/four-box control alternatives.
        """
        active = []
        closed = []
        for group in groups:
            length, boundaries = group
            if boundaries == 2 and length >= 3:
                closed.append(group)
            elif boundaries == -1 and length >= 4 and length % 2 == 0:
                closed.append(group)
            elif boundaries in (0, 1) and length >= 2 - boundaries:
                active.append(group)
            else:
                return None
        if len(active) > 1:
            return None
        if active:
            length, boundaries = active[0]
            continuation = self._future(tuple(closed), deadline)
            best = length + continuation
            handout = 2 if boundaries == 1 else 4
            if length >= handout:
                best = max(best, length - 2 * handout - continuation)
            return best
        best = -sum(length for length, _ in groups)
        previous = None
        for index, group in enumerate(groups):
            if group == previous:
                continue
            previous = group
            length, boundaries = group
            continuation = self._future(groups[:index] + groups[index + 1 :], deadline)
            handout = 2 if boundaries == 2 else 4
            best = max(best, min(-length - continuation, -length + 2 * handout + continuation))
        return best

    def _future(self, groups: Position, deadline: float | None) -> int:
        if deadline is not None and monotonic() >= deadline:
            raise EndgameTimeout
        if not groups:
            return 0
        if groups in self.cache:
            self.cache.move_to_end(groups)
            return self.cache[groups]
        self.nodes += 1
        forced = sum(
            length for length, boundary in groups if (length, boundary) in {(1, 1), (2, 0)}
        )
        if forced:
            rest = tuple(group for group in groups if group not in {(1, 1), (2, 0)})
            best = forced + self._future(rest, deadline)
        else:
            best = self._loony_future(groups, deadline)
        if best is None:
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

    def fast_value(self, state: State) -> float | None:
        """Use only the long-component recurrence and forced tiny captures.

        This avoids the general multi-front solver without a clock-dependent
        cutoff, so seeded rollout estimates do not depend on scheduling.
        """
        groups = position(state)
        if groups is None:
            return None
        future = self.fast_future(groups)
        if future is None:
            return None
        ours, theirs = state.scores[state.player], state.scores[1 - state.player]
        return (ours - theirs + future) / state.board.num_boxes

    def fast_future(self, groups: Position) -> int | None:
        """Exact future margin on the restricted rollout domain."""
        active = 0
        for length, boundaries in groups:
            if (length, boundaries) in {(1, 1), (2, 0)}:
                continue
            if boundaries == 2 and length >= 3:
                continue
            if boundaries == -1 and length >= 4 and length % 2 == 0:
                continue
            if boundaries in (0, 1) and length >= 2 - boundaries:
                active += 1
            else:
                return None
        if active > 1:
            return None
        return self._future(groups, None)

    def solve(self, state: State, deadline: float | None = None):
        """Return exact margin and all optimal actions, or None outside scope."""
        value = self.value(state, deadline)
        if value is None:
            return None
        if state.terminal:
            return value, ()
        components = free_components(state)
        if all((c.closed_chain and len(c.boxes) >= 3) or c.closed_loop for c in components):
            # Any opening in a complete long component offers the same take-all
            # or handout alternatives. Avoid enumerating two-front child states.
            opening_values = self._opening_values(components, deadline)
            best = max(opening_values.values())
            by_box = {
                box: (len(c.boxes), 2 if c.closed_chain else -1)
                for c in components
                for box in c.boxes
            }
            actions = tuple(
                a
                for a in state.legal_actions
                if opening_values[by_box[state.board.edge_boxes[a][0]]] == best
            )
            return value, actions
        actions = []
        for action in state.legal_actions:
            child = state.play(action)
            child_value = self.value(child, deadline)
            if child.player != state.player:
                child_value = -child_value
            if abs(child_value - value) < 1e-10:
                actions.append(action)
        return value, tuple(actions)

    def _opening_values(self, components, deadline):
        groups = tuple(sorted((len(c.boxes), 2 if c.closed_chain else -1) for c in components))
        values = {}
        for index, group in enumerate(groups):
            if group in values:
                continue
            length, boundary = group
            continuation = self._future(groups[:index] + groups[index + 1 :], deadline)
            handout = 2 if boundary == 2 else 4
            values[group] = min(-length - continuation, -length + 2 * handout + continuation)
        return values

    def action_value(self, state: State, action: int, deadline: float | None = None):
        """Exact normalized margin after an action, from the original player's view.

        A closed-chain interior cut creates two fronts. Its opening value is
        available directly from the control recurrence, without expanding those
        fronts or opening further components during a general minimax solve.
        """
        if not isinstance(action, int) or action not in state.legal_actions:
            raise ValueError("Action must be a legal edge.")
        components = free_components(state)
        if all((c.closed_chain and len(c.boxes) >= 3) or c.closed_loop for c in components):
            box = state.board.edge_boxes[action][0]
            component = next(c for c in components if box in c.boxes)
            group = len(component.boxes), 2 if component.closed_chain else -1
            future = self._opening_values(components, deadline)[group]
            ours, theirs = state.scores[state.player], state.scores[1 - state.player]
            return (ours - theirs + future) / state.board.num_boxes
        child = state.play(action)
        value = self.value(child, deadline)
        return value if value is None or child.player == state.player else -value


# Keys describe game structure, independent of board size, score and weights.
shared_solver = DegreeTwoSolver()
