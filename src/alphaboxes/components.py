"""Components of unclaimed boxes connected through undrawn shared edges."""

from dataclasses import dataclass

import numpy as np

from alphaboxes.game import State

NODE_FEATURES = 4
CONTEXT_FEATURES = 14


@dataclass(frozen=True, slots=True)
class Component:
    boxes: tuple[int, ...]
    boundaries: int
    capture_fronts: int
    min_free: int
    max_free: int

    @property
    def closed_chain(self) -> bool:
        return self.min_free == self.max_free == 2 and self.boundaries == 2

    @property
    def closed_loop(self) -> bool:
        return self.min_free == self.max_free == 2 and self.boundaries == 0


def free_components(state: State) -> tuple[Component, ...]:
    """Scan each unclaimed box once; component labels never enter the network."""
    unseen = {box for box, owner in enumerate(state.owners) if owner < 0}
    result = []
    while unseen:
        pending = [unseen.pop()]
        members = []
        boundaries = fronts = maximum = 0
        minimum = 4
        while pending:
            box = pending.pop()
            members.append(box)
            remaining = 0
            for edge in state.board.boxes[box]:
                if state.edges[edge] >= 0:
                    continue
                remaining += 1
                neighbors = state.board.edge_boxes[edge]
                if len(neighbors) == 1:
                    boundaries += 1
                else:
                    other = neighbors[0] if neighbors[1] == box else neighbors[1]
                    if other in unseen:
                        unseen.remove(other)
                        pending.append(other)
            fronts += remaining == 1
            minimum = min(minimum, remaining)
            maximum = max(maximum, remaining)
        result.append(Component(tuple(members), boundaries, fronts, minimum, maximum))
    return tuple(result)


def structural_features(state: State) -> tuple[np.ndarray, np.ndarray]:
    """Return box features and global counts, in fixed box units across sizes.

    Sizes and counts use four-box units, matching the largest standard handout.
    Closed-component counts are descriptive on general positions: branching
    components are explicitly counted and are never labeled as chains or loops.
    """
    nodes = np.zeros((state.board.num_boxes, NODE_FEATURES), dtype=np.float32)
    chains, loops, short, active, complex_count = [], [], 0, 0, 0
    for component in free_components(state):
        size = len(component.boxes)
        nodes[list(component.boxes)] = (
            size / 4,
            float(component.closed_chain),
            float(component.closed_loop),
            min(component.capture_fronts, 2) / 2,
        )
        if component.closed_chain:
            if size >= 3:
                chains.append(size)
            else:
                short += 1
        if component.closed_loop:
            loops.append(size)
        active += component.capture_fronts > 0
        complex_count += component.max_free > 2 or component.min_free == 0
    ours, theirs = state.scores[state.player], state.scores[1 - state.player]
    context = np.array(
        [
            sum(chains) / 4,
            len(chains) / 4,
            chains.count(3) / 4,
            sum(loops) / 4,
            len(loops) / 4,
            loops.count(4) / 4,
            loops.count(6) / 4,
            short / 4,
            active / 4,
            complex_count / 4,
            len(chains) % 2,
            len(loops) % 2,
            (ours - theirs) / 4,
            state.owners.count(-1) / 4,
        ],
        dtype=np.float32,
    )
    return nodes, context
