"""Conservative capture-chain move reductions, leaving all other states intact."""

from alphaboxes.game import State


def allowed_actions(state: State) -> tuple[int, ...]:
    """Reduce one opened chain to captures or its final control handout.

    Multiple capture regions retain the ordinary legal set. A junction is an
    outlet of a degree-two path, rather than part of that path.
    """
    legal = state.legal_actions
    degree = [sum(state.edges[e] < 0 for e in box) for box in state.board.boxes]
    unseen = {b for b, owner in enumerate(state.owners) if owner < 0 and 1 <= degree[b] <= 2}
    active = []
    while unseen:
        pending, members, outlets = [unseen.pop()], set(), set()
        fronts = 0
        while pending:
            box = pending.pop()
            members.add(box)
            fronts += degree[box] == 1
            for edge in state.board.boxes[box]:
                if state.edges[edge] >= 0:
                    continue
                adjacent = state.board.edge_boxes[edge]
                if len(adjacent) == 1:
                    outlets.add(edge)
                    continue
                other = adjacent[0] if adjacent[1] == box else adjacent[1]
                if degree[other] >= 3:
                    outlets.add(edge)
                elif other in unseen:
                    unseen.remove(other)
                    pending.append(other)
        if fronts:
            active.append((members, fronts, outlets))
    if len(active) != 1:
        return legal
    members, fronts, outlets = active[0]
    captures = [a for a in legal if any(b in members for b in state.captures(a))]
    if not captures:
        return legal
    capture = captures[0]
    if fronts == 1 and len(outlets) == 1:
        if len(members) == 2:
            return tuple(sorted({capture, next(iter(outlets))}))
        return (capture,)
    if fronts == 2 and not outlets:
        if len(members) == 4:
            handouts = [
                a
                for a in legal
                if len(state.board.edge_boxes[a]) == 2
                and all(b in members for b in state.board.edge_boxes[a])
                and not state.captures(a)
            ]
            if len(handouts) == 1:
                return tuple(sorted({capture, handouts[0]}))
            return legal
        return (capture,)
    return legal
