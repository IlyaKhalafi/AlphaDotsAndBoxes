"""Typed legal playout buffers; the graph network still runs through NumPy."""

import numpy as np


cdef object groups_for(
    const Py_ssize_t[:, :] boxes, const Py_ssize_t[:, :] neighbors,
    signed char[:] edges, signed char[:] owners, int[:] remaining,
    unsigned char[:] unseen, Py_ssize_t[:] pending,
):
    cdef Py_ssize_t first, box, edge, other, top, size, index
    cdef int boundaries, fronts
    groups = []
    for box in range(owners.shape[0]):
        unseen[box] = owners[box] < 0
        if unseen[box] and remaining[box] not in (1, 2):
            return None
    for first in range(owners.shape[0]):
        if not unseen[first]:
            continue
        unseen[first] = 0
        pending[0] = first
        top = 1
        size = boundaries = fronts = 0
        while top:
            top -= 1
            box = pending[top]
            size += 1
            fronts += remaining[box] == 1
            for index in range(4):
                edge = boxes[box, index]
                if edges[edge] >= 0:
                    continue
                if neighbors[edge, 1] < 0:
                    boundaries += 1
                else:
                    other = neighbors[edge, 0] if neighbors[edge, 1] == box else neighbors[edge, 1]
                    if unseen[other]:
                        unseen[other] = 0
                        pending[top] = other
                        top += 1
        if boundaries + fronts not in (0, 2):
            return None
        groups.append((size, -1 if not boundaries and not fronts else boundaries))
    return tuple(sorted(groups))


def playout(state, rng, oracle=None, on_move=None):
    from alphaboxes.rollouts import geometry

    layout = geometry(state.board.rows, state.board.cols)
    cdef const Py_ssize_t[:, :] boxes = layout[0]
    cdef const Py_ssize_t[:, :] neighbors = layout[1]
    cdef Py_ssize_t n_edges = state.board.num_edges, n_boxes = state.board.num_boxes
    edges_array = np.asarray(state.edges, dtype=np.int8).copy()
    owners_array = np.asarray(state.owners, dtype=np.int8).copy()
    cdef signed char[:] edges = edges_array, owners = owners_array
    cdef int[:] remaining = np.zeros(n_boxes, dtype=np.int32)
    legal_array = np.empty(n_edges, dtype=np.intp)
    safe_array = np.empty(n_edges, dtype=np.intp)
    captures_array = np.empty(n_edges, dtype=np.intp)
    cdef Py_ssize_t[:] legal = legal_array, safe = safe_array, captures = captures_array
    cdef unsigned char[:] unseen = np.empty(n_boxes, dtype=np.uint8)
    cdef Py_ssize_t[:] pending = np.empty(n_boxes, dtype=np.intp)
    cdef Py_ssize_t box, edge, index, a, b, action, steps = 0, available = 0
    cdef Py_ssize_t n_legal, n_safe, n_captures, score
    cdef int high_degree = 0, captured, count, maximum
    cdef int player = state.player, original_player = state.player
    cdef object groups, future
    cdef double outcome
    if edges.shape[0] != n_edges or owners.shape[0] != n_boxes:
        raise ValueError("State dimensions do not match the board.")
    for edge in range(n_edges):
        available += edges[edge] < 0
    for box in range(n_boxes):
        for index in range(4):
            remaining[box] += edges[boxes[box, index]] < 0
        high_degree += remaining[box] >= 3
    while available:
        if oracle is not None and not high_degree:
            groups = groups_for(boxes, neighbors, edges, owners, remaining, unseen, pending)
            if groups is not None:
                future = oracle.fast_future(groups)
                if future is not None:
                    score = 0
                    for box in range(n_boxes):
                        score += (owners[box] == player) - (owners[box] == 1 - player)
                    score += future
                    outcome = (score > 0) - (score < 0)
                    return outcome if player == original_player else -outcome, steps, True
        n_legal = n_safe = n_captures = maximum = 0
        for edge in range(n_edges):
            if edges[edge] >= 0:
                continue
            legal[n_legal] = edge
            n_legal += 1
            a, b = neighbors[edge, 0], neighbors[edge, 1]
            count = (remaining[a] == 1) + (b >= 0 and remaining[b] == 1)
            if count > maximum:
                maximum = count
                n_captures = 0
            if count == maximum and count:
                captures[n_captures] = edge
                n_captures += 1
            if remaining[a] > 2 and (b < 0 or remaining[b] > 2):
                safe[n_safe] = edge
                n_safe += 1
        if maximum:
            action = rng.choice(captures_array[:n_captures])
        elif n_safe:
            action = rng.choice(safe_array[:n_safe])
        else:
            action = rng.choice(legal_array[:n_legal])
        edges[action] = player
        available -= 1
        captured = 0
        for index in range(2):
            box = neighbors[action, index]
            if box < 0:
                continue
            high_degree -= remaining[box] == 3
            remaining[box] -= 1
            if not remaining[box]:
                owners[box] = player
                captured += 1
        if on_move is not None:
            on_move(action)
        if not captured:
            player = 1 - player
        steps += 1
    score = 0
    for box in range(n_boxes):
        score += (owners[box] == original_player) - (owners[box] == 1 - original_player)
    return float((score > 0) - (score < 0)), steps, False
