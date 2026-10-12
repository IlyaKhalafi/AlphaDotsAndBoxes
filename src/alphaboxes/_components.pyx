"""Optional component encoding using typed traversal buffers."""

import numpy as np


def structural_features(state):
    cdef Py_ssize_t count = state.board.num_boxes
    cdef Py_ssize_t first, box, edge, other, index, top, size
    cdef int remaining, boundaries, fronts, minimum, maximum
    cdef int chains = 0, chain_boxes = 0, threes = 0
    cdef int loops = 0, loop_boxes = 0, fours = 0, sixes = 0
    cdef int short = 0, active = 0, branching = 0, ours = 0, theirs = 0, unclaimed = 0
    cdef bint chain, loop
    cdef object neighbors, box_edges = state.board.boxes
    cdef object edge_boxes = state.board.edge_boxes
    cdef object edges = state.edges, owners = state.owners
    cdef unsigned char[:] unseen = np.zeros(count, dtype=np.uint8)
    cdef Py_ssize_t[:] pending = np.empty(count, dtype=np.intp)
    cdef Py_ssize_t[:] members = np.empty(count, dtype=np.intp)
    nodes = np.zeros((count, 4), dtype=np.float32)
    context = np.zeros(14, dtype=np.float32)
    cdef float[:, :] node_values = nodes
    cdef float[:] global_values = context

    if len(owners) != count or len(edges) != state.board.num_edges:
        raise ValueError("State dimensions do not match the board.")
    for box in range(count):
        unseen[box] = owners[box] < 0
        ours += owners[box] == state.player
        theirs += owners[box] == 1 - state.player
        unclaimed += owners[box] == -1
    for first in range(count):
        if not unseen[first]:
            continue
        unseen[first] = 0
        pending[0] = first
        top = 1
        size = boundaries = fronts = maximum = 0
        minimum = 4
        while top:
            top -= 1
            box = pending[top]
            members[size] = box
            size += 1
            remaining = 0
            for index in range(4):
                edge = box_edges[box][index]
                if edges[edge] >= 0:
                    continue
                remaining += 1
                neighbors = edge_boxes[edge]
                if len(neighbors) == 1:
                    boundaries += 1
                else:
                    other = neighbors[0] if neighbors[1] == box else neighbors[1]
                    if unseen[other]:
                        unseen[other] = 0
                        pending[top] = other
                        top += 1
            fronts += remaining == 1
            if remaining < minimum:
                minimum = remaining
            if remaining > maximum:
                maximum = remaining
        chain = minimum == maximum == 2 and boundaries == 2
        loop = minimum == maximum == 2 and boundaries == 0
        for index in range(size):
            box = members[index]
            node_values[box, 0] = size / 4.0
            node_values[box, 1] = chain
            node_values[box, 2] = loop
            node_values[box, 3] = min(fronts, 2) / 2.0
        if chain:
            if size >= 3:
                chains += 1
                chain_boxes += size
                threes += size == 3
            else:
                short += 1
        if loop:
            loops += 1
            loop_boxes += size
            fours += size == 4
            sixes += size == 6
        active += fronts > 0
        branching += maximum > 2 or minimum == 0
    global_values[0] = chain_boxes / 4.0
    global_values[1] = chains / 4.0
    global_values[2] = threes / 4.0
    global_values[3] = loop_boxes / 4.0
    global_values[4] = loops / 4.0
    global_values[5] = fours / 4.0
    global_values[6] = sixes / 4.0
    global_values[7] = short / 4.0
    global_values[8] = active / 4.0
    global_values[9] = branching / 4.0
    global_values[10] = chains % 2
    global_values[11] = loops % 2
    global_values[12] = (ours - theirs) / 4.0
    global_values[13] = unclaimed / 4.0
    return nodes, context
