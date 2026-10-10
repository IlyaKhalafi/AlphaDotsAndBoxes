"""Optional exact endgames over local remaining-edge masks (at most 18 bits)."""

from array import array
from libc.stdint cimport int16_t, int32_t, uint32_t
cimport cython


cdef int captures(uint32_t occupied, int action, const uint32_t* boxes,
                  const int32_t* neighbors) noexcept nogil:
    cdef int side, box, count = 0
    for side in range(2):
        box = neighbors[2 * action + side]
        if box >= 0 and (occupied & boxes[box]) == boxes[box]:
            count += 1
    return count


cdef int future(uint32_t occupied, uint32_t full, const uint32_t* boxes,
                const int32_t* neighbors, int16_t* memo, int edges) noexcept nogil:
    cdef int action, taken, continuation, score, best = -32767
    cdef uint32_t bit, following
    if occupied == full:
        return 0
    if memo[occupied] != -32768:
        return memo[occupied]
    for action in range(edges):
        bit = <uint32_t>1 << action
        if occupied & bit:
            continue
        following = occupied | bit
        taken = captures(following, action, boxes, neighbors)
        continuation = future(following, full, boxes, neighbors, memo, edges)
        score = taken + continuation if taken else -continuation
        if score > best:
            best = score
    memo[occupied] = <int16_t>best
    return best


@cython.boundscheck(False)
@cython.wraparound(False)
def solve_remaining(const uint32_t[::1] boxes not None,
                    const int32_t[:, ::1] neighbors not None):
    """Return future score difference and optimal local actions, preserving extra turns."""
    cdef int edges = neighbors.shape[0]
    cdef int action, side, box, taken, continuation, score, best
    cdef uint32_t bit, full
    if edges < 0 or edges > 18 or neighbors.shape[1] != 2:
        raise ValueError("Expected at most 18 remaining edges and two neighbors per edge.")
    if boxes.shape[0] < 1 or boxes.shape[0] > 32767:
        raise ValueError("Box count must fit the score cache.")
    if not edges:
        return 0, ()
    full = (<uint32_t>1 << edges) - 1
    for action in range(edges):
        for side in range(2):
            box = neighbors[action, side]
            if box < -1 or box >= boxes.shape[0]:
                raise ValueError("Neighbor index is outside the box array.")
            if box >= 0 and not (boxes[box] & (<uint32_t>1 << action)):
                raise ValueError("Neighbor box must contain its incident edge.")
    cdef int16_t[::1] memo = array('h', [-32768]) * (1 << edges)
    with nogil:
        best = future(0, full, &boxes[0], &neighbors[0, 0], &memo[0], edges)
    optimal = []
    for action in range(edges):
        bit = <uint32_t>1 << action
        with nogil:
            taken = captures(bit, action, &boxes[0], &neighbors[0, 0])
            continuation = future(bit, full, &boxes[0], &neighbors[0, 0], &memo[0], edges)
            score = taken + continuation if taken else -continuation
        if score == best:
            optimal.append(action)
    return best, tuple(optimal)
