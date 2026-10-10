"""Optional scalar PUCT scoring without Python callbacks or temporary numbers."""

def choose_child(dict children, int player, double scale):
    cdef object child, best = None
    cdef int visits, child_player
    cdef double mean, value_sum, prior, score, best_score = -1e300
    for child in children.values():
        visits = child.visits
        child_player = child.player
        value_sum = child.value_sum
        mean = value_sum / visits if visits else 0.0
        if child_player != player:
            mean = -mean
        prior = child.prior
        score = mean + scale * prior / (1 + visits)
        if best is None or score > best_score:
            best_score, best = score, child
    if best is None:
        raise ValueError("Cannot choose from an empty set of children.")
    return best
