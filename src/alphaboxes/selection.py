"""Exact PUCT selection without rescoring every unvisited sibling."""

from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphaboxes.search import Node


class ChildSelection:
    """Rank visited children and leading unvisited priors without group objects.

    Unvisited children share one value and denominator, so their priority is
    monotonic in their prior. Equal floating-point scores still use original
    dictionary order, including ties between different priors. Rebuild after
    changing child priors. Visit/value updates are read directly from nodes.
    """

    def __init__(self, children: dict[int, "Node"]):
        self.children = children
        self.size = len(children)
        ordered = list(enumerate(children.values()))
        self.visited = {index: child for index, child in ordered if child.visits}
        self.ranked = sorted(ordered, key=lambda item: item[1].prior, reverse=True)
        self.group_ends = {child.prior: rank + 1 for rank, (_, child) in enumerate(self.ranked)}
        self.cursor = 0

    def select(self, player: int, scale: float, unvisited_value: float):
        def score(child):
            value = child.value if child.player == player else -child.value
            return (value if child.visits else unvisited_value) + scale * child.prior / (
                1 + child.visits
            )

        best, best_score, best_index = None, float("-inf"), self.size
        for index, child in self.visited.items():
            visits = child.visits
            if visits:
                value = child.value_sum / visits
                if child.player != player:
                    value = -value
            else:
                value = unvisited_value
            priority = value + scale * child.prior / (1 + visits)
            if not isfinite(priority):
                return max(self.children.values(), key=score)
            if priority > best_score or (priority == best_score and index < best_index):
                best, best_score, best_index = child, priority, index
        rank = self.cursor
        while rank < self.size:
            index, child = self.ranked[rank]
            if child.visits:
                if rank == self.cursor:
                    self.cursor += 1
                rank += 1
                continue
            priority = unvisited_value + scale * child.prior
            if priority < best_score:
                break
            if priority > best_score or (priority == best_score and index < best_index):
                best, best_score, best_index = child, priority, index
            # Equal priors are stable-sorted in original dictionary order.
            # Later unvisited entries in this run cannot beat its first one.
            rank = self.group_ends[child.prior]
        if best is None:
            raise ValueError("Selection requires nonempty children.")
        # A cancelled simulation may leave this child unvisited; its score
        # remains the same unvisited value until an actual backup arrives.
        self.visited[best_index] = best
        return best
