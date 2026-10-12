"""Exact PUCT selection without rescoring every unvisited sibling."""

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphaboxes.search import Node


@dataclass(slots=True)
class PriorGroup:
    prior: float
    children: list[tuple[int, "Node"]]
    cursor: int = 0

    def first_unvisited(self):
        while self.cursor < len(self.children) and self.children[self.cursor][1].visits:
            self.cursor += 1
        return self.children[self.cursor] if self.cursor < len(self.children) else None


class ChildSelection:
    """Cache prior groups; rank visited children and leading unvisited groups.

    Unvisited children share one value and denominator, so their priority is
    monotonic in their prior. Equal floating-point scores still use original
    dictionary order, including ties between different priors. Rebuild after
    changing child priors. Visit/value updates are read directly from nodes.
    """

    def __init__(self, children: dict[int, "Node"]):
        self.children = children
        self.size = len(children)
        groups = defaultdict(list)
        self.visited = {}
        for index, child in enumerate(children.values()):
            groups[child.prior].append((index, child))
            if child.visits:
                self.visited[index] = child
        self.groups = [PriorGroup(prior, groups[prior]) for prior in sorted(groups, reverse=True)]
        self.cursor = 0

    def select(self, player: int, scale: float, unvisited_value: float):
        def score(child):
            value = child.value if child.player == player else -child.value
            return (value if child.visits else unvisited_value) + scale * child.prior / (
                1 + child.visits
            )

        best, best_score, best_index = None, float("-inf"), self.size
        for index, child in self.visited.items():
            priority = score(child)
            if not isfinite(priority):
                return max(self.children.values(), key=score)
            if priority > best_score or (priority == best_score and index < best_index):
                best, best_score, best_index = child, priority, index
        for group_index in range(self.cursor, len(self.groups)):
            group = self.groups[group_index]
            candidate = group.first_unvisited()
            if candidate is None:
                if group_index == self.cursor:
                    self.cursor += 1
                continue
            index, child = candidate
            priority = unvisited_value + scale * group.prior
            if priority < best_score:
                break
            if priority > best_score or (priority == best_score and index < best_index):
                best, best_score, best_index = child, priority, index
        if best is None:
            raise ValueError("Selection requires nonempty children.")
        # A cancelled simulation may leave this child unvisited; its score
        # remains the same unvisited value until an actual backup arrives.
        self.visited[best_index] = best
        return best
