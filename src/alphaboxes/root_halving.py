"""Budgeted root exploration: compare a shortlist, then halve it repeatedly.

This deterministic acting variant is inspired by sequential halving in
Danihelka et al., Policy Improvement by Planning with Gumbel (ICLR 2022).
Interior nodes keep their existing PUCT rule; no Gumbel sampling is claimed.
"""

from collections.abc import Callable, Mapping
from math import ceil, log, log2
from typing import Protocol


class Branch(Protocol):
    prior: float
    visits: int


class RootHalving:
    """Spend new simulations evenly within each progressively smaller shortlist."""

    def __init__(
        self,
        children: Mapping[int, Branch],
        budget: int,
        width: int,
        value: Callable[[Branch], float],
    ):
        self.children, self.budget, self.value = children, budget, value
        self.active = sorted(children, key=lambda a: children[a].prior, reverse=True)[
            : min(width, budget)
        ]
        self.rounds_left = ceil(log2(len(self.active))) if len(self.active) > 1 else 1
        self.target = max(1, budget // (self.rounds_left * len(self.active)))

    def _scores(self, actions) -> dict[int, float]:
        scale = 50 + max(branch.visits for branch in self.children.values())
        return {
            action: log(max(self.children[action].prior, 1e-12))
            + scale * (self.value(self.children[action]) if self.children[action].visits else 0.0)
            for action in actions
        }

    def select(self) -> Branch:
        scores = self._scores(self.active)
        if len(self.active) > 1 and all(
            self.children[a].visits >= self.target for a in self.active
        ):
            self.active = sorted(self.active, key=scores.__getitem__, reverse=True)[
                : ceil(len(self.active) / 2)
            ]
            self.rounds_left = max(1, self.rounds_left - 1)
            remaining = self.budget - sum(branch.visits for branch in self.children.values())
            quota = max(1, remaining // (self.rounds_left * len(self.active)))
            self.target = min(self.children[a].visits for a in self.active) + quota
        action = min(self.active, key=lambda a: (self.children[a].visits, -scores[a]))
        return self.children[action]

    def best_action(self) -> int:
        evaluated = [a for a in self.active if self.children[a].visits]
        return max(self._scores(evaluated or self.active).items(), key=lambda pair: pair[1])[0]
