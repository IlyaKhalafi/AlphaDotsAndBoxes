"""Immutable rules and cached board geometry. Dimensions count boxes, not dots."""

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class Board:
    rows: int
    cols: int
    edges: tuple[tuple[int, int, int, int], ...]
    boxes: tuple[tuple[int, int, int, int], ...]
    edge_boxes: tuple[tuple[int, ...], ...]

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def num_boxes(self) -> int:
        return self.rows * self.cols

    @property
    def num_nodes(self) -> int:
        return self.num_edges + self.num_boxes


@lru_cache(maxsize=128)
def board(rows: int, cols: int) -> Board:
    if not isinstance(rows, int) or not isinstance(cols, int) or rows < 1 or cols < 1:
        raise ValueError("Board dimensions must be positive integers.")
    horizontal = tuple((r, c, r, c + 1) for r in range(rows + 1) for c in range(cols))
    vertical = tuple((r, c, r + 1, c) for r in range(rows) for c in range(cols + 1))
    offset = len(horizontal)
    boxes = tuple(
        (
            r * cols + c,
            (r + 1) * cols + c,
            offset + r * (cols + 1) + c,
            offset + r * (cols + 1) + c + 1,
        )
        for r in range(rows)
        for c in range(cols)
    )
    neighbors: list[list[int]] = [[] for _ in horizontal + vertical]
    for b, edges in enumerate(boxes):
        for edge in edges:
            neighbors[edge].append(b)
    return Board(rows, cols, horizontal + vertical, boxes, tuple(map(tuple, neighbors)))


@dataclass(frozen=True, slots=True)
class State:
    board: Board
    edges: tuple[int, ...]
    owners: tuple[int, ...]
    player: int = 0

    @classmethod
    def new(cls, rows: int = 3, cols: int = 3) -> "State":
        geometry = board(rows, cols)
        return cls(geometry, (-1,) * geometry.num_edges, (-1,) * geometry.num_boxes)

    @property
    def legal_actions(self) -> tuple[int, ...]:
        return tuple(i for i, owner in enumerate(self.edges) if owner < 0)

    @property
    def terminal(self) -> bool:
        return all(owner >= 0 for owner in self.edges)

    @property
    def scores(self) -> tuple[int, int]:
        return self.owners.count(0), self.owners.count(1)

    def outcome(self, player: int) -> float:
        """Terminal win/draw/loss from a specified player's perspective."""
        if not self.terminal:
            raise ValueError("An unfinished game has no outcome.")
        a, b = self.scores
        result = float((a > b) - (a < b))
        return result if player == 0 else -result

    def captures(self, action: int) -> tuple[int, ...]:
        return tuple(
            b
            for b in self.board.edge_boxes[action]
            if self.owners[b] < 0
            and all(e == action or self.edges[e] >= 0 for e in self.board.boxes[b])
        )

    def play(self, action: int) -> "State":
        if self.terminal:
            raise ValueError("The game has finished.")
        if not isinstance(action, int) or not 0 <= action < len(self.edges):
            raise ValueError("Edge index is outside the board.")
        if self.edges[action] >= 0:
            raise ValueError("That edge is already drawn.")
        captured = self.captures(action)
        edges = list(self.edges)
        owners = list(self.owners)
        edges[action] = self.player
        for box_index in captured:
            owners[box_index] = self.player
        return State(
            self.board, tuple(edges), tuple(owners), self.player if captured else 1 - self.player
        )

    def as_dict(self) -> dict:
        return {
            "rows": self.board.rows,
            "cols": self.board.cols,
            "edges": list(self.edges),
            "owners": list(self.owners),
            "geometry": self.board.edges,
            "player": self.player,
            "scores": self.scores,
            "terminal": self.terminal,
        }
