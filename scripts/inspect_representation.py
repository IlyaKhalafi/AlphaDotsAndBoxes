"""Compare retained-network values on two reachable large loop endgames."""

import argparse
import hashlib
import json
from pathlib import Path

from alphaboxes.checkpoint import load_evaluator
from alphaboxes.endgame import DegreeTwoSolver, position
from alphaboxes.game import State


def loop_fixture(mixed: bool) -> tuple[State, list[int]]:
    """Change only central loops beyond the six-layer border receptive field."""
    state = State.new(14, 14)
    pairs = set()

    def loop(cells):
        for a, b in zip(cells, cells[1:] + cells[:1], strict=True):
            pairs.add(tuple(sorted((a, b))))

    for row in range(0, 14, 2):
        for col in range(0, 14, 2):
            if mixed and 4 <= row < 8 and 4 <= col < 10:
                continue
            a = row * 14 + col
            loop((a, a + 1, a + 15, a + 14))
    if mixed:
        for row in (4, 6):
            for col in (4, 7):
                a = row * 14 + col
                loop((a, a + 1, a + 2, a + 16, a + 15, a + 14))
    free = {edge for edge, boxes in enumerate(state.board.edge_boxes) if tuple(boxes) in pairs}
    edges_by_pair = {
        tuple(boxes): edge for edge, boxes in enumerate(state.board.edge_boxes) if len(boxes) == 2
    }
    corner = [edges_by_pair[pair] for pair in ((1, 15), (14, 15), (0, 1), (0, 14))]
    neutral = state.board.boxes[-1][1]
    opening = [e for e in range(state.board.num_edges) if e not in free and e != neutral]
    # Capture one corner box, pass on a safe edge, then let the other player take three.
    opening.extend((corner[0], corner[1], neutral, corner[2], corner[3]))
    for action in opening:
        state = state.play(action)
    assert state.scores == (1, 3) and state.player == 1
    return state, opening


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/larger.npz"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluator, _ = load_evaluator(args.checkpoint)
    solver = DegreeTwoSolver()
    records = []
    for mixed in (False, True):
        state, opening = loop_fixture(mixed)
        _, neural = evaluator(state)
        exact = solver.value(state)
        records.append(
            {
                "mixed_loops": mixed,
                "scores": state.scores,
                "player": state.player,
                "opening": opening,
                "components": position(state),
                "neural_value": neural,
                "exact_margin": exact,
                "exact_outcome": (exact > 0) - (exact < 0),
            }
        )
    report = {
        "board": [14, 14],
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "records": records,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path("src/alphaboxes/numpy_network.py"),
                Path("src/alphaboxes/graph.py"),
                Path("src/alphaboxes/components.py"),
                Path("src/alphaboxes/endgame.py"),
            )
        },
        "absolute_neural_value_difference": abs(
            records[0]["neural_value"] - records[1]["neural_value"]
        ),
        "scope": "Reachable 14x14 representation diagnostic, not a playing-strength match",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "values": [(r["neural_value"], r["exact_outcome"]) for r in records],
                "difference": report["absolute_neural_value_difference"],
            }
        )
    )


if __name__ == "__main__":
    main()
