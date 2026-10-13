"""Create board-balanced frozen teacher targets from a trusted training replay."""

import argparse
import json
from pathlib import Path

from alphaboxes.distillation import build_replay_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=Path("models/larger.npz"))
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions-per-board", type=int, default=800)
    parser.add_argument("--seed", type=int, default=51001)
    args = parser.parse_args()
    print(
        json.dumps(
            build_replay_dataset(
                args.teacher, args.replay, args.output, args.positions_per_board, args.seed
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
