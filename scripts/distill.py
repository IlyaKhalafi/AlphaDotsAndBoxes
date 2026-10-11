"""Train a smaller evaluator against frozen released-network predictions."""

import argparse
from pathlib import Path

from alphaboxes.distillation import distill


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=Path("models/larger.npz"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=3601)
    parser.add_argument("--gpu-memory-gb", type=float, default=8)
    print(distill(**vars(parser.parse_args())), flush=True)


if __name__ == "__main__":
    main()
