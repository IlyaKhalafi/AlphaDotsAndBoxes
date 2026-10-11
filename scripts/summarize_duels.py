"""Summarize repeated, equally budgeted search or checkpoint comparisons."""

import argparse
import json
from pathlib import Path

from alphaboxes.experiments import deployment_gate, pool_duels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = pool_duels([json.loads(path.read_text()) for path in args.receipts])
    result = {
        "receipts": [str(path) for path in args.receipts],
        "results": rows,
        "deployment_gate": deployment_gate(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
