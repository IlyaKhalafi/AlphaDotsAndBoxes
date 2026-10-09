"""Export reproducible run metrics and evaluation receipts to the docs directory."""

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import statistics
from pathlib import Path

FIELDS = [
    "iteration",
    "games_total",
    "positions_total",
    "policy_loss",
    "value_loss",
    "total_loss",
    "sample_seconds",
    "update_seconds",
    "iteration_seconds",
    "elapsed_seconds",
    "gpu_peak_mb",
]


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--evaluation", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("docs/data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"python": platform.python_version(), "software": {}, "runs": []}
    for package in ["torch", "ray", "gymnasium", "numpy", "fastapi"]:
        summary["software"][package] = importlib.metadata.version(package)
    for run in args.run:
        rows = [
            clean(json.loads(line)) for line in (run / "metrics.jsonl").read_text().splitlines()
        ]
        with (args.output / f"training-{run.name}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(
                file, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
        result = {
            "label": run.name,
            "config": json.loads((run / "config.json").read_text()),
            "first_iteration": rows[0]["iteration"],
            "last_iteration": rows[-1]["iteration"],
            "games_total": rows[-1]["games_total"],
            "positions_total": rows[-1]["positions_total"],
            "initialization": rows[-1].get("initialization", {}),
            "stop_reason": rows[-1].get("stop_reason"),
            "elapsed_seconds": rows[-1]["elapsed_seconds"],
            "median_iteration_seconds": statistics.median(row["iteration_seconds"] for row in rows),
            "peak_allocated_mb": max(row["gpu_peak_mb"] for row in rows),
            "last10_mean_policy_loss": statistics.mean(row["policy_loss"] for row in rows[-10:]),
            "last10_mean_value_loss": statistics.mean(row["value_loss"] for row in rows[-10:]),
        }
        summary["runs"].append(result)
    for evaluation in args.evaluation:
        result = clean(json.loads(evaluation.read_text()))
        (args.output / evaluation.name).write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
    (args.output / "training-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
