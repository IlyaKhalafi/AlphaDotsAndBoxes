"""Test search budgets sequentially with fixed weights and endgame assistance."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[2048, 8192])
    parser.add_argument("--baseline-budget", type=int, default=512)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[3200, 3201])
    parser.add_argument("--sizes", nargs="+", default=["4x4", "5x5"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--exact-threshold", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1 or args.games < 2 or args.games % 2:
        parser.error("Use positive workers and an even game count >= 2.")
    if min([args.baseline_budget, *args.budgets]) < 1:
        parser.error("Search budgets must be positive.")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.budgets)) != len(args.budgets):
        parser.error("Seeds and budgets must be distinct.")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("Board sizes must be distinct.")
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "budgets": args.budgets,
        "baseline_budget": args.baseline_budget,
        "games_per_seed_and_board": args.games,
        "seeds": args.seeds,
        "sizes": args.sizes,
        "exact_threshold": args.exact_threshold,
        "opening_moves": 6,
        "leaf_exact_threshold": 0,
        "reuse_tree": False,
        "limitation": "Seeded screening, not independent confirmation or human-strength evidence.",
    }
    plan_path = args.output / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing output belongs to a different experiment.")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }

    def run_match(job):
        budget, size, seed = job
        output = args.output / f"budget-{budget}-{size}-{seed}.json"
        if not output.exists():
            command = [
                "rtk",
                "proxy",
                sys.executable,
                "-m",
                "alphaboxes.cli",
                "duel",
                "--checkpoint",
                str(args.checkpoint),
                "--opponent-checkpoint",
                str(args.checkpoint),
                "--output",
                str(output),
                "--sizes",
                size,
                "--games",
                str(args.games),
                "--seed",
                str(seed),
                "--simulations",
                str(budget),
                "--opponent-simulations",
                str(args.baseline_budget),
                "--exact-threshold",
                str(args.exact_threshold),
            ]
            with output.with_suffix(".log").open("w") as log:
                subprocess.run(
                    command, env=environment, stdout=log, stderr=log, check=True, timeout=12 * 3600
                )
        receipt = json.loads(output.read_text())
        if (
            receipt["checkpoint_sha256"] != plan["checkpoint_sha256"]
            or receipt["opponent_sha256"] != plan["checkpoint_sha256"]
            or receipt["search_config"]["simulations"] != budget
            or receipt["opponent_search_config"]["simulations"] != args.baseline_budget
            or receipt["exact_threshold"] != args.exact_threshold
            or receipt["seed"] != seed
            or receipt["opening_moves"] != plan["opening_moves"]
            or receipt["results"][0]["games"] != args.games
            or receipt["results"][0]["size"] != list(map(int, size.split("x")))
        ):
            raise ValueError(f"Mismatched match receipt: {output}")
        return receipt

    summary = {}
    for budget in args.budgets:
        status = {"status": "running", "budget": budget, "completed_budgets": list(summary)}
        (args.output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        jobs = [(budget, size, seed) for size in args.sizes for seed in args.seeds]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            receipts = list(executor.map(run_match, jobs))
        rows = []
        for size in args.sizes:
            matches = [
                r["results"][0]
                for r in receipts
                if r["results"][0]["size"] == list(map(int, size.split("x")))
            ]
            counts = {
                key: sum(m[key] for m in matches) for key in ["games", "wins", "draws", "losses"]
            }
            times = [
                sum(m["search_seconds"][i] for m in matches)
                / max(sum(m["move_counts"][i] for m in matches), 1)
                for i in range(2)
            ]
            rows.append(
                {
                    "size": size,
                    **counts,
                    "score_rate": (counts["wins"] + counts["draws"] / 2) / counts["games"],
                    "mean_move_seconds": times,
                }
            )
        summary[budget] = rows
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"budget": budget, "results": rows}), flush=True)
    (args.output / "status.json").write_text(
        json.dumps({"status": "complete", "completed_budgets": list(summary)}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
