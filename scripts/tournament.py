"""Evaluate spaced checkpoints while training and retain a transparent leaderboard."""

import argparse
import json
import time
from contextlib import redirect_stdout
from pathlib import Path

from alphaboxes.evaluation import compare_agents, evaluate


def last_metric(path: Path) -> dict:
    if path.exists():
        for line in reversed(path.read_text().splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue  # An active writer may not have finished its final line.
    return {}


def run_evaluation(path: Path, label: str, args) -> dict:
    receipt_path = args.output / f"{label}.json"
    with (args.output / "evaluation.log").open("a") as log, redirect_stdout(log):
        receipt = evaluate(
            path,
            receipt_path,
            args.sizes,
            games=args.games,
            simulations=args.simulations,
            seed=args.seed,
            opponent_names=["tactical_endgame", "chain_control"],
        )
        duel = compare_agents(
            path,
            args.baseline,
            args.output / f"{label}-duel.json",
            args.sizes,
            games=args.games,
            simulations=args.simulations,
            seed=args.seed,
        )
    scores = {
        f"{r['size'][0]}x{r['size'][1]}/"
        f"{'released_agent' if r['opponent'] == 'checkpoint' else r['opponent']}": r["score_rate"]
        for r in receipt["results"] + duel["results"]
    }
    row = {
        "label": label,
        "checkpoint": str(path),
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "mean_score": sum(scores.values()) / len(scores),
        "scores": scores,
    }
    print(json.dumps(row), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3030)
    parser.add_argument("--sizes", default="5x5")
    parser.add_argument("--every", type=int, default=40)
    parser.add_argument("--watch-seconds", type=float, default=0)
    args = parser.parse_args()
    args.sizes = [tuple(map(int, size.split("x"))) for size in args.sizes.split(",")]
    if args.every < 1 or args.watch_seconds < 0:
        parser.error("Checkpoint interval must be positive; watch seconds must be nonnegative.")
    args.output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.watch_seconds
    rows = [run_evaluation(args.baseline, "baseline", args)]
    completed = set()
    while True:
        metrics = args.run / "metrics.jsonl"
        last = last_metric(metrics)
        finished = bool(last.get("stop_reason"))
        final_path = args.run / f"agent-{last.get('iteration', 0):05d}.pt"
        candidates = [
            path
            for path in sorted(args.run.glob("agent-*.pt"))
            if int(path.stem.split("-")[-1]) % args.every == 0 or (finished and path == final_path)
        ]
        for path in candidates:
            if path not in completed:
                rows.append(run_evaluation(path, path.stem, args))
                completed.add(path)
        # Keep baseline on an exact tie; selection noise is not evidence of improvement.
        ranked = sorted(rows, key=lambda row: row["mean_score"], reverse=True)
        summary = {
            "criterion": "Mean score vs tactical_endgame, chain_control, released_agent",
            "sizes": args.sizes,
            "seed": args.seed,
            "games_per_opponent": args.games,
            "selected": ranked[0],
            "leaderboard": ranked,
            "training_finished": finished,
        }
        temporary = args.output / "leaderboard.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(args.output / "leaderboard.json")
        if (finished and final_path in completed) or time.monotonic() >= deadline:
            break
        time.sleep(min(30, max(0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
