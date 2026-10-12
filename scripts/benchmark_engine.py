"""Compare search engines with independently selected NumPy checkpoints.

Run with one BLAS thread for reproducible CPU resource use. Matches swap seats
on identical random openings; wall-clock budgets make exact traces nondeterministic.
"""

import argparse
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from time import monotonic

import numpy as np

from alphaboxes.checkpoint import load_evaluator
from alphaboxes.engine import AlphaBeta, EngineConfig
from alphaboxes.game import State
from alphaboxes.opponents import solve_remaining
from alphaboxes.rollouts import RolloutEvaluator
from alphaboxes.search import MCTS, SearchConfig, choose_child
from alphaboxes.timed_search import BoundedSearch


def planner(evaluator, engine, seconds, search_options=None, simulations=8192):
    options = {"simulations": simulations, "time_limit": seconds, "exact_threshold": 12}
    options.update(search_options or {})
    if isinstance(evaluator, BoundedSearch):
        evaluator.config = (
            EngineConfig(time_limit=seconds) if engine == "alpha_beta" else SearchConfig(**options)
        )
        return evaluator
    if engine == "alpha_beta":
        return AlphaBeta(evaluator, EngineConfig(time_limit=seconds))
    return MCTS(evaluator, SearchConfig(**options))


def measure(search, state):
    statistics = getattr(getattr(search, "evaluator", None), "statistics", lambda: {})
    before = statistics()
    started = monotonic()
    policy, value = search.policy(state)
    return {
        "remaining_edges": len(state.legal_actions),
        "seconds": monotonic() - started,
        "action": int(policy.argmax()),
        "value": value,
        "simulations": getattr(search, "completed_simulations", 0),
        "endgame_hits": getattr(search, "endgame_hits", 0),
        "proven_nodes": getattr(search, "proven_nodes", 0),
        "inherited_visits": getattr(search, "inherited_visits", 0),
        "timed_out": getattr(search, "timed_out", False),
        "depth": getattr(search, "depth", None),
        "nodes": getattr(search, "nodes", None),
        "cutoffs": getattr(search, "cutoffs", None),
        "table_hits": getattr(search, "table_hits", None),
        **{key: count - before[key] for key, count in statistics().items()},
    }


def latency(evaluator, args):
    results = []
    for size in args.boards:
        state = State.new(size, size)
        states = {len(state.legal_actions): state}
        rng = np.random.default_rng(args.seed)
        targets = {size * (size + 1), max(13, int(state.board.num_edges * 0.18))}
        while not state.terminal:
            state = state.play(int(rng.choice(state.legal_actions)))
            if len(state.legal_actions) in targets:
                states[len(state.legal_actions)] = state
        for remaining, state in states.items():
            for seconds in args.seconds:
                for engine in ("mcts", "alpha_beta"):
                    samples = []
                    for _ in range(args.repeats):
                        evaluator.clear_cache()
                        samples.append(measure(planner(evaluator, engine, seconds), state))
                    results.append(
                        {
                            "board": [size, size],
                            "remaining_edges": remaining,
                            "engine": engine,
                            "time_limit": seconds,
                            "samples": samples,
                        }
                    )
    return results


def play_pair(evaluators, args, size, pair_index):
    games = []
    for index in (pair_index * 2, pair_index * 2 + 1):
        for evaluator in evaluators.values():
            evaluator.clear_cache()
        rng = np.random.default_rng(args.seed + pair_index)
        state = State.new(size, size)
        opening = []
        for _ in range(6):
            action = int(rng.choice(state.legal_actions))
            opening.append(action)
            state = state.play(action)
        alpha_player = index % 2
        moves = []
        reused_planners = {}
        move_log = None
        if args.move_progress_dir:
            move_log = args.move_progress_dir / f"{size}x{size}-game-{index:04d}.jsonl"
            with move_log.open("x") as output:
                output.write(json.dumps({"opening": opening, "alpha_player": alpha_player}) + "\n")
        while not state.terminal:
            side = "candidate" if state.player == alpha_player else "reference"
            engine = args.candidate_engine if side == "candidate" else args.reference_engine
            seconds = args.match_seconds if side == "candidate" else args.reference_seconds
            options = args.search_options if side == "candidate" else args.reference_search_options
            if engine == "mcts" and options.get("reuse_tree", False):
                if side not in reused_planners:
                    reused_planners[side] = planner(
                        evaluators[side], engine, seconds, options, args.match_simulations
                    )
                search = reused_planners[side]
            else:
                search = planner(evaluators[side], engine, seconds, options, args.match_simulations)
            step = measure(search, state)
            moves.append({"player": state.player, "side": side, "engine": engine, **step})
            state = state.play(step["action"])
            if move_log:
                with move_log.open("a") as output:
                    output.write(
                        json.dumps({"move": len(moves), "scores": list(state.scores), **moves[-1]})
                        + "\n"
                    )
        margin = state.scores[alpha_player] - state.scores[1 - alpha_player]
        games.append(
            {
                "board": [size, size],
                "index": index,
                "alpha_player": alpha_player,
                "opening": opening,
                "scores": list(state.scores),
                "alpha_margin": margin,
                "moves": moves,
            }
        )
        print(f"{size}x{size} game {index + 1}: candidate margin {margin}", flush=True)
    return games


def load_players(args):
    players = {
        "candidate": load_evaluator(args.checkpoint)[0],
        "reference": load_evaluator(args.reference_checkpoint or args.checkpoint)[0],
    }
    if getattr(args, "candidate_rollouts", 0):
        players["candidate"] = RolloutEvaluator(
            players["candidate"],
            args.candidate_rollouts,
            args.rollout_weight,
            args.rollout_seed,
            backend=args.rollout_backend,
            max_remaining_fraction=args.rollout_max_remaining,
        )
    if args.response_deadline:
        players["candidate"] = BoundedSearch(players["candidate"], timeout=args.response_deadline)
    return players


def worker_pair(args, size, pair_index):
    players = load_players(args)
    try:
        return play_pair(players, args, size, pair_index)
    finally:
        for player in players.values():
            if isinstance(player, BoundedSearch):
                player.close()


def matches(evaluators, args):
    tasks = [(size, pair) for size in args.match_boards for pair in range(args.games // 2)]
    games = []

    def record(pair):
        games.extend(pair)
        if args.progress:
            with args.progress.open("a") as output:
                output.write(json.dumps({"completed_games": len(games), "games": pair}) + "\n")

    if args.workers == 1:
        for size, pair in tasks:
            record(play_pair(evaluators, args, size, pair))
    elif tasks:
        # Spawn avoids inheriting BLAS state or sharing evaluator caches.
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            pending = [pool.submit(worker_pair, args, size, pair) for size, pair in tasks]
            for future in as_completed(pending):
                record(future.result())
    return sorted(games, key=lambda game: (game["board"], game["index"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/larger.npz"))
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--reference-engine", choices=("mcts", "alpha_beta"), default="mcts")
    parser.add_argument("--candidate-engine", choices=("mcts", "alpha_beta"), default="alpha_beta")
    parser.add_argument("--candidate-rollouts", type=int, default=0)
    parser.add_argument("--rollout-weight", type=float, default=1.0)
    parser.add_argument("--rollout-seed", type=int, default=1729)
    parser.add_argument("--rollout-backend", choices=("python", "cython"), default="python")
    parser.add_argument("--rollout-max-remaining", type=float, default=1.0)
    parser.add_argument(
        "--search-options", type=json.loads, default={}, help="Candidate MCTS config JSON"
    )
    parser.add_argument(
        "--reference-search-options", type=json.loads, default={}, help="Reference MCTS config JSON"
    )
    parser.add_argument("--reference-seconds", type=float)
    parser.add_argument(
        "--response-deadline", type=float, help="Supervise candidate in a warm process"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--boards", type=int, nargs="+", default=[4, 5, 10])
    parser.add_argument("--seconds", type=float, nargs="+", default=[0.5, 2])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--match-boards", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--match-seconds", type=float, default=0.1)
    parser.add_argument("--match-simulations", type=int, default=8192)
    parser.add_argument(
        "--untimed", action="store_true", help="Equal MCTS work without move timers"
    )
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3501)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--move-progress-dir", type=Path)
    args = parser.parse_args()
    if (
        not 0 <= args.candidate_rollouts <= 16
        or not np.isfinite(args.rollout_weight)
        or not 0 <= args.rollout_weight <= 1
        or args.rollout_seed < 0
        or not np.isfinite(args.rollout_max_remaining)
        or not 0 < args.rollout_max_remaining <= 1
    ):
        parser.error("Use 0..16 rollouts, a weight in [0, 1] and a nonnegative rollout seed.")
    if not isinstance(args.search_options, dict) or not isinstance(
        args.reference_search_options, dict
    ):
        parser.error("Search options must be a JSON object.")
    if args.reference_seconds is None:
        args.reference_seconds = args.match_seconds
    if args.match_simulations < 1:
        parser.error("Match simulations must be positive.")
    if args.untimed:
        if (
            args.candidate_engine != "mcts"
            or args.reference_engine != "mcts"
            or args.response_deadline is not None
            or not args.skip_latency
        ):
            parser.error(
                "Untimed matches require both MCTS engines, no response deadline "
                "and --skip-latency."
            )
        if any(
            name in options
            for options in (args.search_options, args.reference_search_options)
            for name in ("time_limit", "simulations")
        ):
            parser.error("Untimed match budgets must be set with --match-simulations.")
        args.match_seconds = args.reference_seconds = None
    if args.response_deadline is not None and not 0 < args.response_deadline < 2:
        parser.error("Response deadline must be positive and below two seconds.")
    if args.response_deadline is not None and args.search_options.get("reuse_tree", False):
        parser.error(
            "The supervised worker does not retain search trees; omit --response-deadline."
        )
    try:
        SearchConfig(**args.search_options)
        SearchConfig(**args.reference_search_options)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if args.games < 0 or args.games % 2 or args.repeats < 1:
        parser.error("Use a nonnegative even match count and positive repeats.")
    if not 1 <= args.workers <= 8:
        parser.error("Use 1..8 single-threaded match workers.")
    if args.progress:
        args.progress.parent.mkdir(parents=True, exist_ok=True)
        if args.progress.exists():
            parser.error("Progress output already exists; use a fresh path.")
    if args.move_progress_dir:
        args.move_progress_dir.mkdir(parents=True, exist_ok=True)
        if any(args.move_progress_dir.iterdir()):
            parser.error("Move progress directory must be empty; use a fresh path.")
    if not all(2 <= size <= 12 for size in args.boards + args.match_boards):
        parser.error("Benchmark board sizes must be 2..12.")
    for seconds in args.seconds + [args.match_seconds, args.reference_seconds]:
        if seconds is not None:
            EngineConfig(time_limit=seconds)
    evaluators = load_players(args)
    modules = [
        sys.modules[f"alphaboxes.{name}"]
        for name in (
            "engine",
            "game",
            "graph",
            "components",
            "numpy_network",
            "opponents",
            "search",
            "root_halving",
            "rollouts",
            "checkpoint",
            "endgame",
            "timed_search",
        )
    ]
    extensions = [sys.modules[fn.__module__] for fn in (choose_child, solve_remaining) if fn]
    candidate = evaluators["candidate"]
    if isinstance(candidate, BoundedSearch):
        candidate = candidate.evaluator
    if isinstance(candidate, RolloutEvaluator) and candidate.backend == "cython":
        extensions.append(sys.modules[candidate.playout.__module__])
    report = {
        "started_utc": datetime.now(UTC).isoformat(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "reference_checkpoint_sha256": hashlib.sha256(
            (args.reference_checkpoint or args.checkpoint).read_bytes()
        ).hexdigest(),
        "reference_engine": args.reference_engine,
        "candidate_engine": args.candidate_engine,
        "candidate_value_estimator": {
            "rollouts": args.candidate_rollouts,
            "weight": args.rollout_weight,
            "seed": args.rollout_seed,
            "backend": args.rollout_backend,
            "max_remaining_fraction": args.rollout_max_remaining,
            "oracle": "Long-component recurrence; general multi-front positions excluded",
        }
        if args.candidate_rollouts
        else {"rollouts": 0},
        "search_options": args.search_options,
        "reference_search_options": args.reference_search_options,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threads": {
            key: os.environ.get(key)
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "seed": args.seed,
        "match_seconds": args.match_seconds,
        "match_simulations": args.match_simulations,
        "budget_mode": "simulations" if args.untimed else "seconds",
        "reference_seconds": args.reference_seconds,
        "response_deadline": args.response_deadline,
        "games_per_board": args.games,
        "workers": args.workers,
        "root_exact_edges": 12,
        "acceleration": {
            "search": choose_child is not None,
            "endgame": solve_remaining is not None,
            "rollout": args.candidate_rollouts > 0 and args.rollout_backend == "cython",
        },
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                *(Path(module.__file__) for module in modules + extensions),
            )
        },
        "latency": [] if args.skip_latency else latency(evaluators["candidate"], args),
        "games": matches(evaluators, args),
    }
    report["completed_utc"] = datetime.now(UTC).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for player in evaluators.values():
        if isinstance(player, BoundedSearch):
            player.close()


if __name__ == "__main__":
    main()
