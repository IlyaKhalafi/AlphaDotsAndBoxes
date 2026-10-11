"""Seat-balanced evaluation with raw outcomes and uncertainty intervals."""

import importlib.metadata
import json
import math
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from alphaboxes.chains import chain_action
from alphaboxes.checkpoint import load_evaluator
from alphaboxes.game import State
from alphaboxes.opponents import endgame_action, random_action, solve, tactical_action
from alphaboxes.search import MCTS, SearchConfig


def software_versions(*checkpoints: Path) -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    if any(path.suffix != ".npz" for path in checkpoints):
        versions.update({name: importlib.metadata.version(name) for name in ("torch", "ray")})
    return versions


def wilson(wins: int, games: int) -> tuple[float, float]:
    z = 1.96
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games**2)) / denominator
    return max(0, center - radius), min(1, center + radius)


def match_record(size, opponent, outcomes, margins, seats, started, actions_total) -> dict:
    games = len(outcomes)
    wins, draws, losses = outcomes.count(1), outcomes.count(0), outcomes.count(-1)
    return {
        "size": list(size),
        "opponent": opponent,
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / games,
        "score_rate": (wins + draws / 2) / games,
        "win_rate_95ci": list(wilson(wins, games)),
        "mean_box_margin": float(np.mean(margins)),
        "seconds": time.monotonic() - started,
        "agent_moves": actions_total,
        "outcomes": outcomes,
        "box_margins": margins,
        "agent_seats": seats,
    }


def evaluate(
    checkpoint: Path,
    output: Path,
    sizes: list[tuple[int, int]],
    games: int = 40,
    simulations: int = 128,
    exact_threshold: int = 0,
    seed: int = 2026,
    policy_only: bool = False,
    opponent_names: list[str] | None = None,
    leaf_exact_threshold: int = 0,
) -> dict:
    if games < 2 or games % 2:
        raise ValueError("Use an even number of games >= 2 for balanced seats.")
    neural, metadata = load_evaluator(checkpoint)
    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        opponents = {
            "random": random_action,
            "tactical": tactical_action,
            "chain_control": chain_action,
        }
        if State.new(*size).board.num_edges <= 12:
            opponents["exact"] = lambda state, rng: int(rng.choice(solve(state)[1]))
        else:
            opponents["tactical_endgame"] = endgame_action
        selected = (
            opponent_names
            if opponent_names is not None
            else ["random", "tactical", "exact" if "exact" in opponents else "tactical_endgame"]
        )
        if not selected or len(set(selected)) != len(selected) or set(selected) - opponents.keys():
            raise ValueError(
                f"Choose distinct supported opponents for {size}: {', '.join(opponents)}"
            )
        for opponent_name in selected:
            opponent = opponents[opponent_name]
            started = time.monotonic()
            outcomes, scores, seats = [], [], []
            actions_total = 0
            for game in range(games):
                agent_seat = game % 2
                state = State.new(*size)
                search = MCTS(
                    neural,
                    SearchConfig(
                        simulations=simulations,
                        exact_threshold=exact_threshold,
                        leaf_exact_threshold=leaf_exact_threshold,
                    ),
                    seed + game,
                )
                while not state.terminal:
                    if state.player == agent_seat:
                        policy, _ = neural(state) if policy_only else search.policy(state)
                        # Randomize tied visit counts, avoiding edge-index bias.
                        action = int(rng.choice(np.flatnonzero(policy == policy.max())))
                        actions_total += 1
                    else:
                        action = opponent(state, rng)
                    state = state.play(action)
                outcomes.append(state.outcome(agent_seat))
                a, b = state.scores
                scores.append((a - b) * (1 if agent_seat == 0 else -1))
                seats.append(agent_seat)
            record = match_record(
                size, opponent_name, outcomes, scores, seats, started, actions_total
            )
            rows.append(record)
            print(json.dumps(record), flush=True)
    result = {
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "checkpoint_metadata": metadata,
        "software": software_versions(checkpoint),
        "inference": "numpy" if checkpoint.suffix == ".npz" else "torch",
        "seed": seed,
        "simulations": 0 if policy_only else simulations,
        "exact_threshold": 0 if policy_only else exact_threshold,
        "leaf_exact_threshold": 0 if policy_only else leaf_exact_threshold,
        "mode": "policy" if policy_only else "search",
        "opponents": opponent_names,
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def compare_agents(
    checkpoint: Path,
    opponent_checkpoint: Path,
    output: Path,
    sizes: list[tuple[int, int]],
    games: int = 40,
    simulations: int = 128,
    exact_threshold: int = 0,
    seed: int = 3031,
    opening_moves: int = 6,
    search_config: SearchConfig | None = None,
    opponent_search_config: SearchConfig | None = None,
) -> dict:
    """Seat-balanced checkpoint matches with reproducible randomized openings."""
    if games < 2 or games % 2 or opening_moves < 0:
        raise ValueError("Use an even game count >=2 and nonnegative opening_moves.")
    neural, metadata = load_evaluator(checkpoint)
    opponent, opponent_metadata = load_evaluator(opponent_checkpoint)
    evaluators = [neural, opponent]
    config = search_config or SearchConfig(simulations=simulations, exact_threshold=exact_threshold)
    opponent_config = opponent_search_config or config
    rows = []
    for size in sizes:
        started = time.monotonic()
        outcomes, margins, seats, histories = [], [], [], []
        agent_moves = 0
        for game in range(games):
            seat = game % 2
            state = State.new(*size)
            opening_rng = np.random.default_rng(np.random.SeedSequence([seed, game, 0]))
            move_rng = [
                np.random.default_rng(np.random.SeedSequence([seed, game, i + 1])) for i in range(2)
            ]
            searches = [
                MCTS(evaluator, settings, seed + game)
                for evaluator, settings in zip(evaluators, (config, opponent_config), strict=True)
            ]
            moves = []
            for _ in range(min(opening_moves, state.board.num_edges)):
                action = int(opening_rng.choice(state.legal_actions))
                state = state.play(action)
                moves.append(action)
            while not state.terminal:
                index = 0 if state.player == seat else 1
                policy, _ = searches[index].policy(state)
                action = int(move_rng[index].choice(np.flatnonzero(policy == policy.max())))
                agent_moves += index == 0
                state = state.play(action)
                moves.append(action)
            outcomes.append(state.outcome(seat))
            margins.append(state.scores[seat] - state.scores[1 - seat])
            seats.append(seat)
            histories.append(moves)
        row = match_record(size, "checkpoint", outcomes, margins, seats, started, agent_moves)
        row["moves"] = histories
        rows.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "moves"}), flush=True)
    result = {
        "mode": "head_to_head",
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "checkpoint_metadata": metadata,
        "opponent_sha256": opponent_metadata["checkpoint_sha256"],
        "opponent_metadata": opponent_metadata,
        "software": software_versions(checkpoint, opponent_checkpoint),
        "inference": "numpy" if checkpoint.suffix == ".npz" else "torch",
        "opponent_inference": "numpy" if opponent_checkpoint.suffix == ".npz" else "torch",
        "seed": seed,
        "simulations": config.simulations,
        "exact_threshold": config.exact_threshold,
        "search_config": asdict(config),
        "opponent_search_config": asdict(opponent_config),
        "opening_moves": opening_moves,
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
