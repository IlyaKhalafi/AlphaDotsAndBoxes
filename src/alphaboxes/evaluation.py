"""Seat-balanced evaluation with raw outcomes and uncertainty intervals."""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from alphaboxes.checkpoint import load_agent
from alphaboxes.game import State
from alphaboxes.opponents import random_action, solve, tactical_action
from alphaboxes.search import MCTS, NeuralEvaluator, SearchConfig


def wilson(wins: int, games: int) -> tuple[float, float]:
    z = 1.96
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    radius = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games**2)) / denominator
    return max(0, center - radius), min(1, center + radius)


def evaluate(
    checkpoint: Path,
    output: Path,
    sizes: list[tuple[int, int]],
    games: int = 40,
    simulations: int = 128,
    exact_threshold: int = 0,
    seed: int = 2026,
) -> dict:
    if games < 2 or games % 2:
        raise ValueError("Use an even number of games >= 2 for balanced seats.")
    torch.set_num_threads(1)
    module, metadata = load_agent(checkpoint)
    neural = NeuralEvaluator(module)
    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        opponents = {"random": random_action, "tactical": tactical_action}
        if State.new(*size).board.num_edges <= 12:
            opponents["exact"] = lambda state, rng: int(rng.choice(solve(state)[1]))
        for opponent_name, opponent in opponents.items():
            started = time.monotonic()
            outcomes, scores, seats = [], [], []
            actions_total = 0
            for game in range(games):
                agent_seat = game % 2
                state = State.new(*size)
                search = MCTS(
                    neural,
                    SearchConfig(simulations=simulations, exact_threshold=exact_threshold),
                    seed + game,
                )
                while not state.terminal:
                    if state.player == agent_seat:
                        policy, _ = search.policy(state)
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
            wins, draws, losses = outcomes.count(1), outcomes.count(0), outcomes.count(-1)
            record = {
                "size": list(size),
                "opponent": opponent_name,
                "games": games,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "win_rate": wins / games,
                "score_rate": (wins + draws / 2) / games,
                "win_rate_95ci": list(wilson(wins, games)),
                "mean_box_margin": float(np.mean(scores)),
                "seconds": time.monotonic() - started,
                "agent_moves": actions_total,
                "outcomes": outcomes,
                "box_margins": scores,
                "agent_seats": seats,
            }
            rows.append(record)
            print(json.dumps(record), flush=True)
    result = {
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_metadata": metadata,
        "seed": seed,
        "simulations": simulations,
        "exact_threshold": exact_threshold,
        "results": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
