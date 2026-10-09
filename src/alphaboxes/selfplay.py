"""CPU self-play workers. Only the RLlib learner needs a GPU."""

from dataclasses import dataclass

import numpy as np
import ray
import torch

from alphaboxes.game import State
from alphaboxes.network import module_spec
from alphaboxes.search import MCTS, NeuralEvaluator, SearchConfig


@dataclass(frozen=True)
class Example:
    state: State
    policy: np.ndarray
    value: float


def play_episode(
    search: MCTS, size: tuple[int, int], rng: np.random.Generator, temperature_moves: int = 12
) -> list[Example]:
    state = State.new(*size)
    history: list[tuple[State, np.ndarray]] = []
    while not state.terminal:
        policy, _ = search.policy(state, explore=True)
        history.append((state, policy))
        if len(history) <= temperature_moves:
            action = int(rng.choice(len(policy), p=policy / policy.sum()))
        else:
            best = np.flatnonzero(policy == policy.max())
            action = int(rng.choice(best))
        state = state.play(action)
    return [
        Example(position, policy, state.outcome(position.player)) for position, policy in history
    ]


@ray.remote(num_cpus=1, num_gpus=0)
class SelfPlayWorker:
    def __init__(self, width: int, depth: int, config: SearchConfig, seed: int):
        torch.set_num_threads(1)
        self.module = module_spec(width=width, depth=depth).build()
        self.search = MCTS(NeuralEvaluator(self.module), config, seed)
        self.rng = np.random.default_rng(seed)

    def collect(
        self, weights: dict, sizes: list[tuple[int, int]], games: int, temperature_moves: int
    ) -> list[Example]:
        self.module.set_state(weights)
        samples = []
        for _ in range(games):
            size = sizes[int(self.rng.integers(len(sizes)))]
            samples.extend(play_episode(self.search, size, self.rng, temperature_moves))
        return samples
