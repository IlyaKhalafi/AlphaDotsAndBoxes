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


def play_episodes(
    search: MCTS,
    sizes: list[tuple[int, int]],
    rng: np.random.Generator,
    temperature_moves: int = 12,
) -> list[Example]:
    """Advance independent games together; preserve turns and per-game targets."""
    states = [State.new(*size) for size in sizes]
    histories = [[] for _ in states]
    while active := [i for i, state in enumerate(states) if not state.terminal]:
        predictions = search.policies([states[i] for i in active], explore=True)
        for index, (policy, _) in zip(active, predictions, strict=True):
            history = histories[index]
            history.append((states[index], policy))
            if len(history) <= temperature_moves:
                action = int(rng.choice(len(policy), p=policy / policy.sum()))
            else:
                action = int(rng.choice(np.flatnonzero(policy == policy.max())))
            states[index] = states[index].play(action)
    return [
        Example(position, policy, final.outcome(position.player))
        for final, history in zip(states, histories, strict=True)
        for position, policy in history
    ]


@ray.remote(num_cpus=1, num_gpus=0)
class SelfPlayWorker:
    def __init__(
        self,
        width: int,
        depth: int,
        config: SearchConfig,
        seed: int,
        device: str = "cpu",
        batch_size: int = 1,
        cuda_graph_capacity: int | None = None,
    ):
        torch.set_num_threads(1)
        if device == "cuda":
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(1024**3 / total)
        self.module = module_spec(width=width, depth=depth).build().to(device)
        self.batch_size = batch_size
        self.search = MCTS(
            NeuralEvaluator(
                self.module,
                cuda_batch_size=batch_size if cuda_graph_capacity is not None else None,
                capacity=cuda_graph_capacity,
            ),
            config,
            seed,
        )
        self.rng = np.random.default_rng(seed)

    def collect(
        self, weights: dict, sizes: list[tuple[int, int]], games: int, temperature_moves: int
    ) -> list[Example]:
        self.module.set_state(weights)
        self.search.evaluator.clear_cache()
        samples = []
        for start in range(0, games, self.batch_size):
            count = min(self.batch_size, games - start)
            selected = [sizes[int(self.rng.integers(len(sizes)))] for _ in range(count)]
            if count == 1:
                samples.extend(play_episode(self.search, selected[0], self.rng, temperature_moves))
            else:
                samples.extend(play_episodes(self.search, selected, self.rng, temperature_moves))
        return samples
