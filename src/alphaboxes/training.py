"""Search-guided self-play, Ray sampling, replay, and RLlib LearnerGroup updates."""

import json
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import ray
import torch
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner_group import LearnerGroup
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, MultiAgentBatch, SampleBatch

from alphaboxes.checkpoint import load_agent, save_agent
from alphaboxes.game import board
from alphaboxes.graph import encode_batch
from alphaboxes.learner import AlphaZeroLearner
from alphaboxes.network import module_spec
from alphaboxes.search import MCTS, NeuralEvaluator, SearchConfig
from alphaboxes.selfplay import Example, SelfPlayWorker, play_episode


@dataclass(frozen=True)
class TrainConfig:
    sizes: tuple[tuple[int, int], ...] = ((1, 2), (2, 2), (2, 3), (3, 3))
    iterations: int = 100
    games_per_iteration: int = 16
    workers: int = 4
    selfplay_device: str = "cpu"
    selfplay_batch_size: int = 1
    updates_per_iteration: int = 16
    batch_size: int = 128
    replay_capacity: int = 20_000
    learning_rate: float = 3e-4
    width: int = 96
    depth: int = 6
    seed: int = 42
    temperature_moves: int = 12
    device: str = "cpu"
    gpu_memory_gb: float = 4.0
    gpu_duty_cycle: float = 0.15
    checkpoint_every: int = 10
    max_seconds: float | None = None
    search: SearchConfig = field(default_factory=lambda: SearchConfig(simulations=64))

    def __post_init__(self):
        for name in (
            "iterations",
            "games_per_iteration",
            "updates_per_iteration",
            "batch_size",
            "replay_capacity",
            "checkpoint_every",
            "selfplay_batch_size",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.workers < 0 or self.workers > self.games_per_iteration:
            raise ValueError("workers must be between zero and games_per_iteration.")
        if not self.sizes or any(board(*size).num_boxes > 144 for size in self.sizes):
            raise ValueError("Training sizes must contain boards of at most 144 boxes.")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda.")
        if self.selfplay_device not in ("cpu", "cuda"):
            raise ValueError("selfplay_device must be cpu or cuda.")
        if self.selfplay_device == "cuda" and (self.device != "cuda" or not self.workers):
            raise ValueError("CUDA self-play requires a CUDA learner and Ray workers.")
        if not 0 < self.gpu_duty_cycle <= 1 or self.gpu_memory_gb <= 0:
            raise ValueError("GPU memory and duty-cycle limits must be positive.")
        if self.max_seconds is not None and (
            not np.isfinite(self.max_seconds) or self.max_seconds <= 0
        ):
            raise ValueError("max_seconds must be finite and positive.")

    @classmethod
    def from_json(cls, path: Path) -> "TrainConfig":
        config = json.loads(path.read_text())
        config["sizes"] = tuple(tuple(size) for size in config.get("sizes", cls().sizes))
        config["search"] = SearchConfig(**config.get("search", {}))
        return cls(**config)


def make_batch(examples: list[Example]) -> MultiAgentBatch:
    capacity = max(example.state.board.num_nodes for example in examples)
    observations = encode_batch([example.state for example in examples], capacity)
    policies = np.zeros((len(examples), capacity), dtype=np.float32)
    for i, example in enumerate(examples):
        policies[i, : len(example.policy)] = example.policy
    batch = SampleBatch(
        {
            Columns.OBS: observations,
            "policy_target": policies,
            "value_target": np.array([example.value for example in examples], dtype=np.float32),
        }
    )
    return MultiAgentBatch({DEFAULT_POLICY_ID: batch}, len(examples))


def build_learner(config: TrainConfig) -> LearnerGroup:
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable. Use --device cpu.")
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        fraction = min(config.gpu_memory_gb * 1024**3 / total_bytes, 1.0)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
    capacity = max(board(*size).num_nodes for size in config.sizes)
    spec = MultiRLModuleSpec(
        rl_module_specs={
            DEFAULT_POLICY_ID: module_spec(capacity, config.width, config.depth),
        }
    )
    learner_config = (
        AlgorithmConfig()
        .framework("torch")
        .learners(
            num_learners=0,
            num_gpus_per_learner=0.1 if config.device == "cuda" else 0,
            learner_class=AlphaZeroLearner,
        )
        .training(lr=config.learning_rate, grad_clip=1.0, grad_clip_by="global_norm")
        .rl_module(rl_module_spec=spec)
    )
    return LearnerGroup(config=learner_config, module_spec=spec)


def metric_values(results: list[dict]) -> dict[str, float]:
    metrics = {}
    for result in results:
        for key, value in result.get(DEFAULT_POLICY_ID, {}).items():
            if hasattr(value, "peek"):
                value = value.peek()
            if isinstance(value, (float, int, np.number)) and np.isfinite(value):
                metrics[key] = float(value)
    return metrics


def train(
    config: TrainConfig,
    output: Path,
    resume: Path | None = None,
    initial_checkpoint: Path | None = None,
) -> Path:
    if resume and initial_checkpoint:
        raise ValueError("Choose resume or an initial checkpoint, not both.")
    initial_model, initial_metadata = (
        load_agent(initial_checkpoint) if initial_checkpoint else (None, {})
    )
    if initial_model and any(
        initial_model.model_config[key] != getattr(config, key) for key in ("width", "depth")
    ):
        raise ValueError(
            "Initial checkpoint width and depth must match the training configuration."
        )
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    random.seed(config.seed)
    learner = build_learner(config)
    initialization = {}
    if initial_model:
        learner.set_weights({DEFAULT_POLICY_ID: initial_model.get_state()})
        initialization = {
            "checkpoint_sha256": initial_metadata["checkpoint_sha256"],
            "training_games": initial_metadata.get("games_total", 0),
            "training_positions": initial_metadata.get("positions_total", 0),
        }
    replay: deque[Example] = deque(maxlen=config.replay_capacity)
    start_iteration, games_total, positions_total = 0, 0, 0
    elapsed_before_resume = 0.0
    config_path = output / "config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2) + "\n")
    if resume:
        # A resumable state is trusted local training data, unlike an inference checkpoint.
        state = torch.load(resume, map_location="cpu", weights_only=False)
        old_config = state["config"]
        for key in ("width", "depth", "sizes"):
            if old_config[key] != asdict(config)[key]:
                raise ValueError(f"Cannot resume with a different {key}.")
        learner.set_state(state["learner"])
        initialization = state.get("initialization", {})
        replay.extend(state["replay"])
        rng.bit_generator.state = state["rng"]
        start_iteration = state["iteration"]
        games_total, positions_total = state["games_total"], state["positions_total"]
        if resume.resolve().parent == output.resolve():
            # Continuing a run extends its total budget; a new output starts a new clock.
            elapsed_before_resume = state.get("elapsed_seconds", 0.0)
            metrics_path = output / "metrics.jsonl"
            if metrics_path.exists():
                previous = [json.loads(line) for line in metrics_path.read_text().splitlines()]
                if previous and previous[-1]["iteration"] > start_iteration:
                    raise ValueError(
                        "Resume checkpoint is older than run metrics. Use a new output directory."
                    )
                if previous and "elapsed_seconds" not in state:
                    elapsed_before_resume = previous[-1]["elapsed_seconds"]
                if state.get("last_metric") and (
                    not previous or previous[-1]["iteration"] < start_iteration
                ):
                    with metrics_path.open("a") as log:
                        log.write(json.dumps(state["last_metric"]) + "\n")
    workers = []
    owns_ray = False
    local_search = None
    if config.workers:
        if not ray.is_initialized():
            # Always a new local instance. Never attach to someone else's Ray cluster.
            ray.init(
                address="local",
                num_cpus=config.workers,
                num_gpus=1 if config.selfplay_device == "cuda" else 0,
                include_dashboard=False,
                object_store_memory=256 * 1024**2,
                log_to_driver=False,
            )
            owns_ray = True
        workers = [
            SelfPlayWorker.options(num_gpus=0.01 if config.selfplay_device == "cuda" else 0).remote(
                config.width,
                config.depth,
                config.search,
                config.seed + 1000 * (i + 1) + start_iteration,
                config.selfplay_device,
                config.selfplay_batch_size,
            )
            for i in range(config.workers)
        ]
    else:
        module = module_spec(width=config.width, depth=config.depth).build()
        local_search = MCTS(NeuralEvaluator(module), config.search, config.seed)
    started = time.monotonic()
    try:
        with (output / "metrics.jsonl").open("a") as log:
            for iteration in range(start_iteration + 1, config.iterations + 1):
                iteration_start = time.monotonic()
                weights = learner.get_weights()[DEFAULT_POLICY_ID]
                if workers:
                    ref = ray.put(weights)
                    tasks = [
                        worker.collect.remote(
                            ref,
                            list(config.sizes),
                            config.games_per_iteration // len(workers)
                            + (i < config.games_per_iteration % len(workers)),
                            config.temperature_moves,
                        )
                        for i, worker in enumerate(workers)
                    ]
                    examples = [example for chunk in ray.get(tasks) for example in chunk]
                else:
                    local_search.evaluator.module.set_state(weights)
                    local_search.evaluator.clear_cache()
                    examples = []
                    for _ in range(config.games_per_iteration):
                        size = config.sizes[int(rng.integers(len(config.sizes)))]
                        examples.extend(
                            play_episode(local_search, size, rng, config.temperature_moves)
                        )
                sample_seconds = time.monotonic() - iteration_start
                replay.extend(examples)
                games_total += config.games_per_iteration
                positions_total += len(examples)
                update_start = time.monotonic()
                metrics = {}
                for _ in range(config.updates_per_iteration):
                    indices = rng.choice(
                        len(replay), min(config.batch_size, len(replay)), replace=False
                    )
                    batch = make_batch([replay[int(index)] for index in indices])
                    result = learner.update(batch=batch)
                    metrics = metric_values(result)
                if config.device == "cuda":
                    torch.cuda.synchronize()
                    del batch
                    torch.cuda.empty_cache()
                update_seconds = time.monotonic() - update_start
                if config.device == "cuda":
                    # Pace GPU work; no exclusive mode, reset, or changes to other processes.
                    cooldown = update_seconds * (1 / config.gpu_duty_cycle - 1) - sample_seconds
                    if cooldown > 0:
                        time.sleep(cooldown)
                metadata = {
                    "iteration": iteration,
                    "games_total": games_total,
                    "positions_total": positions_total,
                    "seed": config.seed,
                    "sizes": [list(size) for size in config.sizes],
                    "simulations": config.search.simulations,
                    "selfplay_device": config.selfplay_device,
                    "selfplay_batch_size": config.selfplay_batch_size,
                    "training_exact_threshold": config.search.exact_threshold,
                    "initialization": initialization,
                }
                budget_exhausted = (
                    config.max_seconds is not None
                    and elapsed_before_resume + time.monotonic() - started >= config.max_seconds
                )
                record = (
                    metadata
                    | metrics
                    | {
                        "replay_positions": len(replay),
                        "sample_seconds": sample_seconds,
                        "update_seconds": update_seconds,
                        "iteration_seconds": time.monotonic() - iteration_start,
                        "elapsed_seconds": elapsed_before_resume + time.monotonic() - started,
                        "stop_reason": "time_budget"
                        if budget_exhausted
                        else "iterations"
                        if iteration == config.iterations
                        else None,
                        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2
                        if config.device == "cuda"
                        else 0,
                    }
                )
                weights = learner.get_weights()[DEFAULT_POLICY_ID]
                save_agent(output / "latest.pt", weights, config.width, config.depth, metadata)
                if (
                    iteration % config.checkpoint_every == 0
                    or iteration == config.iterations
                    or budget_exhausted
                ):
                    save_agent(
                        output / f"agent-{iteration:05d}.pt",
                        weights,
                        config.width,
                        config.depth,
                        metadata,
                    )
                state = {
                    "config": asdict(config),
                    "learner": learner.get_state(),
                    "replay": list(replay),
                    "rng": rng.bit_generator.state,
                    "iteration": iteration,
                    "games_total": games_total,
                    "positions_total": positions_total,
                    "initialization": initialization,
                    "elapsed_seconds": record["elapsed_seconds"],
                    "last_metric": record,
                }
                temporary = output / "resume.tmp"
                torch.save(state, temporary)
                os.replace(temporary, output / "resume.pt")
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if budget_exhausted:
                    break
    finally:
        for worker in workers:
            ray.kill(worker)
        learner.shutdown()
        if owns_ray:
            ray.shutdown()
    return output / "latest.pt"
