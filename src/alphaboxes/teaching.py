"""Exact late-game teaching with an unchanged network and frozen prediction anchors."""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alphaboxes.checkpoint import load_evaluator
from alphaboxes.game import State
from alphaboxes.opponents import solve


def state_key(state: State) -> tuple:
    return (state.board.rows, state.board.cols, state.edges, state.owners, state.player)


def collect_positions(
    receipts: list[dict],
    seed: int,
    maximum_exact: int = 4096,
    maximum_anchors: int = 4096,
    min_remaining: int = 13,
    max_remaining: int = 16,
) -> tuple[list[State], list[State]]:
    """Replay complete legal games; retain distinct late-game and earlier states."""
    if not 1 <= min_remaining <= max_remaining <= 18:
        raise ValueError("Exact positions require one to 18 remaining edges.")
    if min(maximum_exact, maximum_anchors) < 1:
        raise ValueError("Dataset capacities must be positive.")
    exact, anchors = {}, {}
    for receipt in receipts:
        for row in receipt["results"]:
            for history in row["moves"]:
                state = State.new(*row["size"])
                for action in history:
                    remaining = len(state.legal_actions)
                    if min_remaining <= remaining <= max_remaining:
                        exact[state_key(state)] = state
                    elif remaining > max_remaining:
                        anchors[state_key(state)] = state
                    state = state.play(action)
                if not state.terminal:
                    raise ValueError("Teaching requires complete legal game histories.")
    rng = np.random.default_rng(seed)
    result = []
    for pool, capacity in [(exact, maximum_exact), (anchors, maximum_anchors)]:
        states = list(pool.values())
        if not states:
            raise ValueError("Source games must contain both exact and anchor positions.")
        indices = rng.permutation(len(states))[:capacity]
        result.append([states[int(i)] for i in indices])
    return result[0], result[1]


def oracle_target(state: State) -> tuple[np.ndarray, float]:
    """Best-margin actions also preserve the optimal winning outcome."""
    if state.terminal or len(state.legal_actions) > 18:
        raise ValueError("Teaching oracle requires an unfinished, solvable position.")
    margin, actions = solve(state)
    policy = np.zeros(state.board.num_edges, dtype=np.float32)
    policy[list(actions)] = 1 / len(actions)
    return policy, float(np.sign(margin))


@dataclass
class TeachingDataset:
    states: list[State]
    anchor_policy: np.ndarray
    anchor_value: np.ndarray
    oracle_policy: np.ndarray
    oracle_value: np.ndarray
    exact_count: int
    metadata: dict


def build_dataset(checkpoint: Path, sources: list[Path], output: Path, seed: int = 3310) -> dict:
    if output.exists():
        raise ValueError("Dataset already exists; use a separate output path.")
    snapshots = [path.read_bytes() for path in sources]
    receipts = [json.loads(snapshot) for snapshot in snapshots]
    exact, anchors = collect_positions(receipts, seed)
    states = exact + anchors
    evaluator, model_metadata = load_evaluator(checkpoint)
    count = len(states)
    max_edges = max(s.board.num_edges for s in states)
    max_boxes = max(s.board.num_boxes for s in states)
    edges = np.full((count, max_edges), -1, dtype=np.int8)
    owners = np.full((count, max_boxes), -1, dtype=np.int8)
    anchor_policy = np.zeros((count, max_edges), dtype=np.float32)
    anchor_value = np.zeros(count, dtype=np.float32)
    oracle_policy = np.zeros_like(anchor_policy)
    oracle_value = np.zeros_like(anchor_value)
    started = time.monotonic()
    for i, state in enumerate(states):
        edges[i, : state.board.num_edges] = state.edges
        owners[i, : state.board.num_boxes] = state.owners
        policy, value = evaluator(state)
        anchor_policy[i, : len(policy)] = policy
        anchor_value[i] = value
        if i < len(exact):
            policy, value = oracle_target(state)
            oracle_policy[i, : len(policy)] = policy
            oracle_value[i] = value
    metadata = {
        "format_version": 1,
        "seed": seed,
        "checkpoint_sha256": model_metadata["checkpoint_sha256"],
        "source_checkpoint_sha256": model_metadata.get(
            "source_checkpoint_sha256", model_metadata["checkpoint_sha256"]
        ),
        "sources": [
            {"name": path.name, "sha256": hashlib.sha256(snapshot).hexdigest()}
            for path, snapshot in zip(sources, snapshots, strict=True)
        ],
        "source_seeds": sorted({receipt["seed"] for receipt in receipts}),
        "exact_positions": len(exact),
        "anchor_positions": len(anchors),
        "oracle_outcomes": {
            str(value): int((oracle_value[: len(exact)] == value).sum()) for value in [-1, 0, 1]
        },
        "generation_seconds": time.monotonic() - started,
        "oracle": "Exact final WDL value and uniform best-box-margin policy",
        "anchors": "Frozen released-network policy and value, earlier in source games",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sizes=np.array([(s.board.rows, s.board.cols) for s in states]),
        players=np.array([s.player for s in states]),
        edges=edges,
        owners=owners,
        anchor_policy=anchor_policy,
        anchor_value=anchor_value,
        oracle_policy=oracle_policy,
        oracle_value=oracle_value,
        metadata=np.array(json.dumps(metadata)),
    )
    return metadata


def load_dataset(path: Path) -> TeachingDataset:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"].item()))
        if metadata["format_version"] != 1:
            raise ValueError("Unsupported teaching dataset version.")
        states = []
        for i, size in enumerate(data["sizes"]):
            geometry = State.new(*map(int, size)).board
            states.append(
                State(
                    geometry,
                    tuple(map(int, data["edges"][i, : geometry.num_edges])),
                    tuple(map(int, data["owners"][i, : geometry.num_boxes])),
                    int(data["players"][i]),
                )
            )
        return TeachingDataset(
            states,
            data["anchor_policy"],
            data["anchor_value"],
            data["oracle_policy"],
            data["oracle_value"],
            metadata["exact_positions"],
            metadata,
        )


def teaching_examples(dataset: TeachingDataset, indices, teach_policy: bool):
    from alphaboxes.selfplay import Example

    return [
        Example(
            dataset.states[int(i)],
            (
                dataset.oracle_policy
                if teach_policy and i < dataset.exact_count
                else dataset.anchor_policy
            )[i, : dataset.states[int(i)].board.num_edges],
            float((dataset.oracle_value if i < dataset.exact_count else dataset.anchor_value)[i]),
        )
        for i in indices
    ]


def fine_tune(
    checkpoint: Path,
    dataset_path: Path,
    output: Path,
    teach_policy: bool = False,
    steps: int = 300,
    batch_size: int = 4096,
    exact_fraction: float = 0.25,
    device: str = "cuda",
    seed: int = 3312,
    gpu_memory_gb: float = 28,
) -> Path:
    """Use RLlib updates; keep inference format, architecture and search unchanged."""
    import ray
    import torch
    from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID

    from alphaboxes.checkpoint import load_agent, save_agent
    from alphaboxes.training import TrainConfig, build_learner, make_batch, metric_values

    if steps < 1 or batch_size < 2 or not 0 < exact_fraction < 1:
        raise ValueError("Use positive steps, batch >= 2 and an exact fraction in (0, 1).")
    if output.exists():
        raise ValueError("Training output already exists; use a separate directory.")
    dataset = load_dataset(dataset_path)
    model, initial = load_agent(checkpoint)
    if dataset.metadata["source_checkpoint_sha256"] != initial["checkpoint_sha256"]:
        raise ValueError("Teaching anchors belong to a different starting checkpoint.")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    config = TrainConfig(
        sizes=tuple(sorted({(s.board.rows, s.board.cols) for s in dataset.states})),
        width=model.model_config["width"],
        depth=model.model_config["depth"],
        device=device,
        gpu_memory_gb=gpu_memory_gb,
        workers=0,
        batch_size=batch_size,
        learning_rate=3e-5,
    )
    owns_ray = not ray.is_initialized()
    if owns_ray:
        ray.init(
            address="local",
            num_cpus=2,
            num_gpus=1 if device == "cuda" else 0,
            include_dashboard=False,
            object_store_memory=256 * 1024**2,
            log_to_driver=False,
        )
    try:
        learner = build_learner(config)
    except Exception:
        if owns_ray:
            ray.shutdown()
        raise
    learner.set_weights({DEFAULT_POLICY_ID: model.get_state()})
    rng = np.random.default_rng(seed)
    exact_batch = max(1, min(batch_size - 1, round(batch_size * exact_fraction)))
    output.mkdir(parents=True)
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    settings = {
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "exact_fraction": exact_batch / batch_size,
        "teach_policy": teach_policy,
        "learning_rate": config.learning_rate,
        "gpu_memory_gb": gpu_memory_gb,
        "gpu_duty_cycle": 0.5,
        "dataset_sha256": dataset_hash,
        "dataset": dataset.metadata,
        "initialization": initial,
        "checkpoint_steps": [25, 100, steps],
    }
    (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
    started = time.monotonic()
    try:
        with (output / "metrics.jsonl").open("w") as log:
            for step in range(1, steps + 1):
                tick = time.monotonic()
                indices = np.concatenate(
                    [
                        rng.integers(dataset.exact_count, size=exact_batch),
                        rng.integers(
                            dataset.exact_count, len(dataset.states), size=batch_size - exact_batch
                        ),
                    ]
                )
                batch = make_batch(teaching_examples(dataset, indices, teach_policy))
                metrics = metric_values(learner.update(batch=batch))
                if device == "cuda":
                    torch.cuda.synchronize()
                active = time.monotonic() - tick
                if device == "cuda":
                    time.sleep(active)
                record = {
                    "step": step,
                    **metrics,
                    "update_seconds": active,
                    "elapsed_seconds": time.monotonic() - started,
                    "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1024**2
                    if device == "cuda"
                    else 0,
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                if step in {25, 100, steps}:
                    metadata = {
                        "supervised_updates": step,
                        "teaching": settings,
                        "games_total": 0,
                        "positions_total": len(dataset.states),
                        "initialization": {"checkpoint_sha256": initial["checkpoint_sha256"]},
                    }
                    save_agent(
                        output / f"agent-{step:05d}.pt",
                        learner.get_weights()[DEFAULT_POLICY_ID],
                        config.width,
                        config.depth,
                        metadata,
                    )
                if step % 25 == 0:
                    print(json.dumps(record), flush=True)
    finally:
        learner.shutdown()
        if owns_ray:
            ray.shutdown()
    return output / f"agent-{steps:05d}.pt"
