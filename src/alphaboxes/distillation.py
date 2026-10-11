"""Compress frozen policy/value predictions with the existing RLlib learner."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from alphaboxes.search import CachedEvaluator
from alphaboxes.teaching import TeachingDataset, load_dataset


def split_positions(dataset: TeachingDataset, seed: int, fraction: float = 0.1):
    """Keep identical evaluator keys together within each board size.

    This is an imitation validation split, not an independent playing-strength
    test: nearby positions can still come from the same source game.
    """
    if not 0 < fraction < 1:
        raise ValueError("Holdout fraction must be in (0, 1).")
    groups = {}
    for index, state in enumerate(dataset.states):
        board_groups = groups.setdefault((state.board.rows, state.board.cols), {})
        board_groups.setdefault(CachedEvaluator._key(state), []).append(index)
    rng = np.random.default_rng(seed)
    training, validation = [], []
    for board_groups in groups.values():
        equivalent = list(board_groups.values())
        if len(equivalent) < 2:
            raise ValueError("Each board size needs at least two distinct evaluator positions.")
        shuffled = rng.permutation(len(equivalent))
        count = min(len(equivalent) - 1, max(1, round(len(equivalent) * fraction)))
        validation.extend(index for group in shuffled[:count] for index in equivalent[group])
        training.extend(index for group in shuffled[count:] for index in equivalent[group])
    return np.array(training), np.array(validation)


def frozen_examples(dataset: TeachingDataset, indices):
    """Use teacher predictions even where separate exact labels are available."""
    from alphaboxes.selfplay import Example

    return [
        Example(
            dataset.states[i],
            dataset.anchor_policy[i, : dataset.states[i].board.num_edges],
            float(dataset.anchor_value[i]),
        )
        for i in indices
    ]


def imitation_metrics(module, batch, indices, chunk_size: int = 128) -> dict[str, float]:
    import torch
    from ray.rllib.core.columns import Columns

    totals = dict(policy_kl=0.0, value_mse=0.0, value_mae=0.0, policy_agreement=0.0)
    for offset in range(0, len(indices), chunk_size):
        selected = indices[offset : offset + chunk_size]
        observations = {
            key: torch.from_numpy(value[selected]) for key, value in batch[Columns.OBS].items()
        }
        prediction = module.predict(observations)
        log_policy = prediction[Columns.ACTION_DIST_INPUTS].log_softmax(-1).numpy()
        targets = batch["policy_target"][selected]
        kl = np.where(targets > 0, targets * (np.log(np.maximum(targets, 1e-30)) - log_policy), 0)
        delta = prediction[Columns.VF_PREDS].numpy() - batch["value_target"][selected]
        totals["policy_kl"] += float(kl.sum())
        totals["value_mse"] += float(np.square(delta).sum())
        totals["value_mae"] += float(np.abs(delta).sum())
        totals["policy_agreement"] += int((log_policy.argmax(-1) == targets.argmax(-1)).sum())
    return {key: value / len(indices) for key, value in totals.items()}


def distill(
    teacher: Path,
    dataset_path: Path,
    output: Path,
    width: int = 32,
    depth: int = 6,
    steps: int = 1500,
    batch_size: int = 1024,
    validate_every: int = 100,
    device: str = "cuda",
    seed: int = 3601,
    gpu_memory_gb: float = 8,
) -> Path:
    """Select by held-out teacher KL + value MSE; never by match results."""
    import ray
    import torch
    from ray.rllib.core.columns import Columns
    from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, MultiAgentBatch, SampleBatch

    from alphaboxes.checkpoint import save_agent
    from alphaboxes.network import module_spec
    from alphaboxes.numpy_network import export_agent, load_network
    from alphaboxes.training import TrainConfig, build_learner, make_batch, metric_values

    if steps < 1 or batch_size < 2 or validate_every < 1:
        raise ValueError("Use positive steps and validation interval, and batch size >= 2.")
    if output.exists():
        raise ValueError("Training output already exists; use a separate directory.")
    dataset = load_dataset(dataset_path)
    teacher_network, teacher_metadata = load_network(teacher)
    if dataset.metadata["checkpoint_sha256"] != teacher_metadata["checkpoint_sha256"]:
        raise ValueError("Frozen targets belong to a different teacher checkpoint.")
    config = TrainConfig(
        sizes=tuple(sorted({(s.board.rows, s.board.cols) for s in dataset.states})),
        width=width,
        depth=depth,
        device=device,
        gpu_memory_gb=gpu_memory_gb,
        workers=0,
        batch_size=batch_size,
        gpu_duty_cycle=0.5,
        seed=seed,
    )
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    training, validation = split_positions(dataset, seed)
    # Encode immutable observations once instead of rebuilding the same graphs
    # for every minibatch. Only selected arrays are sent to the learner.
    encoded = make_batch(frozen_examples(dataset, range(len(dataset.states))))
    batch = encoded.policy_batches[DEFAULT_POLICY_ID]
    validator = module_spec(width=width, depth=depth).build().eval()
    settings = {
        "seed": seed,
        "width": width,
        "depth": depth,
        "teacher_width": teacher_network.width,
        "teacher_depth": teacher_network.depth,
        "teacher_sha256": teacher_metadata["checkpoint_sha256"],
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "dataset": dataset.metadata,
        "training_indices": training.tolist(),
        "validation_indices": validation.tolist(),
        "split_mode": "Board-stratified groups of identical evaluator keys",
        "steps": steps,
        "batch_size": batch_size,
        "validate_every": validate_every,
        "learning_rate": config.learning_rate,
        "gpu_memory_gb": gpu_memory_gb,
        "gpu_duty_cycle": config.gpu_duty_cycle,
        "targets": "Frozen teacher policy and value only; no oracle targets",
        "selection": "Minimum validation policy KL + value MSE",
    }
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
    learner = None
    rng = np.random.default_rng(seed)
    best, selected_step = float("inf"), None
    started = time.monotonic()
    try:
        learner = build_learner(config)
        output.mkdir(parents=True)
        (output / "config.json").write_text(json.dumps(settings, indent=2) + "\n")
        with (output / "metrics.jsonl").open("w") as log:
            for step in range(1, steps + 1):
                tick = time.monotonic()
                indices = rng.choice(training, size=batch_size, replace=True)
                sampled = SampleBatch(
                    {
                        Columns.OBS: {
                            key: value[indices] for key, value in batch[Columns.OBS].items()
                        },
                        "policy_target": batch["policy_target"][indices],
                        "value_target": batch["value_target"][indices],
                    }
                )
                metrics = metric_values(
                    learner.update(batch=MultiAgentBatch({DEFAULT_POLICY_ID: sampled}, batch_size))
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                active = time.monotonic() - tick
                if device == "cuda":
                    time.sleep(active * (1 / config.gpu_duty_cycle - 1))
                record = {"step": step, **metrics, "update_seconds": active}
                if step == 1 or step % validate_every == 0 or step == steps:
                    weights = learner.get_weights()[DEFAULT_POLICY_ID]
                    validator.load_state_dict(
                        {key: torch.as_tensor(value) for key, value in weights.items()}
                    )
                    heldout = imitation_metrics(validator, batch, validation)
                    record["validation"] = heldout
                    score = heldout["policy_kl"] + heldout["value_mse"]
                    if np.isfinite(score) and score < best:
                        best, selected_step = score, step
                        save_agent(
                            output / "best.pt",
                            weights,
                            width,
                            depth,
                            {
                                "distillation": settings,
                                "supervised_updates": step,
                                "validation": heldout,
                            },
                        )
                    print(json.dumps(record), flush=True)
                record["elapsed_seconds"] = time.monotonic() - started
                record["gpu_peak_mb"] = (
                    torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
                )
                log.write(json.dumps(record) + "\n")
                log.flush()
        if selected_step is None:
            raise RuntimeError("No checkpoint had finite validation metrics.")
        exported = export_agent(output / "best.pt", output / "best.npz")
        (output / "selection.json").write_text(
            json.dumps(
                {
                    "selected_step": selected_step,
                    "validation_score": best,
                    "elapsed_seconds": time.monotonic() - started,
                    "parameters": sum(parameter.numel() for parameter in validator.parameters()),
                    "checkpoint_sha256": exported["checkpoint_sha256"],
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        if learner is not None:
            learner.shutdown()
        if owns_ray:
            ray.shutdown()
    return output / "best.npz"
