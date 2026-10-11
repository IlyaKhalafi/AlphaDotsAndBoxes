import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from alphaboxes.game import State
from alphaboxes.numpy_network import weight_shapes


def test_parallel_pairs_match_serial_oracle_games_and_preserve_progress(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    receipts = []
    for workers in (1, 2):
        output = tmp_path / f"workers-{workers}.json"
        progress = tmp_path / f"workers-{workers}.jsonl"
        command = [
            sys.executable,
            str(root / "scripts/benchmark_engine.py"),
            "--checkpoint",
            str(root / "models/larger.npz"),
            "--match-boards",
            "2",
            "--games",
            "4",
            "--match-seconds",
            "0.05",
            "--workers",
            str(workers),
            "--skip-latency",
            "--output",
            str(output),
            "--progress",
            str(progress),
        ]
        subprocess.run(
            command, cwd=root, env=environment, check=True, capture_output=True, timeout=30
        )
        receipt = json.loads(output.read_text())
        receipts.append(receipt)
        assert receipt["workers"] == workers and receipt["latency"] == []
        assert [game["index"] for game in receipt["games"]] == list(range(4))
        records = [json.loads(line) for line in progress.read_text().splitlines()]
        assert [record["completed_games"] for record in records] == [2, 4]
        for first, second in zip(receipt["games"][::2], receipt["games"][1::2], strict=True):
            assert first["opening"] == second["opening"]
            assert (first["alpha_player"], second["alpha_player"]) == (0, 1)
        for game in receipt["games"]:
            state = State.new(*game["board"])
            for action in game["opening"]:
                state = state.play(action)
            for move in game["moves"]:
                assert move["player"] == state.player
                assert move["remaining_edges"] == len(state.legal_actions)
                state = state.play(move["action"])
            assert state.terminal and list(state.scores) == game["scores"]
        before = progress.read_bytes()
        refused = subprocess.run(
            command, cwd=root, env=environment, capture_output=True, timeout=30
        )
        assert refused.returncode != 0 and progress.read_bytes() == before
    # Every post-opening state has <= 6 free edges, so both engines use the
    # exact root solver. Scheduling and elapsed-time limits cannot change play.
    for serial, parallel in zip(receipts[0]["games"], receipts[1]["games"], strict=True):
        assert serial["scores"] == parallel["scores"]
        assert [move["action"] for move in serial["moves"]] == [
            move["action"] for move in parallel["moves"]
        ]


def test_distinct_checkpoints_with_same_search_algorithm(tmp_path):
    root = Path(__file__).resolve().parents[1]
    candidate = tmp_path / "small.npz"
    payload = {
        "format_version": 1,
        "model_config": {"width": 8, "depth": 1},
        "metadata": {},
    }
    weights = {key: np.zeros(shape, np.float32) for key, shape in weight_shapes(8, 1).items()}
    np.savez_compressed(
        candidate, _metadata=np.frombuffer(json.dumps(payload).encode(), np.uint8), **weights
    )
    output = tmp_path / "same-engine.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/benchmark_engine.py"),
            "--checkpoint",
            str(candidate),
            "--reference-checkpoint",
            str(root / "models/larger.npz"),
            "--reference-engine",
            "alpha_beta",
            "--match-boards",
            "2",
            "--games",
            "2",
            "--match-seconds",
            "0.05",
            "--skip-latency",
            "--output",
            str(output),
        ],
        cwd=root,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
        check=True,
        capture_output=True,
        timeout=30,
    )
    receipt = json.loads(output.read_text())
    assert receipt["checkpoint_sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert receipt["reference_checkpoint_sha256"] != receipt["checkpoint_sha256"]
    assert receipt["reference_engine"] == "alpha_beta"
    for game in receipt["games"]:
        for move in game["moves"]:
            assert move["engine"] == "alpha_beta"
            assert (move["side"] == "candidate") == (move["player"] == game["alpha_player"])
