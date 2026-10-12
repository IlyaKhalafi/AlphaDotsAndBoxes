import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

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
        if workers == 1:
            summary_path = tmp_path / "summary.json"
            summarize = [
                sys.executable,
                str(root / "scripts/summarize_engine.py"),
                "--receipts",
                str(output),
                "--output",
                str(summary_path),
            ]
            subprocess.run(
                summarize, cwd=root, env=environment, check=True, capture_output=True, timeout=30
            )
            summary = json.loads(summary_path.read_text())["reports"][0]
            assert summary["receipt_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
            assert summary["results"][0]["games"] == 4
            assert summary["results"][0]["players"]["candidate"]["neural_moves"] == 0
            assert summary["results"][0]["players"]["candidate"]["opening_positions"] == 2
            assert summary["results"][0]["players"]["reference"]["opening_positions"] == 2
            assert summary["results"][0]["players"]["candidate"]["opening_depth_counts"] == {"6": 2}
            damaged = json.loads(output.read_text())
            damaged["games"][0]["scores"][0] += 1
            corrupt = tmp_path / "corrupt.json"
            corrupt.write_text(json.dumps(damaged))
            invalid = subprocess.run(
                [*summarize[:3], str(corrupt), *summarize[4:]],
                cwd=root,
                env=environment,
                capture_output=True,
                timeout=30,
            )
            assert invalid.returncode != 0
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


@pytest.mark.parametrize("candidate_engine", ["alpha_beta", "mcts"])
def test_distinct_checkpoints_and_search_algorithms(tmp_path, candidate_engine):
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
            "--candidate-engine",
            candidate_engine,
            "--response-deadline",
            "1.5",
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
    assert receipt["candidate_engine"] == candidate_engine
    for game in receipt["games"]:
        for move in game["moves"]:
            assert move["engine"] == (
                candidate_engine if move["side"] == "candidate" else "alpha_beta"
            )
            assert (move["side"] == "candidate") == (move["player"] == game["alpha_player"])


def test_untimed_matches_have_equal_work_and_reproducible_parallel_histories(tmp_path):
    root = Path(__file__).resolve().parents[1]
    histories = []
    for workers in (1, 2):
        output = tmp_path / f"untimed-{workers}.json"
        move_logs = tmp_path / f"moves-{workers}"
        command = [
            sys.executable,
            str(root / "scripts/benchmark_engine.py"),
            "--candidate-engine",
            "mcts",
            "--search-options",
            '{"fpu_reduction":0.33}',
            "--match-boards",
            "3",
            "--games",
            "2",
            "--match-simulations",
            "16",
            "--untimed",
            "--skip-latency",
            "--workers",
            str(workers),
            "--move-progress-dir",
            str(move_logs),
            "--output",
            str(output),
        ]
        subprocess.run(
            command,
            cwd=root,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
            capture_output=True,
            check=True,
            timeout=30,
        )
        receipt = json.loads(output.read_text())
        assert receipt["budget_mode"] == "simulations"
        assert receipt["match_seconds"] is None and receipt["reference_seconds"] is None
        games = receipt["games"]
        histories.append([[move["action"] for move in game["moves"]] for game in games])
        moves = [move for game in games for move in game["moves"]]
        assert any(move["remaining_edges"] > 12 for move in moves)
        assert all(move["simulations"] == 16 for move in moves if move["remaining_edges"] > 12)
        for game in games:
            log = move_logs / f"3x3-game-{game['index']:04d}.jsonl"
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            assert rows[0]["opening"] == game["opening"]
            assert [row["action"] for row in rows[1:]] == [move["action"] for move in game["moves"]]
            assert rows[-1]["scores"] == game["scores"]
        refused = subprocess.run(command, cwd=root, capture_output=True, timeout=30)
        assert refused.returncode != 0
    assert histories[0] == histories[1]
