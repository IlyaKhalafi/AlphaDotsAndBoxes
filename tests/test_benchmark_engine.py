import json
import os
import subprocess
import sys
from pathlib import Path

from alphaboxes.game import State


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
