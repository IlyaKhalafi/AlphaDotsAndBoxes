import json
import sys

import pytest

from alphaboxes.supervision import supervise


def test_memory_failure_resumes_once_with_smaller_batch(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"max_seconds": 10, "batch_size": 32}))
    output = tmp_path / "run"
    stub = tmp_path / "child.py"
    stub.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "output = Path(sys.argv[2])\n"
        "config = json.loads(Path(sys.argv[1]).read_text())\n"
        "if not (output / 'resume.pt').exists():\n"
        "    (output / 'resume.pt').touch()\n"
        "    print('torch.OutOfMemoryError: CUDA out of memory', flush=True)\n"
        "    sys.exit(1)\n"
        "assert config['batch_size'] == 16\n"
        "assert '--resume' in sys.argv\n"
        "(output / 'metrics.jsonl').write_text(json.dumps({'iteration': 2, "
        "'elapsed_seconds': 1, 'stop_reason': 'time_budget'}) + '\\n')\n"
    )

    def command(config, output, resume, initial):
        return [sys.executable, str(stub), str(config), str(output)] + (
            ["--resume", str(resume)] if resume else []
        )

    status = supervise(config, output, poll_seconds=0.01, command_factory=command)
    assert status["status"] == "complete"
    assert status["attempt"] == 1
    assert status["events"][0]["reason"] == "cuda_oom"
    assert json.loads((output / "supervision.json").read_text()) == status


def test_unrelated_failure_is_not_retried(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"max_seconds": 10, "batch_size": 32}))
    output = tmp_path / "run"

    def command(*args):
        return [sys.executable, "-c", "raise ValueError('invalid configuration')"]

    with pytest.raises(RuntimeError, match="Training stopped"):
        supervise(config, output, poll_seconds=0.01, command_factory=command)
    status = json.loads((output / "supervision.json").read_text())
    assert status["status"] == "failed"
    assert status["attempt"] == 0
    assert len(status["events"]) == 1
