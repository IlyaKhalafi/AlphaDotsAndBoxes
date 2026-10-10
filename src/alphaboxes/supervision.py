"""Bounded training with durable status and limited CUDA-memory recovery."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def read_metric(output):
    path = output / "metrics.jsonl"
    if path.exists():
        for line in reversed(path.read_text().splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def save_status(output, status):
    temporary = output / "supervision.tmp"
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(output / "supervision.json")


def stop_owned_child(child):
    if child.poll() is not None:
        return
    child.send_signal(signal.SIGINT)
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


def training_command(config, output, resume, initial):
    command = [sys.executable, "-m", "alphaboxes.cli", "train", "--config", str(config)]
    command.extend(["--output", str(output)])
    if resume:
        command.extend(["--resume", str(resume)])
    elif initial:
        command.extend(["--initial-checkpoint", str(initial)])
    return command


def supervise(
    config_path,
    output,
    resume=None,
    initial=None,
    used_seconds=None,
    max_restarts=3,
    stall_seconds=1200,
    poll_seconds=5,
    command_factory=training_command,
    device=None,
):
    config = json.loads(config_path.read_text())
    if device is not None:
        config["device"] = device
    budget = config.get("max_seconds")
    if budget is None or budget <= 0:
        raise ValueError("Supervised training requires a positive max_seconds budget.")
    output.mkdir(parents=True, exist_ok=True)
    used = read_metric(output).get("elapsed_seconds", 0) if used_seconds is None else used_seconds
    if used < 0 or used >= budget:
        raise ValueError("Used training time must be below the total budget.")
    deadline = time.monotonic() + budget - used
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    status = {"supervisor_pid": os.getpid(), "events": [], "allocator": env["PYTORCH_ALLOC_CONF"]}
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    for attempt in range(max_restarts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        if attempt:
            used = read_metric(output).get("elapsed_seconds", used)
        attempt_config = dict(config, max_seconds=used + remaining)
        config_file = output / f"supervised-{stamp}-{attempt}.json"
        config_file.write_text(json.dumps(attempt_config, indent=2) + "\n")
        log_path = config_file.with_suffix(".log")
        with log_path.open("w") as log:
            child = subprocess.Popen(
                command_factory(config_file, output, resume, initial),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            status.update(
                status="running",
                child_pid=child.pid,
                attempt=attempt,
                batch_size=config["batch_size"],
                log=str(log_path),
                remaining_seconds=remaining,
            )
            save_status(output, status)
            last_iteration = read_metric(output).get("iteration", 0)
            progress_at = time.monotonic()
            stalled = False
            try:
                while child.poll() is None:
                    time.sleep(poll_seconds)
                    metric = read_metric(output)
                    if metric.get("iteration", 0) != last_iteration:
                        last_iteration = metric["iteration"]
                        progress_at = time.monotonic()
                    status.update(
                        last_iteration=last_iteration,
                        remaining_seconds=max(0, deadline - time.monotonic()),
                    )
                    save_status(output, status)
                    if time.monotonic() - progress_at > stall_seconds:
                        stalled = True
                        stop_owned_child(child)
                        break
            finally:
                stop_owned_child(child)
        metric = read_metric(output)
        if child.returncode == 0 and metric.get("stop_reason"):
            status.update(status="complete", last_iteration=metric["iteration"])
            save_status(output, status)
            return status
        oom = "CUDA out of memory" in log_path.read_text()
        status["events"].append(
            {
                "attempt": attempt,
                "returncode": child.returncode,
                "reason": "stalled" if stalled else "cuda_oom" if oom else "training_failed",
                "log": str(log_path),
            }
        )
        resume = output / "resume.pt"
        if not (oom or stalled) or not resume.exists() or attempt == max_restarts:
            break
        if oom:
            config["batch_size"] = max(1, config["batch_size"] // 2)
    status["status"] = "failed"
    save_status(output, status)
    raise RuntimeError("Training stopped; inspect supervision.json and its attempt logs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", type=Path)
    source.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--used-seconds", type=float)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    supervise(
        args.config,
        args.output,
        args.resume,
        args.initial_checkpoint,
        args.used_seconds,
        device=args.device,
    )


if __name__ == "__main__":
    main()
