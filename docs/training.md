# Training and evaluation

## Install

Create a Python 3.11+ virtual environment and install:

```bash
python -m pip install -e '.[dev]'
```

For a CUDA machine, install the appropriate PyTorch wheel for its driver before
installing the project. Training defaults to CPU; CUDA is explicitly selected
with `--device cuda`.

RLlib is pinned to 2.58.0 because the module/learner APIs evolve. The tested
environment and versions are recorded alongside the experiment results.

## Presets

| Preset | Purpose | Self-play search | CPU workers |
| --- | --- | --- | --- |
| `smoke.json` | Two tiny iterations; verify updates and saving | 8 simulations | 0 |
| `bootstrap.json` | 1,280 games on 1×2, 2×2, 2×3, 3×3 | 32 simulations | 4 |
| `refine.json` | Resume bootstrap through iteration 160 | 64 simulations + exact last 12 edges | 4 |
| `strong.json` | Longer mixed-size experiment, through 5×5 | 128 simulations | 8 |

The strong preset is a proposed experiment, not a run claimed in the results.
It is not an automatic guarantee of expert-level strength.

```bash
adb train --config configs/bootstrap.json --output runs/bootstrap --device cuda
adb train --config configs/refine.json --output runs/refined \
  --resume runs/bootstrap/resume.pt --device cuda
```

The refinement preset changes the search budget and enables a solved endgame,
while retaining the network, optimizer, and replay. No human games or human move
labels enter either run. Refinement results must be identified as solver-guided
self-play, even when the solver is disabled at evaluation time.

## Shared hardware

The code never resets the GPU, changes compute mode, or terminates other jobs.
Only the learner uses CUDA. Each Ray worker reserves one CPU and uses one
PyTorch inference thread. A new private local Ray instance is started if none
exists; the driver does not attach to an existing remote cluster.

`gpu_memory_gb` caps the **PyTorch allocator** at 4 GiB by default (or half the
device, whichever is smaller). CUDA context/driver allocations are additional.
`gpu_duty_cycle` defaults to 0.15: the driver synchronizes learner work and paces
iterations so update time is at most that fraction of sampling/update/cooldown
time. This is application pacing, not a hardware-enforced compute quota and
does not guarantee zero contention with another job. Use CPU if GPU latency is
critical to another workload. Smaller batches and fewer workers reduce resource
demand further. No global GPU settings are modified.

Inspect usage before launching a long run:

```bash
nvidia-smi
```

## Outputs and resuming

Every run writes:

| File | Contents |
| --- | --- |
| `config.json` | Resolved settings |
| `metrics.jsonl` | Per-iteration sample time, update time, losses, counts, allocation peak |
| `latest.pt` | Latest inference weights and metadata; written atomically |
| `agent-NNNNN.pt` | Periodic inference snapshots |
| `resume.pt` | RLlib learner/optimizer state, replay, driver RNG, iteration and counts |

```bash
adb train --config configs/bootstrap.json --iterations 120 \
  --output runs/bootstrap --resume runs/bootstrap/resume.pt --device cuda
```

`--iterations` is the final iteration number, not additional iterations. The
network dimensions and board-size set must match the resume state. Search
budgets, training length, and batch settings may change. Restarting workers
creates new worker RNG streams, so resume is not bitwise equivalent to an
uninterrupted run. Resume files contain pickled local replay objects: only load
files you trust. Inference checkpoints use PyTorch's restricted weights loader.

## Evaluate fairly

```bash
adb evaluate --checkpoint models/bootstrap.pt --sizes 2x2,3x3,4x4 \
  --games 40 --simulations 128 --seed 2026 --output runs/pure.json

adb evaluate --checkpoint models/bootstrap.pt --sizes 2x2,3x3,4x4 \
  --games 40 --simulations 128 --exact-threshold 12 \
  --seed 2026 --output runs/aided.json
```

Use an even game count so both seats receive equal coverage. Raw records include
outcomes, seats, box margins, checkpoint hash, and settings. Win-rate intervals
use the Wilson formula with z=1.96; draws count as half a point only in the
separately reported `score_rate`. Intervals are descriptive for this seeded
sample, not evidence of strength against other opponents or humans.

Baselines:

- **Random:** uniformly sample a legal edge.
- **Tactical:** capture the most immediate boxes; otherwise avoid giving a box's
  third side when possible; randomly choose among ties.
- **Exact:** full minimax on boards with at most 12 edges.
- **Tactical endgame:** tactical opening, exact minimax with at most 12 edges
  remaining. It is stronger than the tactical baseline, but not a validated
  expert-human substitute.

For a serious strength claim, evaluate multiple training seeds, unseen sizes and
rectangles, independent chain-aware opponents, and a balanced series of matches
against experienced human players. Report fixed search budgets and solver use.
Keep a held-out evaluation set for checkpoint selection and a separate final
test set. These broader experiments remain future work.
