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

| Preset           | Purpose                                        | Self-play search                     | CPU workers |
| ---------------- | ---------------------------------------------- | ------------------------------------ | ----------- |
| `smoke.json`     | Two tiny iterations; verify updates and saving | 8 simulations                        | 0           |
| `bootstrap.json` | 1,280 games on 1×2, 2×2, 2×3, 3×3              | 32 simulations                       | 4           |
| `refine.json`    | Resume bootstrap through iteration 160         | 64 simulations + exact last 12 edges | 4           |
| `larger.json`    | 80-minute warm start on 3×3, 4×4, 3×5, 5×5     | 64 simulations + exact last 12 edges | 4           |
| `strong.json`    | Longer mixed-size experiment, through 5×5      | 128 simulations                      | 8           |

The strong preset is a proposed experiment, not a run claimed in the results.
It is not an automatic guarantee of expert-level strength.

For larger-board fine-tuning, transfer the released weights into a fresh run:

```bash
adb train --config configs/larger.json --output runs/larger \
  --initial-checkpoint models/agent.pt --device cuda
```

`--initial-checkpoint` transfers only network weights, so the board-size set can
change. Width and depth must match. Optimizer, replay, and run counters start
fresh; checkpoint metadata retains the source hash and earlier training counts.
This differs from `--resume`, which restores the full learner and replay. The
two options are mutually exclusive. Evaluate before replacing a playable model:
fine-tuning can also weaken previously learned play.

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

| File             | Contents                                                                |
| ---------------- | ----------------------------------------------------------------------- |
| `config.json`    | Resolved settings                                                       |
| `metrics.jsonl`  | Per-iteration sample time, update time, losses, counts, allocation peak |
| `latest.pt`      | Latest inference weights and metadata; written atomically               |
| `agent-NNNNN.pt` | Periodic inference snapshots                                            |
| `resume.pt`      | RLlib learner/optimizer state, replay, driver RNG, iteration and counts |

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

`max_seconds` in a configuration, or `--seconds` on the command line, limits the
training loop at completed iteration boundaries. Startup is excluded and the
last iteration may exceed the budget. The driver saves an inference snapshot
and full resume state before stopping, including between ordinary checkpoint
intervals. The larger-board preset budgets 80 minutes for training, leaving
evaluation time within a two-hour experiment.

## Evaluate fairly

```bash
adb evaluate --checkpoint models/agent.pt --sizes 2x2,3x3,4x4 \
  --games 40 --simulations 128 --seed 2026 --output runs/pure.json

adb evaluate --checkpoint models/agent.pt --sizes 2x2,3x3,4x4 \
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
- **Chain control (opt-in):** tactical opening, a reduced chain/loop game that
  can hand back two or four boxes to retain control, and exact minimax for the
  final 12 edges. The abstraction is a heuristic, particularly around short
  chains and unusual interleavings; it is not a perfect-play claim. It supplies
  no training labels.

Select a particular benchmark without changing the original default schedule:

```bash
adb evaluate --checkpoint runs/larger/latest.pt --sizes 5x5 --games 40 \
  --opponents tactical_endgame chain_control
```

Compare checkpoints directly with balanced seats and six randomized opening
moves. The two networks receive the same search budget and independent move RNG
streams; openings are reproducible by game number. Receipts retain complete move
lists for replay and both checkpoint hashes. Endgame assistance defaults to zero.

```bash
adb duel --checkpoint runs/larger/latest.pt --opponent-checkpoint models/agent.pt \
  --sizes 5x5 --games 40 --seed 3031 --output runs/larger-duel.json
```

To evaluate spaced snapshots during training:

```bash
python scripts/tournament.py --run runs/larger --baseline models/agent.pt \
  --output runs/selection --every 40 --games 20 --watch-seconds 5400
```

The tournament defaults to 5×5; use `--sizes 4x4,5x5` to include both boards.
It ranks the mean score against tactical endgame, chain control,
and the released checkpoint. It also evaluates the final timed snapshot and
keeps the baseline on an exact selection-score tie. This is checkpoint selection,
not a final strength claim. Test the selected weights on a fresh seed and verify
smaller-board retention before replacing a playable checkpoint.

Export completed runs and draw a separate chart for a warm-start phase whose
iteration counter begins again at one:

```bash
python scripts/report.py --run runs/bootstrap --run runs/refined --run runs/larger
python scripts/plot_training.py --input docs/data/training-larger.csv \
  --output docs/assets/larger-training.svg --title 'Larger-board refinement'
```

The summary retains the initialization checkpoint hash and its earlier training
counts. Warm-start game and position counters describe the new phase alone.

For a serious strength claim, evaluate multiple training seeds, unseen sizes and
rectangles, independent chain-aware opponents, and a balanced series of matches
against experienced human players. Report fixed search budgets and solver use.
Keep a held-out evaluation set for checkpoint selection and a separate final
test set. These broader experiments remain future work.
