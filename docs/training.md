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
| `deep.json`      | Eight-hour warm start; emphasize 5×5           | 512 simulations + exact last 12 edges | 12        |
| `wide.json`      | Eight-hour refinement with 288 hidden channels | 512 simulations + exact last 12 edges | 4 GPU samplers |
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

### Expand a trained model

The wide preset has 2,421,512 parameters: 288 channels and six residual GIN
blocks, compared with 272,648 parameters at 96 channels. First widen the
recommended checkpoint, then start a fresh optimizer and replay buffer:

```bash
adb widen --checkpoint models/larger.pt --output runs/wide/initial.pt --factor 3
python scripts/supervise_training.py --config configs/wide.json --output runs/wide \
  --initial-checkpoint runs/wide/initial.pt
```

Widening duplicates hidden channels and splits outgoing weights. Zero-sum
outgoing perturbations let the copies learn independently while retaining the
original policy and value at initialization, up to floating-point rounding.
Global readouts and LayerNorm expand consistently. Depth stays unchanged.
Only weights transfer; source hashes, dimensions, noise and seed are recorded.

Four GPU samplers each run 32 concurrent games with captured inference. The
learner uses 12,288-position batches and a 72 GiB allocator cap; CUDA contexts
and the samplers consume additional memory. Reduce these settings for smaller
GPUs. The deployment export still uses ordinary NumPy. A larger model costs
more per searched move and must pass fresh comparisons before replacing the
playable checkpoint. [Expansion measurements](wide-run.md) record the checks.

The deep preset uses 512 self-play simulations and batches of 40,960 positions
with a 72 GiB learner allocator cap and 50% learner-update pacing. Fourteen workers collect 128
games per iteration; the learner performs 16 updates, with a 500,000-position
replay buffer. This preset targets a large-memory GPU: a full 5×5 batch used
65.4 GiB of allocated memory in a disposable learner probe. Reduce the batch
size and memory cap for smaller hardware. The repeated 5×5 entry
is intentional: workers sample that board twice as often as each other size.
It retains the same network dimensions so selected weights transfer directly.

```bash
adb train --config configs/deep.json --output runs/deep \
  --initial-checkpoint models/larger.pt --device cuda
```

The larger batch increases actual GPU allocation; raising the cap alone would
not do so. Tree traversal runs on CPU workers; two workers batch neural calls
on the GPU. Check the other workload and free
CPU capacity before using the larger preset on a shared machine.

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
Samplers use CPU by default; the deep and wide presets also run neural inference
on CUDA. Each Ray worker reserves one CPU and uses one
PyTorch inference thread. A new private local Ray instance is started if none
exists; the driver does not attach to an existing remote cluster.

`gpu_memory_gb` caps the learner's **PyTorch allocator** at 4 GiB by default,
with an explicit larger cap in GPU presets. CUDA context/driver allocations
are additional; each GPU sampler separately caps its allocator at 1 GiB.
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

The full resume state is saved atomically after every completed iteration.
Numbered inference snapshots follow `checkpoint_every`. If interrupted between
the full checkpoint and its metrics append, resuming restores that missing row.

For long shared-GPU runs, use the bounded supervisor:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/supervise_training.py \
  --config configs/deep.json --output runs/deep \
  --initial-checkpoint models/larger.pt --device cuda
```

It enables expandable CUDA allocations before PyTorch starts, writes progress to
`supervision.json`, and resumes from the full checkpoint after a memory failure
or stalled iteration. Memory failures halve the batch size while preserving the
learner GPU memory cap. Recovery is limited to three restarts and retains the original
remaining time budget. Other training errors stop with their log path recorded.
Only the supervisor's own child processes are stopped during recovery. Replace
`--initial-checkpoint` with `--resume runs/deep/resume.pt` to continue an existing
run. Training also releases unused CUDA cache between iterations.

Self-play defaults to single-game CPU workers. Set `selfplay_batch_size` to
advance independent games together and combine their neural predictions. Set
`selfplay_device` to `cuda` to run those predictions on the GPU; this requires
`--device cuda` and nonzero workers. Each GPU sampler has a separate 1 GiB
PyTorch allocator cap. Its Ray GPU fraction is an admission-control hint,
not a VRAM allocation or hardware partition. Include sampler contexts and the
other job's allocation when choosing the learner's `gpu_memory_gb` cap.

Search creates states only for visited moves. Single-state graph encoding avoids
training-batch setup work, while the training encoder remains vectorized.
Dedicated evaluation copies avoid repeatedly changing every layer's mode.
All three changes preserve the game rules and search budget.

The deep preset mixes twelve CPU workers (eight games each) with two GPU workers
(sixteen games each). `selfplay_gpu_workers` chooses how many workers use CUDA;
`games_per_worker` assigns their game counts explicitly and must sum to
`games_per_iteration`. Without these fields, the configured device and balanced
assignment apply to every worker. This mixed workload uses 14 of this VM's 18
visible CPU cores for sampling and leaves room for the driver and the other job.
GPU workers use one captured network call (`selfplay_cuda_graphs`) over fixed
padded inputs, reuse it for partial batches and reload weights in place.
Capture remains local to each worker; deployment continues to use NumPy.
The duty-cycle setting paces learner updates only, including sampling time in
its cooldown calculation; it does not throttle GPU sampling or partition the GPU.

Build the optional exact-endgame accelerator locally:

```bash
pip install -e '.[accelerate]'
python scripts/build_acceleration.py
```

The build also compiles child scoring in search, removing Python callbacks and
temporary arithmetic objects from the inner loop. The Cython solver uses masks
over at most 18 **remaining** edges, so it supports
boards whose total edge count exceeds 64. It preserves double captures and extra
turns. Ordinary installation uses the Python solver and requires neither Cython
nor a compiler; the generated binary is excluded from version control.

Measure search and endgames separately on fixed positions:

```bash
python scripts/benchmark_search.py --checkpoint models/larger.pt \
  --device cuda --batch 8 --output runs/search-benchmark.json
python scripts/benchmark_search.py --mode oracle --output runs/oracle-benchmark.json
```

Compare identical positions, checkpoints, simulation counts and thread limits.
Microbenchmarks do not establish end-to-end training speed or playing strength.

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
last iteration may exceed the budget. Resuming into the same output directory
includes the time already recorded in that run, so changing three hours to
eight hours grants a total of eight hours. Resuming into a new output directory
starts a new timer. The driver saves an inference snapshot
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

## Controlled refinement options

`start_fraction` defaults to zero. Set it to, for example, `0.5` to start half
the games from reachable midgame positions collected by that sampler. Each board
shape has a bounded archive (`start_archive_capacity`, default 4,096 states).
`start_min_moves` and `start_max_moves` select the number of edges already drawn,
with defaults 12 and 28. A cold archive uses a random legal prefix. Finished
positions are excluded; small boards that cannot meet the minimum start empty.
Resumed workers rebuild their initial archives from saved replay positions.

Targets use the complete final score, including boxes claimed before the start,
and the player active at each recorded position. Temperature applies to the
first sampled moves after that start. Keep some empty-start games to retain
opening practice, and test this option separately before making it a default.

Metrics include `midgame_starts`, `mean_initial_moves`, `sample_presentations`
and `presentations_per_new_position`. The last metric is the number of examples
presented to the learner in this round divided by new positions collected in
the round; it is not a count of how often every individual position was used.
Vary `updates_per_iteration` while holding batch size and other settings fixed
to compare replay update ratios.

`target_entropy` measures how spread out the search targets are; `policy_kl`
subtracts that entropy from policy cross-entropy to measure their mismatch with
the network. A flat policy loss can hide a changing mismatch when target entropy
changes. `policy_entropy` measures the network distribution and `value_mae`
measures absolute outcome-prediction error. These are final-minibatch metrics on
replay data, not held-out scores; use fresh matches to assess playing strength.

Search experiments are also opt-in. `--leaf-exact-threshold 12` solves small
search leaves; `--reuse-tree` retains visited descendants across moves. A duel
can give the opponent separate values through `--opponent-leaf-exact-threshold`
and `--opponent-reuse-tree`. Both players retain the configured fresh simulation
budget per move. Receipts record move times and inherited visits. Changing
network weights requires clearing both evaluator predictions and the search
tree; self-play workers do this at the beginning of each collection round.

The controlled presets `configs/refine-control.json`, `configs/refine-ratio.json`
and `configs/refine-midgame.json` share the same starting-model shape, board mix,
batch size and search settings. The ratio preset changes 16 updates to four;
the midgame preset then changes only start sampling. Each limits training to
64 rounds or two hours, saving after completed rounds, whichever comes first.
Use the same immutable initial checkpoint for all arms and compare equal-round
snapshots if their time limits produce different round counts. A 40 GiB learner
allocator cap reserves room for the four samplers and other shared-GPU work;
CUDA context allocations are additional. Adjust this cap to local free memory.

```bash
python scripts/supervise_training.py --config configs/refine-control.json \
  --output runs/control --initial-checkpoint models/larger.pt
python scripts/supervise_training.py --config configs/refine-ratio.json \
  --output runs/ratio --initial-checkpoint models/larger.pt
python scripts/supervise_training.py --config configs/refine-midgame.json \
  --output runs/midgame --initial-checkpoint models/larger.pt
python scripts/summarize_duels.py runs/matches/*.json --output runs/comparison.json
```

Only pass matching duel receipts to the summary command. It rejects duplicate
seeds or mixed model/search treatments and weights latency by actual move counts.
