# Component-aware GIN refinement

## Why change the encoding

The retained six-block GIN gives effectively the same value to two reachable
14×14 positions with opposite exact outcomes. The changes lie in central loops,
away from border information available to its local message passing. Full
prefixes and values are in the
[representation diagnostic](data/large-search-20261012/representation-diagnostic.json).
This example identifies missing information; it is separate from 10×10 matches.

The new optional encoding adds four box-node features: component size in
four-box units, closed-chain status, closed-loop status, and capture fronts.
Fourteen global features describe chain/loop box totals and counts, three-box
chains, four/six-box loops, short chains, active/branching components, parity,
current score margin and remaining boxes. Counts use fixed box units across
board sizes. The graph still uses shared edge/box weights and no coordinates.

Legacy checkpoints use 12 node features and three context features. Structured
checkpoints use 16 and 17 respectively. Both retain width 96 and six residual
GIN blocks. Parameter counts are 272,648 and 277,064. The initializer copies
existing weights into their corresponding columns and zeros new columns;
prediction-preservation tests cover 1×1 through 14×14 and NumPy export.

## Training experiment

Two runs use RLlib's LearnerGroup and the same backbone, initial predictions,
seed 75, board curriculum, sampling/search settings and update budgets:

- `component-large.json` enables component features and starts from expanded
  retained weights. Initial checkpoint SHA-256:
  `5c7bed891c467ffcb86bcbe5a046b3097b617b914b9e062520be323ca23ec0b2`.
- `data-large.json` retains the original encoding and starts from `larger.pt`.
  This control tests whether larger-board data alone accounts for improvement.

Each run has an eight-hour maximum, three CUDA samplers with 16 games per batch,
48 games per iteration, 256 new simulations per nonexact move, tree reuse and
the exact structural oracle. Board sizes are 4×4, 5×5, 8×8, 10×10 (twice in the
sampling list) and 12×12. A quarter of starts are legally generated chain/loop
endgames; another optional archive draw uses 10% of the remaining starts.
Every endgame start is reached by legal play, including current scores.
Neither run uses the experimental root-shortlist mode.

Updates use batches of up to 2,048 positions, 16 updates per iteration, learning
rate 0.0001 and replay capacity 300,000. The matching configuration and seed
control budgets; trajectories can diverge after learning. Logs record policy
loss, value loss, value MAE, entropy/KL, starts, sampling/update time and GPU
peak allocation. Checkpoints are saved every five iterations.

## Shared resources and inference

Each learner has a 24 GiB allocator limit. Each CUDA sampler retains its
existing 1 GiB limit. Two learners plus six samplers therefore have aggregate
allocator ceilings of 54 GiB. Before launch, the H200 reported 72,895 MiB free;
the other GPU workload remains running. Training uses low CPU scheduling
priority and private Ray session directories. Only owned clusters/processes
are managed.

A single RLlib update with 2,048 12×12 positions used 19,514.6 MiB peak allocation
and 20,514 MiB reserved, completing in 1.183 seconds. This probe is a memory and
correctness check, not a throughput or strength benchmark.
[Probe receipt](data/large-model-20261012/gpu-probe.json).

CUDA self-play keeps padded adjacency matrices on device and changes them only
when a batch slot changes board shape. Ordinary deployment still requires NumPy;
PyTorch, Ray/RLlib and CUDA remain training dependencies.

## Selection protocol

Checkpoint screens use 12 paired-seat 10×10 games at 512 simulations against the
released constant-0.33 NumPy search, on disjoint opening ranges starting at
10001 and increasing by 1000 per screening treatment. No oracle, proof, reuse,
noise or timer is enabled in these matches: the comparison changes the learned
model and its encoding. Screen results only select a candidate.

Before confirmation, freeze the checkpoint and settings. A fresh 40-game match
at 1,024 simulations against released search must score above 50%, with the
lower 2.5th percentile of the 20,000-resample paired-opening bootstrap (seed
4199) above 50%. Confirmation ranges start at 200001, with a 1000-seed increment
for a new attempt. A separate 20-game control against the older zero-first-play
search uses ranges starting at 300001 and requires at least 50% score. Full
replays and actual work are checked. An optimistic completion may reject a
mathematically futile run; stopped receipts must identify planned and retained
counts. Smaller-board behavior and actual web settings are checked before
deployment. No larger-board improvement is claimed at launch.

The first launches stopped before their first update: reuse attempted to promote
a value-only solved leaf with no action list. Search now rebuilds such nodes as
fresh roots when the root oracle cannot finish an action solve. A regression
test reproduces the timed-out root solve. Settings are unchanged; restarted runs
use `runs/large-model-20261012/component-v2` and `data-control-v2`.
