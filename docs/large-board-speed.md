# Large-board speed work

2026-10-13. Keep the confirmed 8,192-simulation large-board search; reduce its
cost rather than shortening Deep thinking. The proposed two-second UI reset
was withdrawn before deployment. The live service retains the longer profile.

## Profile and code changes

A cached-selector profile on one six-move 10×10 opening takes 12.41 seconds
for 2,048 simulations under instrumentation. Neural evaluation accounts for
9.04 seconds. This identifies inference as the main remaining bottleneck;
instrumented times are not normal response measurements.

The NumPy optimization keeps the same weights, six width-96 GIN layers, and
search settings. It sums at most four ordered incidence neighbors instead of
multiplying a dense mostly-zero adjacency matrix. The policy head projects the
shared global features once, then computes logits only for legal edges. SiLU,
linear bias addition and layer normalization reuse intermediate arrays. Small
boards retain dense aggregation because gathering neighbors costs more there.
The sparse path starts at 256 graph nodes; training observations remain unchanged.

Alternating baseline/optimized calls on identical legal positions, with one
BLAS thread and 40 retained samples after five warmups, give these medians:

| Board | Baseline inference | Optimized inference | Reduction |
| --- | --- | --- | --- |
| 4×4 opening | 1.358 ms | 1.312 ms | 3.3% |
| 5×5 opening | 1.743 ms | 1.679 ms | 3.7% |
| 10×10 opening | 6.047 ms | 4.767 ms | 21.2% |
| 12×12 opening | 10.296 ms | 7.697 ms | 25.2% |

One cold 10×10 decision at the full 8,192 simulations drops from **45.52 to
36.11 seconds** (20.7%). Both choose edge 43; no simulation budget is cut.
The baseline runs first, so this single sequential measurement is preliminary,
not a general latency guarantee. The alternating inference probe covers openings,
middle positions and 13-edge positions. Maximum policy difference is 1.42e−7
and maximum value difference 5.96e−8 on those probes. Sparse aggregation and
policy projection change floating-point summation order, so complete search
traces need not be bit-identical. Random-network parity tests also cover
rectangles, padding, structured features, and late positions.

This is a useful reduction, but it does not make a 36-second move suitable for
ordinary interactive play. More BLAS threads are also being measured; their
overhead can outweigh the benefit on these small matrix multiplications.

## Distillation experiment

Compress the retained network to **width 48, depth 6**, retaining its message
passing depth. Freeze teacher predictions on 3,735 distinct positions from
training replay iteration 3, seed 75: 644 on 4×4, 691 on 5×5, and 800 each on
8×8, 10×10 and 12×12. Dataset seed is 51001. Human games and evaluation histories
are not training inputs; targets are teacher predictions, not oracle labels.

Use the existing RLlib learner for 1,500 updates, batches of 256, learning rate
0.0003, seed 51002, an 8 GiB allocation cap and 50% update pacing. Select by
held-out teacher-policy KL plus value MSE every 100 steps. Identical evaluator
keys cannot cross the board-stratified validation split; nearby positions may
still come from the same game. Validation imitation is not playing strength.

After training, measure NumPy inference and full-search cost. Screen the frozen
selected student on 12 fresh paired 10×10 games, seed 511001, against retained
weights with 8,192 simulations on both sides and the same cached selector,
constant first-play reduction and 12-edge root aid. No timers, pruning, proofs,
rollouts, structural solving, reuse or noise. A score above 50% advances to
40 fresh games, seed 512001, requiring score above 50% and paired-bootstrap
lower bound above 50%. Smaller-board and actual deployment checks remain
necessary before replacing weights. The existing model stays live while these
checks run. A faster student that loses strength is not a successful upgrade.

## Completed training and further probes

Distillation completes all 1,500 updates in 599.33 seconds. Peak PyTorch
allocation is 1,397.97 MiB, below its private 8 GiB cap. Selection chooses step
1,400: held-out policy KL 0.01715, value MSE 0.01810 and policy-maximum agreement
71.31%. The student has 69,512 parameters, versus 272,648 in the retained model.
Its uncached 10×10 opening inference takes 2.411 ms versus the optimized
retained model's 4.790 ms; on 12×12, 3.189 versus 7.692 ms. These are inference
measurements, not evidence of retained playing strength.

The selected student is frozen at SHA-256
`91a6e8a7a3b5540294e63fcc2a456882cf7fd3651e1e66f7227095c8fa1d5c85`.
Its completed 12-game full-budget screen finishes **5 wins and 7 losses**
(41.67%, paired-bootstrap interval 16.67–66.67%, mean box margin −12.83).
It fails the predeclared screen; confirmation and smaller-board promotion
checks are not run. The retained model stays live. The student's mean neural
decision is 12.61 seconds versus 17.12 for the reference, but positions differ
between the players, so this is not a paired latency estimate. Both complete
8,192 simulations on every neural decision. The
[screen](data/large-model-20261012/student-48x6-screen.json),
[summary](data/large-model-20261012/student-48x6-screen-summary.json), and
[gate receipt](data/large-model-20261012/student-48x6-gates.json) record the
rejection. The [training receipt](data/large-model-20261012/student-48x6-training-receipt.json),
configuration, portable frozen targets, experimental weights, per-update metrics
and [evaluation source fingerprint](data/large-model-20261012/speed-source-snapshot.json)
preserve this stage for reproduction.

Three further code hypotheses were tested separately:

- Two BLAS threads gave 5.24–5.61 ms on the sampled 10×10 positions, versus
  about 4.77 ms with one. Four threads gave 4.19–5.15 ms; results vary by position
  and do not establish a consistent gain. Keep one thread.
- A Cython scalar SiLU prototype was roughly four to six times slower than
  NumPy's vectorized version. Its normalization speed was mixed. Do not add
  this optional compiled path to deployment.
- Folding identical local GIN inputs reduced an empty 10×10 graph's inference
  from 4.640 to 2.355 ms, but gave little benefit in middle positions. Extending
  this to a context-keyed cache passed 60-position parity checks, but a complete
  2,048-simulation search slowed from 9.44 to 10.85 seconds. Lookup and cache
  maintenance outweighed the saved computation. The cross-position cache is
  rejected; the within-position folding remains a separate timing experiment.

The [alternating inference timings](data/large-model-20261012/numpy-optimization-paired-latency.json),
[full-work decision](data/large-model-20261012/numpy-optimization-deep-latency.json),
and [NumPy-only API probe](data/large-model-20261012/numpy-speed-web-probe.json)
record the code optimization. A fresh empty-board API request still takes
47.02 seconds while completing all 8,192 simulations; workload and position
differ from the six-move 36.11-second probe. This has not solved interactive
latency. Port 9003 now uses the optimized retained network; its weights and
Deep budget are unchanged. The full suite passes 277 tests, and another
42 relevant tests pass after replacing NumPy scalar square roots with native
Python scalar arithmetic. The scalar change preserves complete visit/value
traces in constant, policy-mass and branching selection tests.

## Reuse more exact network predictions

The long-search worker now retains 32,768 complete policy/value predictions
instead of 4,096. Ordinary requests retain their existing 4,096-entry cache.
This is different from the rejected local-node cache: it uses the existing
validated whole-position key and returns the exact previously computed network
output. It does not share visits or change the search strategy. Memory remains
bounded by the entry ceiling; on a 10×10 board, policy-array storage at capacity
is approximately 27.5 MiB, in addition to dictionary and key overhead.

In a [three-move continuation probe](data/large-model-20261012/prediction-cache-probe.json),
both cache sizes complete 8,192 simulations on every move and return identical
actions and values. On the second move, uncached neural predictions fall from
7,109 to 1,947; on the third, from 7,124 to 1,944. Corresponding measured decision
times fall from 41.85 to 17.86 seconds and from 36.98 to 18.21 seconds. The cold
first decisions take 41.49 and 35.79 seconds; they run sequentially and differ
in concurrent load, so do not attribute that entire cold difference to cache
capacity. The useful result is retaining predictions across continuations.

These are consecutive searched moves, not a complete human-match latency
distribution. A human choosing a different continuation can reduce reuse, and
the cold first move remains expensive. Tests compare full policies and values
through repeated real searches under forced eviction, verify fewer predictions
with a larger cache, and verify clearing. API tests confirm that only the long
worker gets the larger cache and both workers retain their original deadlines.

The [live continuation check](data/large-model-20261012/prediction-cache-live-probe.json)
completes 8,192 simulations in each of three fresh 10×10 moves: **54.43,
19.81 and 18.46 seconds**. These figures include HTTP time while full-game
benchmarking and tests run concurrently. The expensive cold opening remains
unresolved. Ordinary 4×4 and 5×5 requests still finish below two seconds.
The clean installed sources match the released files, and no training frameworks
are imported. The final complete suite passes **284 tests**.

Search-time feature encoding also avoids building dense adjacency just to count
adjacent boxes with one free side. Cached two-box incidence indices now provide
those features with linear memory. Dense adjacency remains available for training
observations. Bit-for-bit feature checks cover padding and a 20×30 board; the
37 relevant graph, network and API tests pass after this change.

An additional [coverage diagnostic](data/large-model-20261012/student-48x6-coverage.json)
finds 145 of the student's 800 training positions on 10×10 have more than 75%
of edges free; 385 are middle positions and 270 are late. Thus opening positions
are present, although coverage alone does not establish adequate imitation.
On a fresh empty 10×10 board the teacher value is 0.05106 and the student value
0.01472, a difference of 0.03634; the policy KL is 0.00970. Value errors can alter
search allocation, so faster inference and small average imitation losses still
require a full-game comparison. The completed screen above rejects this student;
no model promotion follows from this
diagnostic. Benchmark CPU priority is lowered to favor interactive serving;
weights, opening seeds and fixed simulation counts are unchanged.

## Further runtime optimization

The next release keeps the same checkpoint and budgets. For large unpadded
boards, message passing now uses rectangular array views: horizontal edges,
vertical edges, and boxes. This removes four whole-graph indexed gathers per
layer while preserving the ordered sums. A zero addition on edge messages also
preserves signed-zero behavior. Padded observations retain the gather path;
small-board inference retains dense aggregation. Activations and residuals
reuse arrays owned by the current call, and all-one padding masks skip redundant
work. Weights and observations remain unmodified; message buffers are local to
each call.

Search selection now stores one stable sorted child table and integer end
offsets for equal-prior runs. This avoids separate lists and objects for each
prior, retains the shortcut across equal priors, and scores visited children
without a property or function call. Rounded-score ties still choose original
dictionary order, and cancelled simulations retain their unvisited scores.
Delaying table construction alone was discarded after profiling showed most
tables were used again; avoiding the group objects gives a more useful saving.

The comparison uses retained checkpoint `21d8df7a…`, reference source at
`6725ebb`, and optimized runtime at `46a8eb0`. NumPy 2.5.3 runs with one BLAS
thread and no compiled search extension. The
[full-budget receipt](data/large-model-20261012/grid-runtime-full-budget.json)
alternates player order across two repeats, starting from the same six-move
10×10 position with seed 61001. Each repeat includes three consecutive moves
with a 32,768-entry prediction cache; every decision completes 8,192 simulations.
Every policy, value, chosen action, and fingerprint of the complete visited
tree matches exactly.

| 10×10 decision | Reference median | Optimized median | Reduction |
| --- | --- | --- | --- |
| Cold first move | 38.51 s | 30.49 s | 20.8% |
| Second move | 29.33 s | 25.20 s | 14.1% |
| Third move | 15.07 s | 11.81 s | 21.7% |

Alternating inference probes retain 60 samples after five warmups, including
empty, six-move, and middle positions. Opening medians are:

| Board | Reference inference | Optimized inference | Reduction |
| --- | --- | --- | --- |
| 5×5 | 1.622 ms | 1.527 ms | 5.9% |
| 10×10 | 5.167 ms | 4.352 ms | 15.8% |
| 12×12 | 7.376 ms | 6.032 ms | 18.2% |

All these probes return identical logits and values. Additional
[board checks](data/large-model-20261012/grid-runtime-board-checks.json) cover
5×5 and 12×12 search at 512 simulations, again with exact visited-tree parity.
Their fixed-work comparisons enable the cached selector on both sides; ordinary
small-board UI requests keep their existing selection setting and time budget.
These are measurements on a shared machine, with two search repeats and a small
set of positions. They establish a useful code speedup, not a latency guarantee
or a new playing-strength claim. The cold large-board decision remains too slow
for a two-second response.

A [selector-only comparison](data/large-model-20261012/grid-selector-isolation.json)
uses identical optimized network evaluators on both sides, 512 simulations,
and the same three-move sequence. Second-move medians fall from 0.285 to 0.182
seconds; third-move medians from 0.289 to 0.180 seconds. These small-budget
cache-heavy measurements isolate table overhead; their percentage gain does not
describe a cold 8,192-simulation decision.

The [live API receipt](data/large-model-20261012/grid-runtime-live-probe.json)
checks the installed package on port 9003, confirms matching runtime sources and
unchanged checkpoint bytes, and verifies that PyTorch, Ray and Gymnasium are
absent. Ordinary 5×5 Deep requests take 1.764–1.770 seconds and complete
940, 1,182 and 1,463 simulations in the three measured moves. Faster inference
allows more search within the existing time cap; it does not shorten that cap.
The receipt distinguishes ordinary timed API requests from extended Deep.
On a fresh empty 10×10 board, the extended Deep HTTP requests take **40.34,
14.92 and 14.93 seconds**, completing all 8,192 simulations with no deadline
fallbacks. CPU tests run concurrently. This empty-board result differs from the
six-move opening used by the paired reference comparison above.

Reproduce a reference comparison from a trusted checkout of `6725ebb`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ALPHABOXES_PURE_NUMPY=1 \
PYTHONPATH=src python scripts/benchmark_runtime.py \
  --reference-source /path/to/reference/src --simulations 8192 \
  --sizes 10x10 --search-repeats 2 --inference-repeats 60 \
  --output runtime-comparison.json
```

The script aborts if inference or any visited search trace differs; its output
includes individual timing samples, source fingerprints and search settings.
The final complete suite passes **306 tests**, and Ruff and whitespace checks
pass. The runtime remains a clean NumPy installation with the original trained
weights and the extended Deep option enabled.

Further [C++ runtime acceleration](compiled-runtime.md) preserves the same
model and search traces while reducing the first two full-budget 10×10 decisions
by 24–25% in the next paired comparison. It is optional; ordinary installation
continues to use NumPy.
