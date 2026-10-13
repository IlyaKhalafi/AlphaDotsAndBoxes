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
