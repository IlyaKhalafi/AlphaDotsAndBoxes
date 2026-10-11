# Controlled training refinement

Started 2026-10-11. The sequential controller waits for the remaining tree-reuse
comparison before starting GPU training. [Recorded plan and frozen source hashes](data/refinement-training-20261011/plan.json).

All three arms start from the same released 96-channel, six-layer model. Board
mix, seed, search settings and batch size match. None enables tree reuse or exact
leaf solving; both are evaluated separately in [the search report](search-refinement.md).

| Arm | Learner updates per round | Batch size | Midgame starts |
| --- | --- | --- | --- |
| Control | 16 | 8,192 | 0% |
| Lower update ratio | 4 | 8,192 | 0% |
| Midgame archive | 4 | 8,192 | 50% |

Each arm collects 128 games per round with four GPU samplers, 512 search
simulations and a root-only 12-edge solver. Each stops after 64 rounds or two
hours, saving completed rounds. The campaign therefore allocates at most six
configured training hours; startup, round completion and CPU evaluation add
wall-clock time. The 40 GiB learner allocator cap leaves room for the unrelated
job using about 67 GiB of the shared H200, samplers and CUDA contexts.

Comparison uses common saved rounds across arms, including their last common
checkpoint, rather than comparing unequal training-round counts after time
limits. Selection uses seed 3160 and 20 games per board/budget for 4×4 and 5×5
at 128 and 512 simulations. One selected candidate advances to fresh seeds 3170
and 3171, each providing 40 games per condition. Both players have identical
search settings and balanced seats, with six randomized opening moves.

Deployment requires at least 50% pooled score on every board/budget, strictly
above 50% on 5×5 Deep, and move latency within twice the released model's.
Passing this limited practical regression gate is not statistical proof or a
claim of expert-human strength. Inference comparisons use a separate NumPy-only
installation with Cython extensions absent. A passing candidate must also pass
Torch/NumPy parity and the deployment API smoke test before replacement.

Runs log policy cross-entropy, value loss, target and network entropy, policy
mismatch, value absolute error, midgame-start counts, replay sample
presentations, sampling/update time and peak GPU allocation. Losses describe
the final replay minibatch of each round. Checkpoints and resumable learner
states are saved throughout; the supervisor handles bounded recovery from
memory exhaustion. Completed arms publish numerical receipts and loss plots.

**Status: queued behind the final search comparison; no training-strength
result yet.** The live model on port 9003 remains the previously released model.
