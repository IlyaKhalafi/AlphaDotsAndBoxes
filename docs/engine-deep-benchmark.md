# Alpha–beta against the retained agent at Deep

The retained GIN with graph search beat the experimental alpha–beta engine in
this fresh 40-game benchmark. Both players had a two-second thinking budget.
Graph search remains the default on port 9003.

## Reference and protocol

The reference is `models/larger.npz`, the best retained checkpoint across the
4×4 and 5×5 release checks. Later wider and teaching candidates failed the
overall release gate; none has established a stronger all-board replacement.
The [previous results](next-progress.md) also show that some candidates improve
one board while regressing another. This reference choice does not establish
a universal ranking of every checkpoint.

Both engines use the same 96-channel, six-layer, 272,648-parameter GIN, with
checkpoint SHA-256
`21d8df7a60ba770e45a35d8405e009417904e250813fb3134403fb7707ae87b0`.
The experiment therefore compares the search algorithms while holding weights
fixed. MCTS has the deployed 8,192-traversal ceiling in addition to its deadline.
Alpha–beta retains its configured depth ceiling of 64.

The [plan](data/engine-deep-20261011/plan.json) was recorded before matches.
There are 20 games per board: ten six-move random openings, each played twice
with seats swapped. Opening seeds are 3521–3530, fresh relative to the earlier
short-budget screens. Each player has a separate prediction cache, cleared
before each game, and a fresh planner per move. Both receive the same exact
12-edge root endgame aid. Tree reuse and exact leaf solving are disabled.

Inference, tree selection, and the root solver use the ordinary installed
Python/NumPy runtime, with no optional Cython extensions, PyTorch, Ray, or GPU.
Eight match workers each use one BLAS/OMP/MKL thread. Wall-clock cutoffs depend
on shared CPU load. Ties select the lowest-index maximum in this benchmark;
the web service randomizes tied maxima. Budgets limit thinking rather than
guaranteeing request latency.

## Results

Scores are from alpha–beta's perspective. A draw contributes half a point.

| Board | Games | Alpha–beta wins / draws / losses | Alpha–beta score | Mean final box margin |
| --- | ---: | ---: | ---: | ---: |
| 4×4 | 20 | 4 / 2 / 14 | 25% | −3.8 |
| 5×5 | 20 | 2 / 0 / 18 | 10% | −8.2 |

The reference won 32 games, drew two, and lost six across both boards. This is a
limited comparison against the retained agent, not an expert-human strength or
10×10 strength claim. Each board has only ten independent opening pairs.

## Search behavior

These counters include only moves with more than 12 edges available, excluding
the exact root solver. Depth counts edges drawn, including moves that keep the
same player active. Traversals and alpha–beta depths measure different work.

| Board | Median completed alpha–beta depth | Median MCTS traversals | Mean thinking time, alpha–beta / MCTS |
| --- | ---: | ---: | ---: |
| 4×4 | 3 | 4,271 | 1.988 / 1.767 s |
| 5×5 | 3 | 2,534 | 1.998 / 1.852 s |

Alpha–beta's completed depth ranged from 2–14 on 4×4 and 1–13 on 5×5. It never
fell back to an unsearched prior in these games. The loss therefore persists
after granting enough time to finish several search layers. Depth alone is not
a reliable measure of playing strength with an approximate learned evaluator.
The longest recorded move took 2.059 seconds, consistent with the soft deadline.

## Post-match exact checks

All 40 histories were replayed to validate legal actions, extra turns, seats,
remaining-edge counts, and final scores. Afterwards, an independent exact
minimax check evaluated every observed move with 13–16 edges remaining.
The existing Cython solver accelerated this analysis; its results were not
available to either player during the matches.

An outcome error changes the best achievable win/draw/loss result. A margin
error reduces the best achievable final score difference, even when the
win/draw/loss result stays the same.

| Board | Engine | Checked moves | Outcome errors | Margin errors |
| --- | --- | ---: | ---: | ---: |
| 4×4 | Alpha–beta | 33 | 0 | 1 |
| 4×4 | MCTS | 47 | 1 | 3 |
| 5×5 | Alpha–beta | 31 | 1 | 2 |
| 5×5 | MCTS | 49 | 0 | 1 |

By these late positions, alpha–beta was already in a forced-loss state on
22 of its 33 checked 4×4 moves and 26 of its 31 checked 5×5 moves. The observed
deficit therefore points toward earlier opening and middlegame decisions.
These checks cover different visited positions, so they do not isolate the
cause or establish that one evaluator is better on a common position suite.
Faster inference and evaluation calibrated for minimax remain hypotheses to
test; the current evidence does not support replacing graph search.

## Receipts and reproduction

The run took about 6 minutes 45 seconds with eight CPU workers.
[Complete games, timings, hashes, and runtime settings](data/engine-deep-20261011/matches.json)
and the [validated summary and exact checks](data/engine-deep-20261011/summary.json)
preserve the measurements. The parallel runner was tested against serial play
on exact-solved positions, including progress preservation and seat pairing.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/benchmark_engine.py --checkpoint models/larger.npz \
  --skip-latency --match-seconds 2 --games 20 --seed 3521 --workers 8 \
  --progress runs/engine-deep/progress.jsonl --output runs/engine-deep/matches.json
```

Use the ordinary deployment installation for the same pure Python backends.
The command requires a fresh progress path. Scheduling and CPU load can change
the exact number of completed nodes and the resulting moves.
