# Timed search and the alpha–beta experiment

A simulation here is one MCTS traversal: follow selected edges in the search
tree, evaluate one resulting position, and back up its value. It is usually
not a complete game. Repeating this process distributes work among promising
moves. A fixed traversal count gives variable waiting time across board sizes.

The web UI now budgets elapsed thinking time: Quick 0.15 seconds, Standard 0.5
seconds, and Deep 2 seconds. MCTS stops after its current traversal when the
deadline expires, with an 8,192-traversal ceiling. A legal network-prior fallback
handles budgets too short to complete a traversal. Fixed-count search remains
available to training, command-line evaluation, and legacy API clients.

## What was implemented

The optional engine uses iterative deepening, alpha–beta pruning, policy-based
move ordering, and a bounded transposition table with exact, lower, and upper
bounds. Its completed search depth counts edges drawn. It keeps the last fully
completed iteration when time expires, or the network-prior fallback if none
finishes. Every legal move remains available, including moves that give away
boxes to control a chain. Captures retain the same value perspective; values
are negated only when the active player changes.

These are standard search techniques also used by
[Stockfish](https://github.com/official-stockfish/Stockfish/blob/master/src/search.cpp).
This is an independent implementation for Dots and Boxes, with the existing
GIN as its evaluator. Stockfish's
[NNUE evaluation](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/)
was designed for cheap incremental updates. Our six-layer message-passing GIN
evaluates the whole graph at each uncached leaf. Changing the tree-search
algorithm alone does not provide Stockfish's evaluation speed. Chess-specific
pruning assumptions do not automatically hold for extra turns and chain handouts.

Both interactive engines retain the existing exact 12-edge root endgame solver.
The network weights, RLlib training, export format, and NumPy-only deployment
requirements are unchanged. Alpha–beta is explicitly experimental; graph search
remains the default.

## Measured latency

The [raw receipt](data/engine-search-20261011.json) uses the released checkpoint
`21d8df7a60ba770e45a35d8405e009417904e250813fb3134403fb7707ae87b0`,
one BLAS/OMP/MKL thread, a training-free Python environment, and a cold evaluator
cache before each search. Each latency cell averages two repetitions. Positions
are legal random prefixes with seed 3501; this is a CPU latency test, not a
10×10 playing-strength test.

The checkout had optional Cython search selection and exact endgame extensions
enabled for both match players. The ordinary port-9003 installation uses the
pure Python/NumPy paths; its latency is also measured separately. Raw receipts
identify extension availability and hashes. Optional extensions are not required
to run the benchmark, and their presence can change the number of completed nodes.

| 10×10 position | Engine | Standard elapsed | Deep elapsed | Completed alpha–beta depth, Standard / Deep |
| --- | --- | ---: | ---: | ---: |
| 220 edges available | Graph search | 0.505 s | 2.006 s | — |
| 220 edges available | Alpha–beta | 0.503 s | 2.003 s | 0 / 1 |
| 110 edges available | Graph search | 0.502 s | 2.015 s | — |
| 110 edges available | Alpha–beta | 0.503 s | 2.000 s | 0 / 1 |
| 39 edges available | Graph search | 0.505 s | 2.007 s | — |
| 39 edges available | Alpha–beta | 0.504 s | 2.005 s | 1 / 1 |

The empty board has 220 legal moves. Alpha–beta cannot finish checking all of
them at depth one in half a second, so it returns the prior fallback. MCTS
completes about 101–102 traversals in that time and returns their visit counts.
At two seconds, alpha–beta completes depth one but not depth two in these samples.

The separate [ordinary deployment check](data/engine-deployment-latency-20261011.json)
uses the installed package with both optional extensions absent. Over the same
three 10×10 position types, graph search takes 0.501–0.507 seconds at Standard
and 2.002–2.007 seconds at Deep (one repetition each). Alpha–beta's completed
depths remain 0 / 1, 0 / 1, and 1 / 1 respectively.

These are **soft thinking budgets**, not request-latency guarantees. One GNN
evaluation or the uninterruptible small exact root solve can exceed a deadline.
The server serializes searches, so another session can add queue time; animation
delays and network latency are also excluded from the benchmark.

## Equal-time playing screen

Both players use identical weights, a separate evaluator cache, the same time
allowance, and the same exact root aid. Each game pair swaps seats on the same
six-move random opening. A fresh planner is created per move, matching the web
service. Full move histories and timing counters are included in the receipts.

| Thinking time per player | Board | Alpha–beta wins / draws / losses | Alpha–beta score |
| --- | --- | ---: | ---: |
| 0.1 s | 4×4 | 2 / 0 / 6 | 25% |
| 0.1 s | 5×5 | 2 / 0 / 6 | 25% |
| 0.5 s | 4×4 | 1 / 0 / 7 | 12.5% |
| 0.5 s | 5×5 | 0 / 0 / 8 | 0% |

Eight games per board are a small screen, not a statistical strength claim.
Neither budget supports replacing MCTS with this engine. The
[half-second follow-up](data/engine-standard-20261011.json) uses fresh opening
seed 3511. Improving the evaluation cost and its behavior under minimax is a
better next experiment than deploying this slower-searching alternative as
the default. Expert-human strength and 10×10 strength remain unproven.

A fresh [40-game Deep comparison](engine-deep-benchmark.md) uses the ordinary
deployment backends and a two-second budget for both players. Alpha–beta scored
25% on 4×4 and 10% on 5×5 against the retained model's graph search.

The [smaller-evaluator experiment](evaluator-distillation.md) tests a narrower
six-layer GIN with unchanged search and equal-time paired matches.

## Reproduce

The benchmark needs ordinary deployment dependencies only. Its default run
measures 4×4, 5×5, and 10×10 latency and plays eight equal-time games per board
on 4×4 and 5×5. Wall-clock cutoffs make exact move traces sensitive to machine load.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/benchmark_engine.py --output runs/engine-search.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/benchmark_engine.py --boards 10 --seconds 0.5 --repeats 1 \
  --match-seconds 0.5 --games 8 --seed 3511 --output runs/engine-standard.json
```

Verification covers full-depth search against the independent exact oracle,
extra-turn perspective, pruning and transposition reuse, timeout fallback,
discarding unfinished iterations, API/replay counters, and a training-free
10×10 deployment smoke check. The complete suite passed 103 tests. Browser checks
passed on desktop, mobile, rectangular boards, hints, undo, and delayed self-play;
the README GIF was regenerated from the current port-9003 UI.
