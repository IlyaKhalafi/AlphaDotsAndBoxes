# Decision latency from 3×3 through 10×10

2026-10-13. Measurements use the current C++-accelerated width-96, depth-6
network, retained checkpoint `21d8df7a…`, NumPy 2.5.3 and one BLAS thread.
Dimensions count boxes. Each board starts empty and receives three consecutive
agent moves. This is an opening sample on shared hardware, not a full-game
average or a worst-case response bound.

Two configurations are measured sequentially:

- **Live UI Deep:** HTTP requests on port 9003. Boards through 9×9 use the
  existing 1.75-second internal cap and 1.9-second supervised deadline. On
  10×10, extended Deep completes all 8,192 simulations. Persistent serving
  caches may already contain predictions from earlier games; they are not
  cleared, and existing user sessions are not changed.
- **Fixed work:** an isolated evaluator starts with an empty prediction cache
  for each board, then retains predictions across the next two moves. Every
  decision completes exactly 8,192 simulations, with the cached selector,
  first-play reduction 0.33, root exact threshold 12, no time limit, no noise
  and no tree reuse. The cache limit is 32,768 entries on every size. These
  timings measure the search call, excluding HTTP. They provide a comparison
  at the same work budget; the live UI uses its published size-dependent limits.

All values below are seconds.

| Board | Live UI Deep mean | Full 8,192: cold first move | Full 8,192: next two moves mean |
| --- | --- | --- | --- |
| 3×3 | 1.77 | 3.42 | 3.08 |
| 4×4 | 1.77 | 5.00 | 4.66 |
| 5×5 | 1.77 | 6.85 | 6.10 |
| 6×6 | 1.77 | 10.14 | 9.52 |
| 7×7 | 1.77 | 17.26 | 11.30 |
| 8×8 | 1.77 | 20.69 | 11.17 |
| 9×9 | 1.77 | 19.80 | 10.89 |
| 10×10 | 9.54 | 23.60 | 11.72 |

The three live 10×10 requests take **9.25, 10.07 and 9.29 seconds**, all with
8,192 simulations and no timeout fallback. Their cache is already warm from
earlier play, so they are not cold-opening measurements. Ordinary Deep takes
1.767–1.780 seconds across 3×3–9×9. Completed simulations vary by size and move:
3×3 completes 3,706–5,096, while 9×9 completes 532–845 in the measured requests.
No live request falls back because of its deadline.

The 9×9 cold sample being faster than 8×8 is not evidence that larger boards
always cost less. The graph crosses the 256-node cutoff for C++ grid aggregation
at 9×9; 8×8 uses dense graph aggregation. The searches also visit different
positions, and CPU scheduling varies on the shared machine. Subsequent human
moves can change how much prediction reuse is available. Small exact endgames
can finish much faster than these opening searches.

The [raw receipt](data/large-model-20261012/board-size-latency.json) records all
48 decisions, actions for the isolated searches, simulation counts, deadlines,
checkpoint and source hashes, configuration and timestamps. Validation checks
that each stage contains all eight boards with moves 0, 1 and 2, every fixed-work
decision completes 8,192 simulations, and no UI response times out. No model,
budget, serving setting or training job is changed by this measurement.
