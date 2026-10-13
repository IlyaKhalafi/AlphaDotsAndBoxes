# Deeper search improves 10×10 play

The retained NumPy network wins **35–5 in 40 fresh 10×10 games** when given
8,192 search simulations against the same network at 1,024. The score is 87.5%,
with a paired-opening bootstrap 95% interval of 77.5–97.5% and mean margin
+31.85 boxes. This passes the pre-recorded confirmation gate. A separate
20-game control against zero first-play reduction finishes **18–2** (90%,
interval 75–100%, mean margin +39.4).

This is a gain from more computation, not new weights or a larger network.
Both sides use retained NumPy SHA-256
`21d8df7a60ba770e45a35d8405e009417904e250813fb3134403fb7707ae87b0`.
The network remains a six-block, width-96 residual GIN. Neither this result nor
the earlier human matches establishes expert-human strength.

## Frozen evaluation

The [pre-recorded protocol](later-component-checkpoints.md) fixes the screen
at seed 19001, fresh confirmation at 205001, and control at 305001. Each opening
is played in both seats. Confirmation uses six CPU workers and control uses
two, with one BLAS thread each. The immutable `selection-v1` package and
benchmark come from commit `317d36f`; the earlier source snapshot preserves
their fingerprints. The 12-game screen finished 9–3 before confirmation began.

Both sides use fresh trees, constant first-play reduction 0.33 (except the
zero-reduction reference control), cached child selection, and exact root
solving with at most 12 remaining edges. Structural solving, proofs, hard
safe-move filtering, root safety preference, chain pruning, forced-capture
extensions, rollouts, tree reuse, exploration noise and deadlines are disabled.
The candidate receives 8,192 simulations and the reference 1,024; this is
explicitly an unequal-computation comparison.

Every candidate neural decision completes 8,192 simulations: 4,740 in
confirmation and 2,432 in control. Confirmation candidate decisions average
18.99 seconds and reach 54.56 seconds; reference decisions average 1.67 seconds.
Control candidate decisions average 17.74 seconds and reach 51.82 seconds.
Worker contention differs between runs, so these timings do not establish an
isolated speed comparison. No GPU memory is reserved by these benchmarks.

The [validated summary](data/large-model-20261012/completed-search-summary.json)
links full move histories and file hashes. Validation legally replays every
game, checks scores and extra turns, verifies paired openings and seats, and
checks actual search counts against each side's budget. Confidence intervals
resample whole opening pairs (20,000 samples, seed 4199). They describe these
fresh seeded comparisons, not every possible opponent or opening.

## Optional deployment

Start NumPy serving with `--extended-search`. On boards with at least 100 boxes,
choose the graph engine and **Deep · up to 60 s**. This requests the evaluated
8,192-simulation profile in a separate supervised worker. Smaller boards and
ordinary requests keep their 1.75-second internal cap and 1.9-second response
deadline. The longer mode is optional and disabled unless the server flag is set.

The long worker has a 60-second response deadline, including queueing. Its
internal time allowance can stop search before all 8,192 simulations complete.
Replay analysis reports `simulations`, `extended`, `full_budget_completed`, and
`timed_out`; partial or fallback decisions must not be equated with the fixed-work
benchmark. Boards larger than 10×10 have not received this strength evaluation.
The [actual NumPy-only API probe](data/large-model-20261012/extended-web-probe.json)
checks full-budget completion on 10×10, ordinary 4×4/5×5 responsiveness, and
absence of imported training frameworks.

The clean installed package matches the committed search and web sources;
the [release receipt](data/large-model-20261012/extended-release.json) records
their fingerprints and the port-9003 health settings. All 260 tests pass.
Browser checks cover the large-board thinking label, switching back to ordinary
thinking, mobile layout, keyboard moves, hints, undo, rectangular boards, and
agent-versus-agent pause and delay. The README GIF contains 27 frames from a
real game on the updated service. No weights or architecture diagram changed.
