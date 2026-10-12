# Complete-game value estimates

Protocol recorded before full-game testing, 2026-10-12.

Component features and exact-value teaching improved late-game diagnostics,
but none of the three teaching snapshots passed its 10×10 screen. The 25-update
model finished 3 wins and 9 losses; the 100- and 300-update models each finished
5 wins and 7 losses. Better raw endgame values did not establish stronger full
games. This motivates a separate test of value estimation during branching play.

Keep the released graph network's weights and learned move priors. At each new
evaluation, simulate two legal continuations: take the most boxes when captures
are available, otherwise choose a random safe edge, falling back to a random
legal edge when necessary. Each continuation ends at a terminal game or at a
position supported by the exact long-chain/loop recurrence. Short or multiple
active components outside that fast scope continue playing. Oracle cutoffs do
not use wall time. Values are final win/draw/loss from the original player's
perspective; capturing a box preserves the turn throughout.

Use the mean of the two outcomes instead of the learned value in this first
arm. The network still supplies every move prior. Random streams derive from
the canonical position and fixed seed 1729, independent of cache hits, worker
assignment and match ordering. Mutable rollout buffers are local to an episode;
legal replay tests check their outcomes against the ordinary rules engine.

The screen uses 12 paired-seat 10×10 games, seed 15001, 1,024 tree simulations
for each side, constant-0.33 first-play search, fresh trees and the usual
12-edge root aid. Candidate and reference weights both have SHA-256
`21d8df7a60ba770e45a35d8405e009417904e250813fb3134403fb7707ae87b0`.
The candidate additionally spends computation on two playouts per uncached
evaluation and uses the scoped exact finish; this is **not equal total compute**.
Actual episodes, simulated moves, exact finishes and elapsed time are reported.
The user waived the move-time constraint for these larger-board experiments.

A score above 50% selects the frozen settings for 40 fresh games, seed 202001,
against released search. Promotion requires score above 50% and paired bootstrap
lower bound above 50%. A separate 20-game zero-first-play control uses seed
302001 and requires at least 50% score. Smaller-board behavior and actual web
settings need validation before any release. The live agent remains unchanged.

```bash
ALPHABOXES_PURE_NUMPY=1 python scripts/benchmark_engine.py \
  --checkpoint models/larger.npz --candidate-engine mcts \
  --candidate-rollouts 2 --rollout-weight 1 --rollout-seed 1729 \
  --search-options '{"fpu_reduction":0.33}' \
  --reference-search-options '{"fpu_reduction":0.33}' \
  --match-boards 10 --match-simulations 1024 --untimed --skip-latency \
  --games 12 --workers 6 --seed 15001 --output runs/rollout-screen.json
```

## Full replacement rejected; late-phase test

The first arm finished **1 win and 11 losses**, 8.33% score, paired interval
0–25%, and mean margin −51.33 boxes. Both sides completed 1,024 simulations per
nonexact decision. The candidate averaged 9.00 seconds versus 1.94 for reference,
with 1,055,322 playouts and 129,920,966 simulated moves. This is a decisive
screen failure; no confirmation runs. The continuations use a heuristic policy,
and accurate terminal scoring does not make them strong opponent forecasts.
[Validated results](data/large-model-20261012/rollout-2-summary.json).

The next isolated arm preserves the neural value while more than half the
original edges remain. It uses the same two-playout estimator after that
threshold, with unchanged weights, priors, blend weight and seed. The 12-game
screen uses fresh seed 16001 and the same 1,024-simulation comparison. A passing
candidate needs 40 fresh games at seed 203001 and the previously stated paired
confidence gate, followed by the 20-game zero-first-play control at seed 303001.
This tests whether changing opening estimates caused the earlier regression.

An optional Cython playout kernel preserves the complete action sequence,
outcome, episode length and exact-finish status of the Python implementation.
Comparisons cover seven board shapes from 1×2 through 12×12, with and without
the oracle, plus cached evaluator predictions. The late-phase screen uses this
compiled game loop; neural inference remains NumPy. It is selected explicitly
with `--rollout-backend cython`, independently of search/endgame acceleration.
Kernel builds replace extension files atomically so running processes retain
their loaded code.

The late-phase arm also finished **1 win and 11 losses**, with mean margin
−45 boxes. It is rejected; no confirmation runs. Both implementations preserve
the rules, but their heuristic continuations are poor value estimates for this
opponent. [Validated summaries](data/large-model-20261012/rollout-summary.json).
Later tests return to [additional learned checkpoints](later-component-checkpoints.md).

A four-position 10×10 timing probe compares uncached two-playout values after
the neural cache warms. At 220 / 170 / 110 / 60 remaining edges, Python medians
were approximately 19.7 / 16.9 / 12.1 / 5.76 ms, versus 4.21 / 3.46 / 2.28 /
1.28 ms for Cython. Predictions were identical, with roughly 4.5–5× lower
playout/evaluation cost. These measurements do not reverse the strength failure.
[Probe receipt](data/large-model-20261012/compiled-rollout-probe.json).
