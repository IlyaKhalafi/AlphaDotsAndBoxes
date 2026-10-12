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
