# Search budget and exact supervision

Protocol recorded before results, 2026-10-11.

The previous three-arm campaign produced no checkpoint that passed all release
checks. Its nine checkpoints were screened in 720 games; no independent
confirmation ran because none passed screening. Loss reduction alone did not
predict a reliable playing-strength gain.

## 1. More search with unchanged weights

Use the released 96-channel, six-layer NumPy checkpoint for both players.
Compare **2,048**, then **8,192** simulations per move against **512**.
Both players use the same root-only 12-edge exact solver, fresh trees, no
exploration noise and no exact leaf aid. Seeds 3200 and 3201 each provide
20 games on each of 4×4 and 5×5, with balanced seats and six random opening
moves. Each candidate budget therefore has 40 games per board, 80 total.

Record wins, draws, losses, score rate, move histories and mean search time.
This deliberately spends more computation on the candidate; it tests the
strength/cost tradeoff, not equal-compute superiority. Board sizes and solver
settings stay fixed. The existing equal-budget pooling tool is inappropriate
for this asymmetric comparison.

Separately, a seed-3202 exact diagnostic suite measures raw network value error,
search value error and move regret at all three budgets. Its eight structured
fixtures are shared with older suites; the other 48 positions are fresh.
These late-game positions do not measure opening or full-game strength.

```bash
python scripts/search_budget.py --checkpoint models/larger.npz \
  --output runs/search-budget --budgets 2048 8192 --baseline-budget 512
```

Any promising budget needs fresh seeds before changing the live UI. Screening
requires at least 50% score on both boards and more than 50% on 5×5. Larger
budgets may warrant an optional slower setting rather than a default change;
move time must be reported alongside strength.

Search confirmation uses seeds 3203 and 3204, each with 40 games per board.
If both budgets pass screening, confirm the one with the higher mean score;
prefer 2,048 on a tie. Full match histories and timing receipts remain available.

The diagnostic run completed in the NumPy-only environment:

| Simulations | Outcome correct | Best box margin | Mean root value error | Mean raw network value error |
| --- | --- | --- | --- | --- |
| 512 | 56/56 | 56/56 | 0.145 | 0.320 |
| 2,048 | 56/56 | 55/56 | 0.112 | 0.320 |
| 8,192 | 56/56 | 55/56 | 0.073 | 0.320 |

The larger budgets gave up four additional boxes in the shared
`loops_6_4_chain_5` forced-loss fixture, while preserving the best outcome.
The learning objective is win/draw/loss, so this is a margin regression, not
a lost achievable win. Search improves value estimates here without a
corresponding improvement in chosen moves. Root aid directly solves positions
with 12 or fewer remaining edges; full-game results are still needed.

[Raw diagnostic receipts](data/next-experiments-20261011/) include source and
suite hashes, policies, decisions and timings.

## 2. Better teaching targets

Inspect the raw value errors and remaining mistakes before adding network
capacity. The next isolated experiment uses exact final-outcome labels on
reachable late-game positions, while anchoring predictions on ordinary
self-play positions to reduce forgetting. Keep the architecture and search
budget unchanged. Compare a critic-only correction first, then exact policy
targets if the critic correction fails. Training, diagnostic and confirmation
seeds must remain separate.

The teaching dataset replays the previous campaign's complete game histories
(seed 3160), deduplicating states. It contains **2,760** exact positions with
13–16 remaining edges and **4,096** earlier anchor positions. Oracle labels
include already claimed boxes and capture extra turns. Exact outcomes are
912 losses, 163 draws and 1,685 wins from the active player's perspective.
Prediction anchors come from the immutable released model, rather than from
the candidate under training. Dataset generation used the compiled development
solver; deployment match comparisons use Python and NumPy only.

Both teaching arms start independently from that release, use RLlib, learning
rate 0.00003, batches of 4,096, 25% oracle positions and 75% anchor positions.
All positions retain the released policy target in the critic arm. The second
arm additionally uses uniform exact best-margin policies on oracle positions.
Oracle values replace uncertain self-play outcomes only in the oracle subset;
earlier positions retain both frozen predictions. No head or layer is added.
The mixed loss is a teaching objective, not a playing-strength measurement.

Save updates 25, 100 and 300; screen each at 128 and 512 simulations on 4×4
and 5×5, 20 games per condition, seed 3320. A passing snapshot advances to
80 games per condition using fresh seeds 3330 and 3331. Keep the existing
release gate: at least 50% score throughout, more than 50% on Deep 5×5, and
mean move time within twice the release. If the critic arm passes independent
confirmation, stop before trying the policy arm. Otherwise test the policy
arm with its own confirmation seeds 3340 and 3341. Seed 3311 supplies a
diagnostic suite with explicit training-overlap checks; it never supplies
training targets. The learner uses a 28 GiB allocation cap and 50% duty
cycle; there are no GPU self-play workers in this short teaching experiment.

```bash
python scripts/teach.py --checkpoint models/larger.npz \
  --sources runs/source-matches/*.json --output runs/teaching.npz
python scripts/teach.py --checkpoint models/larger.pt \
  --dataset runs/teaching.npz --output runs/critic-correction
```

An auxiliary box-margin head or a different graph architecture follows only
if these tests leave a measurable weakness. More width already failed the
previous release checks; that result does not rule out other architectures.

**Status:** implementation and screening in progress. The released model is
retained until fresh comparisons establish a practical improvement.
