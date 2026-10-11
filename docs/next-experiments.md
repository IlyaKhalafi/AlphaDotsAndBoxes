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

## 2. Better teaching targets

Inspect the raw value errors and remaining mistakes before adding network
capacity. The next isolated experiment uses exact final-outcome labels on
reachable late-game positions, while anchoring predictions on ordinary
self-play positions to reduce forgetting. Keep the architecture and search
budget unchanged. Compare a critic-only correction first, then exact policy
targets if the critic correction fails. Training, diagnostic and confirmation
seeds must remain separate.

An auxiliary box-margin head or a different graph architecture follows only
if these tests leave a measurable weakness. More width already failed the
previous release checks; that result does not rule out other architectures.

**Status:** implementation and screening in progress. The released model is
retained until fresh comparisons establish a practical improvement.
