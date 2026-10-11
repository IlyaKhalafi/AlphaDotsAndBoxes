# Controlled training refinement

Updated 2026-10-11 12:34 UTC.

All arms start from the same released 96-channel, six-layer checkpoint.
Search uses 512 simulations and a root-only 12-edge solver. Batch size is
8,192. The control uses 16 updates; the ratio arm uses four; the midgame
arm adds 50% archive starts to that four-update setting. Other settings match.
Each arm is bounded to 64 rounds or two hours, saving completed rounds.

| Arm | Rounds | New games | New positions | Hours | Last policy / value loss |
| --- | --- | --- | --- | --- | --- |
| control | 37 | 4,736 | 210,842 | 2.049 | 2.043 / 0.599 |
| ratio | 38 | 4,864 | 215,446 | 2.046 | 2.082 / 0.624 |
| midgame | 44 | 5,632 | 193,485 | 2.043 | 1.753 / 0.528 |

Losses describe the final replay minibatch in each round. Entropy, policy
mismatch, midgame starts and sample presentations are available in raw metrics.
They do not establish strength. Compare equal-round snapshots across arms.

![control loss](data/refinement-training-20261011/control-loss.svg)

![ratio loss](data/refinement-training-20261011/ratio-loss.svg)

![midgame loss](data/refinement-training-20261011/midgame-loss.svg)

**Deployment:** Released model retained.

[Raw settings and receipts](data/refinement-training-20261011/) retain model
hashes and complete game histories. Deployment inference uses NumPy only.
