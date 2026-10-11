# Search and teaching results

Updated 2026-10-11 15:26 UTC. Stage: fresh search confirmation complete.

[Recorded protocol](next-experiments.md) separates screening from fresh confirmation.
All match inference uses NumPy with Python search and solving.

## Search screening

| Budget vs 512 | Board | Games | W–D–L | Score | Seconds/move, agent / release |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4x4 | 40 | 23–6–11 | 65.0% | 0.554 / 0.211 |
| 2048 | 5x5 | 40 | 22–0–18 | 55.0% | 0.938 / 0.350 |
| 8192 | 4x4 | 40 | 24–7–9 | 68.8% | 1.694 / 0.216 |
| 8192 | 5x5 | 40 | 27–0–13 | 67.5% | 3.027 / 0.371 |

## Fresh search confirmation

| Budget vs 512 | Board | Games | W–D–L | Score | Seconds/move, agent / release |
| --- | --- | --- | --- | --- | --- |
| 8192 | 4x4 | 80 | 47–13–20 | 66.9% | 1.630 / 0.210 |
| 8192 | 5x5 | 80 | 47–0–33 | 58.8% | 2.984 / 0.349 |

**Deployment:** previous released weights retained.

Selection scores are optimistic. These limited seeded checks do not establish expert-human strength.
Larger search budgets spend more computation; latency is part of their tradeoff.
Teaching losses measure mixed oracle/anchor batches, not held-out game skill.

[Raw settings and receipts](data/next-experiments-20261011/).
