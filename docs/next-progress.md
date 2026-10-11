# Search and teaching results

Updated 2026-10-11 13:54 UTC. Stage: 2048 screening complete; 8192 screening running.

[Recorded protocol](next-experiments.md) separates screening from fresh confirmation.
All match inference uses NumPy with Python search and solving.

## Search screening

| Budget vs 512 | Board | Games | W–D–L | Score | Seconds/move, agent / release |
| --- | --- | --- | --- | --- | --- |
| 2048 | 4x4 | 40 | 23–6–11 | 65.0% | 0.554 / 0.211 |
| 2048 | 5x5 | 40 | 22–0–18 | 55.0% | 0.938 / 0.350 |

**Deployment:** previous released weights retained.

Selection scores are optimistic. These limited seeded checks do not establish expert-human strength.
Larger search budgets spend more computation; latency is part of their tradeoff.
Teaching losses measure mixed oracle/anchor batches, not held-out game skill.

[Raw settings and receipts](data/next-experiments-20261011/).
