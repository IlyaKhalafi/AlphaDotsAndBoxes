# Search and teaching results

Updated 2026-10-11 15:54 UTC. Stage: critic teaching confirmed.

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

## Exact critic teaching

| Update | Simulations | Board | W–D–L | Score |
| --- | --- | --- | --- | --- |
| 25 | 128 | 4×4 | 14–0–6 | 70.0% |
| 25 | 128 | 5×5 | 11–0–9 | 55.0% |
| 25 | 512 | 4×4 | 10–2–8 | 55.0% |
| 25 | 512 | 5×5 | 9–0–11 | 45.0% |
| 100 | 128 | 4×4 | 9–1–10 | 47.5% |
| 100 | 128 | 5×5 | 9–0–11 | 45.0% |
| 100 | 512 | 4×4 | 10–1–9 | 52.5% |
| 100 | 512 | 5×5 | 9–0–11 | 45.0% |
| 300 | 128 | 4×4 | 11–2–7 | 60.0% |
| 300 | 128 | 5×5 | 10–0–10 | 50.0% |
| 300 | 512 | 4×4 | 11–1–8 | 57.5% |
| 300 | 512 | 5×5 | 14–0–6 | 70.0% |

Screening winner: critic-00300.

Fresh confirmation:

| Simulations | Board | Games | W–D–L | Score |
| --- | --- | --- | --- | --- |
| 128 | 4×4 | 80 | 43–7–30 | 58.1% |
| 128 | 5×5 | 80 | 32–0–48 | 40.0% |
| 512 | 4×4 | 80 | 29–9–42 | 41.9% |
| 512 | 5×5 | 80 | 53–0–27 | 66.2% |

Release gate: failed.

**Deployment:** previous released weights retained.

Selection scores are optimistic. These limited seeded checks do not establish expert-human strength.
Larger search budgets spend more computation; latency is part of their tradeoff.
Teaching losses measure mixed oracle/anchor batches, not held-out game skill.

[Raw settings and receipts](data/next-experiments-20261011/).
