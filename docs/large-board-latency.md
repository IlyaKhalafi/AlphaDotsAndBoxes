# Thinking time on a 10×10 board

This records the earlier fixed-work search. The UI now uses elapsed-time budgets;
see [the timed-search measurements](engine-search.md) for current behavior.

Measured 2026-10-11 using the released 96-channel, six-layer model in the
NumPy-only deployment environment. A 10×10-box board has 220 playable edges
and 320 graph nodes. The UI supports this size through its custom board controls.

| Position | Standard, 128 simulations | Deep, 512 simulations | Experimental, 8,192 simulations |
| --- | --- | --- | --- |
| Empty board | 0.86 s | 3.49 s | 38.10 s |
| 110 edges remaining | 0.47 s | 1.82 s | 21.21 s |
| 40 edges remaining | 0.45 s | 1.30 s | 8.73 s |

Each entry averages two searches with a cleared prediction cache. Later
positions come from a fixed random legal prefix, seed 3400; they are timing
samples rather than representative human positions. Both budgets retain the
same 12-edge root solver, with no leaf solving or tree reuse. All timed states
have more than 12 legal edges, so the root solver does not bypass their searches.
One CPU thread performs inference and search. Shared host load affects timing;
GPU training performance is unrelated to these deployment measurements.

Times exclude human thinking, UI animations, intentional spectator delays and
requests waiting for another game. At measurement time, the UI offered budgets
up to 512; 8,192 was an experimental comparison.

These are **per-agent-move** measurements, not full-game durations. A completed
game draws all 220 edges; the number drawn by each player varies because
capturing a box preserves the turn. The model accepts this board size, but
strong 10×10 play has not been established. The subsequent UI update limits
thinking by elapsed time on every board size.

[Raw timings and provenance](data/next-experiments-20261011/latency-10x10.json).
