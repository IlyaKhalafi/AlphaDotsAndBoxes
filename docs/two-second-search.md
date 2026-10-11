# Search improvements within two seconds

## Protocol (declared before confirmation)

The retained reference is `models/larger.npz` (SHA-256
`21d8df7a60ba770e45a35d8405e009417904e250813fb3134403fb7707ae87b0`),
the 96-channel, six-block GIN. All matches use NumPy inference with optional
compiled search/endgame extensions disabled and one BLAS thread per worker.

Screen independent changes on 12 games per board at 0.5 seconds. Advance a
promising change to 20 games per board with a 1.85-second candidate budget against
the retained reference's original two-second Deep setting. Screening results are
exploratory and do not qualify an upgrade.

The selected confirmation candidate keeps the original weights and changes only
first-play urgency: unvisited children start at the parent neural evaluation
minus 0.33, instead of zero. The exploration constant remains 1.5. This approach
is inspired by the [official Lc0 configuration documentation](https://lczero.org/play/configuration/flags/).

Confirmation uses 40 games per board on 4×4 and 5×5, seed 4201 (20 fresh opening
pairs, seeds 4201–4220). Each opening has six random moves and is played twice
with seats exchanged. These openings are disjoint from screening seeds. The
candidate has 1.75 seconds of internal search and a supervised 1.9-second response
deadline; the reference retains its original two-second soft search limit.

An upgrade requires all of the following:

- Candidate score above 50% on **each** board size.
- Lower 2.5th percentile of a 20,000-resample paired-opening bootstrap above
  50% on **each** board size (resampling whole pairs, seed 4199).
- Every recorded candidate decision below two seconds, including exact roots,
  IPC and deadline fallbacks.
- Full legal replay and score validation, plus opening/midgame/late-position
  latency checks on 10×10.

No claim of expert-human strength follows from engine-only matches. Board sizes
outside the tested matches retain a measured latency requirement, not a strength
guarantee.

## Deadline enforcement

Search runs in a warm, owned child process. The caller includes lock waiting in
its deadline and returns a legal tactical fallback if the worker stalls. A timed
out worker is terminated and replaced on the next request. Only that owned child
is managed. The internal budget leaves 150 ms before the response deadline and
another 100 ms before two seconds. OS scheduling and HTTP transport still need
measurement; an internal loop timer alone is insufficient.

## Other experiment: exact degree-two endgames

When each unclaimed box has at most two undrawn sides, the remaining graph is a
collection of paths and cycles. Canonical component states record their lengths
and number of external boundary edges. Cutting every distinct edge either splits
a path, opens a cycle, or captures endpoint boxes. Captures retain the turn;
other cuts reverse the value's sign. Memoized recursion evaluates these reduced
states exactly, including short chains, multiple capture fronts and sacrifices.
Complex positions or expired deadlines return to neural evaluation.

This extends the idea of reducing chain/loop endgames discussed by
[Buzzard and Ciere](https://arxiv.org/abs/1305.2156), but directly enumerates all
component cuts rather than assuming only take-all or standard handout play.
The implementation is checked against the full edge-subset solver on every
recognized 2×2 position and randomized 3×3 positions. Solver and proof propagation
remain opt-in until their own playing-strength tests justify enabling them.

## Results

All 80 planned confirmation games finished. Full replay validation and every
predeclared release gate passed. The selected search is deployed by the web UI;
checkpoint weights are unchanged. Training and historical fixed-simulation CLI
evaluation retain their original search defaults.

| Board | Wins–draws–losses | Score | Paired 95% interval | Mean box margin | Slowest candidate move |
| --- | --- | --- | --- | --- | --- |
| 4×4 | 24–5–11 | 66.25% | 53.75%–78.75% | +1.90 | 1.788 s |
| 5×5 | 33–0–7 | 82.50% | 70.00%–95.00% | +7.80 | 1.806 s |

Across 1,962 candidate decisions, none reached two seconds and none
used the tactical fallback. The reference kept its original two-second soft
budget; its longest move was 2.063 seconds. Both sides used deterministic
`argmax` tie breaking in the benchmark. The former UI randomized tied maxima;
the released UI uses the benchmark tie rule. Exact-root ties have equal optimal
box margins. These results compare a common tie protocol and the retained
checkpoint/search configuration, not every possible UI random seed.

### Screening decisions

Each row is a separate exploratory arm with its own opening seeds; do not pool
rows as a controlled comparison of changes. Score includes half a point per draw.

| Change | Budget (candidate/reference) | Games per board | 4×4 score | 5×5 score | Decision |
| --- | --- | --- | --- | --- | --- |
| 32-channel student with MCTS | 0.5/0.5 s | 12 | 62.5% | 33.33% | Reject: 5×5 regression |
| First-play reduction 0.33 | 0.5/0.5 s | 12 | 62.5% | 75% | Advance |
| Degree-two oracle + proof propagation | 0.5/0.5 s | 12 | 41.67% | 66.67% | Keep experimental |
| First-play reduction 0.33 | 1.85/2 s | 20 | 77.5% | 80% | Select for fresh confirmation |

### Observations

The useful improvement came from assigning unvisited actions a value relative
to the parent's neural evaluation. This changes where the limited search effort
goes. It does not increase model size, request more thinking time, or add training
games. More neural capacity or more traversals alone did not solve the earlier
strength/latency tradeoff. The smaller evaluator runs faster but lost on 5×5;
correct exact endgame values and proof propagation also failed the 4×4 screening.
Those features remain available for experiments and are disabled in the release.

### Runtime validation

An initial 60-position/repeat probe across 4×4, 5×5 and 10×10 took at most
1.781 seconds, without fallback. After runtime cleanup, the final 36-decision
probe took at most 1.805 seconds, again without fallback.
Both cover opening, midgame, 13-edge late positions and exact 12-edge roots.

A clean installation without Torch, Ray, Gymnasium or compiled extensions
completed API decisions on 1×1, 4×4, 5×5 and 10×10. The slowest complete
TestClient HTTP request took 1.775 seconds. Deliberately stalled evaluator
and queue-wait tests verify that supervision returns legal moves and replaces
only the owned worker. Runtime bookkeeping/formatting was refined during the
campaign; weights, first-play rule, exploration constant and match budgets were
fixed throughout confirmation. The final probes exercise the cleaned runtime.

### Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
ALPHABOXES_PURE_NUMPY=1 python scripts/benchmark_engine.py \
  --candidate-engine mcts --search-options '{"fpu_reduction":0.33}' \
  --match-seconds 1.75 --reference-seconds 2 --response-deadline 1.9 \
  --games 40 --workers 8 --seed 4201 --skip-latency \
  --output runs/fpu-confirmation.json --progress runs/fpu-confirmation.jsonl
python scripts/summarize_engine.py --receipts runs/fpu-confirmation.json \
  --output runs/fpu-summary.json
```

[Full replays, checkpoint/source hashes, summaries and gate receipt](data/two-second-20261011/).
Wall-clock search can change game traces across runs. A 10×10 latency check is
not evidence of 10×10 playing superiority, and engine matches do not establish
expert-human strength.
