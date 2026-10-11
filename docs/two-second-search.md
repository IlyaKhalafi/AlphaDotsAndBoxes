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

Receipts and validated summaries are in `docs/data/two-second-20261011/`.
Confirmation results will be recorded after all planned games finish.
