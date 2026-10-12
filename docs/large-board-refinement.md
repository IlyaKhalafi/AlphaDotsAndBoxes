# Larger-board search refinement

## Objective and protocol

Improve 10×10 play against the released constant first-play reduction 0.33,
using ordinary NumPy inference. The existing model's SHA-256 is
`21d8df7a60ba770e45a35d8405e009417904e250813fb3134403fb7707ae87b0`.
No neural weights change during the initial search experiments.

Two isolated first-play rules are screened:

- `policy_mass`: subtract `0.33 × sqrt(visited policy mass)` from the parent
  neural value. Visited policy mass sums the original priors of children with
  at least one visit. This starts with a smaller penalty and increases it as
  search explores more of the policy. The probability-mass factor is inspired
  by [Lc0's implementation](https://github.com/LeelaChessZero/lc0/blob/master/src/search/classic/search.cc),
  but this arm keeps the current agent's raw parent neural value rather than
  switching to the parent's backed-up mean.
- `branching`: subtract `0.33 × min(1, sqrt(54 / legal actions))`. The penalty
  shrinks on wide nodes and equals the released setting once at most 54 actions
  remain. The anchor is the 5×5 board after six opening moves.

Each screen uses 12 games (six opening pairs), 512 full simulations per nonexact
move, no timers, seed 4501, and swapped seats. Both arms use the same openings
for comparison; their observations are correlated and are not pooled as
independent evidence. Both references explicitly use constant reduction 0.33.
The other settings remain exploration constant 1.5, exact 12-edge roots, no
noise, no tree reuse, no leaf oracle/proof propagation and deterministic argmax.
Four single-threaded CPU workers per arm keep combined match work at eight
cores. Compiled kernels are disabled.

An arm above 50% screen score may advance; a screen alone does not qualify a
release. Confirmation uses 40 fresh games at 1,024 full simulations against the
released search, seed 4601 (opening seeds 4601–4620), with no move timer. Require
score above 50% and the lower 2.5th percentile of a 20,000-resample paired-opening
bootstrap (seed 4199) above 50%. Run a separate 20-game control against the older
zero-first-play search, seed 4701, requiring at least 50% score. Full legal
replays and actual work budgets must be checked. Confirmation settings are
fixed before the fresh matches, including any combined changes.

Changes will remain experimental until those checks pass. Before deployment,
check 4×4 and 5×5 behavior and actual large-board web settings. Larger-board
testing does not require a two-second move limit; the live service retains its
existing deadline until a concrete validated change is ready.

## Opening diagnostics

At 1,024 simulations on one independent 10×10 opening (diagnostic seed 4501 plus
board size), constant reduction 0.33 visited one of 214 root actions, with median
endpoint depth 71. Policy-mass scaling visited 92 (median depth 2, maximum 9);
branching scaling visited 183 (median depth 3, maximum 99). These are illustrative
positions, not strength measurements.

[Full allocation receipt](data/large-search-20261012/allocation-screen.json).

## Results

Initial screens are running. No larger-board improvement is claimed yet.
