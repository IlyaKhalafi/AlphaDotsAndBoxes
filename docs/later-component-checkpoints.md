# Later large-board checkpoints

Protocol recorded before these full-game screens, 2026-10-12.

Both complete-game rollout variants lost 1–11. Full replacement and replacement
only below half the original edge count are rejected. Their scoring rules and
mutable simulation agree with the immutable game; that does not make the
heuristic continuations good forecasts against the retained opponent.

The next experiment tests additional self-play updates with unchanged search.
Freeze component iteration 14 (672 games, 90,442 positions), NumPy SHA-256
`040b8c4ff8ac59372e44b37bcc78d73864766abfd1e2c705d787bba686b2e006`, and
data-only iteration 15 (720 games), SHA-256
`20783de10567f976bd45f2147c32ecedf7f8c32d76efe6d78a0b403d854acb81`.
They are separate model screens, not an equal-training-count causal comparison.
The component run's sampler streams restarted before iteration 14.

Use 12 paired-seat 10×10 games per snapshot at 1,024 simulations, seeds 17001
and 18001 respectively, against retained constant-0.33 search. Both sides use
fresh trees and the same 12-edge root aid, without structural aid, proofs,
rollouts, exploration noise or deadlines. If both score above 50%, select the
higher score and prefer the component snapshot on a tie. Freeze the winner for
40 fresh games, seed 204001; require score above 50% and paired bootstrap lower
bound above 50%. A separate 20-game zero-first-play control uses seed 304001
and requires at least 50% score. Deployment still needs smaller-board and actual
web-setting checks.

An illustrative allocation check used one common six-move 10×10 opening, seed
18101 + board size. At 1,024 simulations with released search, retained and
component snapshots each visited one root action, with median endpoint depths
69 and 67. The data-only snapshot visited nine, with median depth 57. All chose
edge 43. The component snapshot does not show the proposed broad-opening-search
problem on this example, so no fixed first-play value is added in this experiment.
This diagnostic is not a full-game strength result.

## Contingency: deeper planning with retained weights

If neither later snapshot passes screening, the next isolated test keeps the
retained weights and constant-0.33 search and increases candidate work to 8,192
simulations against reference 1,024. Use 12 paired-seat 10×10 games, seed 19001,
without timers, structural aid, proofs, reuse or rollouts. This changes computation
only and is explicitly an asymmetric cost/strength comparison. Record separate
per-side budgets and every actual simulation count. A passing screen requires
40 fresh games, seed 205001, and the same paired confidence gate, followed by
20 zero-first-play controls at seed 305001. No result is assumed from more work.

The benchmark accepts `--candidate-simulations` and `--reference-simulations`;
omitting both retains the shared `--match-simulations` budget. Replay validation
checks each side against its own ceiling. Serial/parallel regression tests cover
equal and unequal budgets and reject receipts with excess work.
