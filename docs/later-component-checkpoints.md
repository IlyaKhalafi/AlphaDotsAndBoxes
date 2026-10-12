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

## Screen results

Component iteration 14 lost 4–8 (33.33%), with mean margin −1.17 boxes. It is
rejected. Data-only iteration 15 won 8–4 (66.67%), with mean margin +30.33;
its paired bootstrap interval is 50–83.33%. The selected data-only snapshot
now enters the predeclared 40-game confirmation at fresh seed 204001. Its
weights and both sides' search settings remain frozen. This is a promising
screen, not a confirmed improvement or an architecture comparison.

The [validated summary](data/large-model-20261012/later-screen-summary.json)
links complete histories for both candidates. The live weights are unchanged
pending confirmation and deployment checks.

The [NumPy parity check](data/large-model-20261012/data-control-00015-export-parity.json)
covers 738 legally reached positions across eight board shapes, including
10×10 and 12×12, with seven padding nodes. It passes relative/absolute
tolerances: maximum logit difference 3.82e−6 and value difference 8.35e−7.
The [responsiveness probe](data/large-model-20261012/data-control-00015-deadline-probe.json)
checks 36 supervised decisions on 4×4, 5×5 and 10×10. All finish below two
seconds (maximum 1.768 seconds), with no fallback. Neither probe is a
full-game strength comparison under the web deadline.

## Data-only confirmation: rejected

Fresh confirmation retains 24 complete opening-paired games: 9 wins and
15 losses (37.5%, paired interval 16.67–58.33%), mean margin −4.33 boxes.
The screen's 8–4 result did not hold on fresh openings, so the selected
iteration-15 checkpoint is rejected and the zero-first-play control is not run.

The planned 40-game run stops when its confidence gate becomes unreachable.
Even 16 further wins would yield only 45% for the final paired-bootstrap lower
bound, below the required strictly-above-50% threshold. The optimistic final
score would be 62.5%. Any other future outcomes can only reduce that bound
under the fixed final resample indices. Stopping rejects a futile candidate;
it does not promote a favorable partial result.

The [receipt](data/large-model-20261012/data-control-00015-confirm.json) records
this bound and reconstruction provenance. Source hashes are verified against
the matching commit-321de70 source snapshot; weights match the frozen selection
hash. The [summary](data/large-model-20261012/data-control-00015-confirm-summary.json)
legally replays every retained game. [Unfinished opening-pair traces](data/large-model-20261012/data-control-00015-confirm-inflight.json)
are retained separately. Only the owned benchmark and its workers were stopped;
self-play continues within the existing budgets.

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
