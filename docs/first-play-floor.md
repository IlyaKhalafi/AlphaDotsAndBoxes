# Keep first-play estimates inside outcome bounds

Protocol recorded before full-game testing, 2026-10-12.

Released search subtracts 0.33 from the parent's network value when estimating
an unvisited action. A parent prediction below −0.67 consequently gives an
estimate below −1, the minimum possible win/draw/loss return. This can make
unvisited alternatives less attractive than a heavily visited losing branch.
The next hypothesis floors that estimate at −1, keeping the reduction and
the rest of search unchanged. A pessimistic heuristic is not inherently a
correctness error; this experiment tests whether bounding it helps play.

Tests construct a strongly negative parent with a known losing branch and a
lower-prior unknown branch. The bounded estimate visits the unknown branch;
the released estimate stays on the losing branch. Predictions within the floor
retain the same choice. Both ordinary and cached sibling selection are tested.
The default floor remains disabled.

Use retained weights on both sides, 12 paired-seat 10×10 games, seed 29001,
1,024 simulations, constant-0.33 and the common 12-edge root aid. Both sides
enable the validated cached selector. Candidate alone sets `fpu_floor=-1`;
reference retains the released unbounded first-play estimate. No timer, new
weights, structural oracle, proof propagation, pruning, rollout or tree reuse
is enabled. Freeze package and script source before launch.

A score above 50% advances to 40 fresh games, seed 212001, with score above
50% and paired bootstrap lower bound above 50%. The separate zero-first-play
control uses 20 games, seed 312001, and requires at least 50% score. Smaller-board
and actual web-setting checks remain necessary before release. No improvement
is assumed from this local allocation example.

## Screen: rejected

The complete screen scores 5 wins and 7 losses (41.67%, paired interval
25–50%), with mean margin −10.67 boxes. Candidate decisions above the common
root-aid threshold average 1.48 seconds versus 1.33 for the reference. The
local allocation behavior changes as intended but does not improve these
complete games. The floor remains disabled and is not deployed.

[Full histories](data/large-model-20261012/fpu-floor-screen.json) and
[legal replay validation](data/large-model-20261012/fpu-floor-screen-summary.json)
retain all 12 games. The complete regression suite passes 213 tests.
