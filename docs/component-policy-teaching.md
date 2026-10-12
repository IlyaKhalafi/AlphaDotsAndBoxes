# Exact policy and value teaching

Protocol recorded before updates, 2026-10-12.

The earlier critic-only experiment improved endgame value error, but all three
full-game screens failed. Its policies remained frozen network targets. This
arm instead teaches both the exact outcome and the exact best-margin move set
on those same legally reached component positions. Earlier replay positions
retain their frozen policy and value anchors.

Reuse the frozen iteration-three starting checkpoint, SHA-256
`7185d4f820cb2dba33ed9bfead8ce3496572ebde1403df03ffce15cd5ffa505e`, and
the existing 2,048-exact/2,048-anchor dataset, SHA-256
`fe54d1bb135ae814a10dacd3c2869ba8c126f7d3873e8728e9089a7529132b29`.
No evaluation game supplies a target. Keep the critic arm's RLlib update
settings: 300 updates, batch 512, 25% exact positions, learning rate 0.00003,
seed 41072, 8 GiB allocator cap and 50% pacing. Identical batch sampling helps
compare learning diagnostics; previous full-game screens used different
opening seeds, so they are not a paired causal game comparison.

Select only the predeclared final update 300 for the full-game screen.
Intermediate saved checkpoints are not screened or selected. Freeze its NumPy
export and use 12 paired-seat 10×10 games, fresh seed 22001, at 1,024
simulations against retained weights. Both sides use constant-0.33 search,
fresh trees and the common 12-edge root aid. No timer, structural solver,
proof propagation, chain pruning, capture extension or rollout is enabled.
This isolates changed targets from changed search.

A score above 50% advances to 40 fresh games, seed 209001. Promotion requires
score above 50% and paired bootstrap lower bound above 50%. The separate
20-game zero-first-play control uses seed 309001 and requires at least 50%
score. Smaller-board, export-parity and actual web-setting checks follow.

The existing self-play jobs keep their eight-hour budgets and resource limits.
The short additional learner raises owned allocator ceilings from 54 to 62 GiB;
the unrelated GPU workload is left running.
