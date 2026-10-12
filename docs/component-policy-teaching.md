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

## Completed updates

All 300 updates complete in 303.46 seconds of update-loop wall time, excluding
loading and Ray startup. Peak allocated GPU memory is 4,971.6 MiB. Final mixed
policy loss is 2.9966, value loss 0.03295, value MAE 0.07557 and policy KL
0.04303. These metrics combine exact targets and anchors; they are not match
results. The compiled component encoder supplies identical features to its
tested Python counterpart.

The frozen final NumPy snapshot has SHA-256
`0810e8ab0e93eb8861645ec98038d4a00ccff91742c9679f6ef2e6b75f5bb111`.
Its predeclared full-game screen has started. [All update records, settings and
source hashes](data/large-model-20261012/component-policy-progress.json) are
preserved. The live model is unchanged pending fresh game and deployment gates.

## Fresh component diagnostic

Seed 41271 samples 100 distinct closed-component positions per board on 4×4,
5×5, 10×10, 12×12 and 14×14. Every canonical teaching-dataset state is
excluded; overlap with other historical training is not measured. No diagnostic
selects an intermediate snapshot. [Complete diagnostic](data/large-model-20261012/component-policy-diagnostic.json).

On 10×10, raw best-margin move accuracy is 37% for retained weights, 86% for
the frozen iteration-three starting model, 82% for critic-only update 300 and
94% for policy-and-value update 300. Mean box regret is 3.96, 0.56, 0.72 and
0.24 respectively. The taught policy increases mass on optimal moves from
62.77% to 72.32% relative to its starting model. Outcome classification is
50%, 73%, 82% and 82%, using the recorded ±0.1 draw threshold.

Raw move accuracy also improves relative to the starting model on 5×5
(88% to 94%) and 14×14 (77% to 91%). These are specialized endgame
diagnostics, not complete matches or proof of arbitrary-board strength.

## Independent low-work policy arm

The policy diagnostic motivates testing the same fixed update-300 weights with
one candidate simulation against 1,024 reference simulations. With no noise
or move reduction, one simulation selects the highest network prior. Both
sides keep the common 12-edge root aid. This is an explicitly unequal-work
comparison of the taught policy against retained search, not a test of the
value of search with identical weights.

Use 12 paired-seat 10×10 games, fresh seed 23001, without deadlines, reuse,
proofs, structural aids, pruning, capture extension or rollouts. A screen score
above 50% requires 40 fresh games, seed 210001, with score above 50% and paired
bootstrap lower bound above 50%. The separate zero-first-play control uses
20 games, seed 310001, and requires at least 50% score. Numerical and web
deadline checks and smaller-board matches remain required before release.
The 1,024-simulation taught-model arm keeps its separate predeclared protocol.

### Low-work screen: rejected

The eight retained games score 1 win and 7 losses (12.5%, paired interval
0–37.5%), mean margin −27.75 boxes. Candidate decisions average 9.83 ms
above the common root-aid threshold, maximum 17.36 ms across all decisions.
The policy is fast, but it loses to retained search in full games.

The planned 12-game screen stops because even four further wins yield only
41.67%, below the strictly-above-50% gate. [Receipt and optimistic bound](data/large-model-20261012/policy-one-screen.json),
[legal replay validation](data/large-model-20261012/policy-one-screen-summary.json),
and [unfinished opening-pair traces](data/large-model-20261012/policy-one-screen-inflight.json)
are preserved. Metadata reconstruction verifies matching commit-cf2310f
sources and the frozen update-300 model hash. This arm is not deployed.
The same model's 1,024-simulation screen continues independently.

## Export and responsiveness checks

The [export check](data/large-model-20261012/policy-00300-export-parity.json)
passes on 738 legal positions across eight shapes, including 10×10 and
12×12, with seven padding nodes. Maximum logit difference is 3.82e−6 and
value difference 9.17e−7. Both use the structured graph encoding.

The [deadline probe](data/large-model-20261012/policy-00300-deadline-probe.json)
checks 36 supervised NumPy decisions on 4×4, 5×5 and 10×10, spanning opening,
middle, late and exact-root positions. Every decision finishes below two
seconds, maximum 1.779 seconds, with no fallback. This proves responsiveness
for these samples, not full-game strength under the deadline.

## Search screen: selected for confirmation

The 1,024-simulation screen completes 8 wins and 4 losses (66.67%, paired
interval 33.33–91.67%), with mean margin +25.83 boxes. It selects the frozen
update-300 model for the predeclared 40-game confirmation at seed 209001.
[Validated summary](data/large-model-20261012/policy-teaching-screen-summary.json)
and [complete histories](data/large-model-20261012/policy-teaching-screen.json).
The wide interval makes fresh confirmation essential.

Confirmation restarts before any completed opening pair to load an immutable
source snapshot; the same seed, weights and settings are retained. The new
receipt uses the `policy-teaching-confirm-v2` label. This restart and the
[search-runtime optimization](sparse-selection.md) are documented separately;
the taught-model confirmation does not enable the new selector.

## Fresh confirmation: rejected

The retained 26 opening-paired games score 11 wins and 15 losses (42.31%,
paired interval 23.08–61.54%), with mean margin +1.15 boxes. This fails the
promotion gate. Better closed-endgame move accuracy and a promising screen
did not establish stronger complete-game play.

The planned 40-game confirmation stops after its confidence gate becomes
unreachable. Even 14 further wins produce only 45% for the final paired
bootstrap lower bound, below the required strictly-above-50% threshold.
The [receipt](data/large-model-20261012/policy-teaching-confirm-v2.json) records
the bound and reconstructed metadata verified against the immutable
commit-317d36f source snapshot. [Legal replay validation](data/large-model-20261012/policy-teaching-confirm-v2-summary.json)
checks all retained games; [unfinished opening-pair traces](data/large-model-20261012/policy-teaching-confirm-v2-inflight.json)
are preserved separately.

The dependent zero-first-play control stops with 5 wins and 3 losses among
eight retained games (62.5%, paired interval 50–87.5%, mean margin +22.5).
It cannot rescue a candidate rejected against the stronger retained search.
Its planned count was 20; [partial receipt](data/large-model-20261012/policy-teaching-legacy-control-v2.json),
[validation](data/large-model-20261012/policy-teaching-legacy-control-v2-summary.json)
and [unfinished traces](data/large-model-20261012/policy-teaching-legacy-control-v2-inflight.json)
remain available. Neither comparison is reported as a completed planned run.
Only the matched benchmarks and their workers were stopped. The released
model remains unchanged; the next hypothesis tests a [first-play floor](first-play-floor.md).
