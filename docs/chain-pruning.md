# Conservative capture-chain pruning

Protocol recorded before full-game testing, 2026-10-12.

[Barker and Korf](https://content.jbarker.net/papers/Barker_Korf_Dots_And_Boxes.pdf)
reduce opened-chain choices to taking the available boxes or preserving control
with a two-box handout; opened loops use a four-box handout. Our conservative
implementation acts only when one capture region is detected. It follows
degree-one/two boxes and treats higher-degree junctions as path outlets.
Other positions retain every legal move.

Longer paths retain a capture. At the final two-box path or four-box opened
loop, search retains a capture and the handout edge. Both alternatives receive
an initial visit, preventing a tiny learned handout prior from excluding it.
No exact value is claimed from this move reduction.

Exhaustive 2×2 comparisons and 144 sampled 3×3, 4×4 and 5×5 positions with 13
remaining edges verify that each reduction retains an exact best-margin move.
The targeted chain, search and acceleration suite passes 36 tests. This is
correctness evidence, not strength evidence.

The screen uses retained weights on both sides, 12 paired-seat 10×10 games,
1,024 simulations, seed 20001, constant-0.33 search and the common 12-edge root
aid. Candidate search additionally sets `chain_pruning=true`. No structural
oracle, proof propagation, tree reuse, rollout values or timer is enabled.
This isolated domain-pruning screen runs independently of later-model screens;
the deeper-budget contingency remains a separate experiment.

A score above 50% selects frozen settings for 40 fresh games, seed 206001.
Promotion requires score above 50% and paired bootstrap lower bound above 50%.
The separate zero-first-play control uses 20 games, seed 306001, and requires
at least 50% score. Smaller-board and actual web-setting checks follow before
release. Retained weights have SHA-256
`21d8df7a60ba770e45a35d8405e009417904e250813fb3134403fb7707ae87b0`.

## Screen result

The complete screen scored 8 wins, 1 draw and 3 losses (70.83%), with mean
margin +21.33 boxes. Its paired bootstrap interval is 54.17–87.5%. Candidate
search averaged 1.69 seconds per decision above the common endgame threshold;
the reference averaged 1.96 seconds. These are untimed fixed-work games and
the slowest candidate decision took 9.61 seconds.

The 40-game confirmation started with the unchanged settings and fresh seed
206001. This screen alone does not authorize release. Every game was legally
replayed by the [summary](data/large-model-20261012/later-screen-summary.json),
and the [complete histories](data/large-model-20261012/chain-pruning-screen.json)
retain per-move work and timing. The full regression suite passed 189 tests.

A [choice diagnostic](data/large-model-20261012/chain-choice-diagnostic.json)
replays the screen's 2,568 decisions. The reduction applied at 497 candidate
positions and 331 reference positions. The reference chose outside the
retained set 54 times; the candidate never did. Among the candidate's 65
capture-or-handout positions, it selected a noncapture 46 times. This shows
the new search actually exercises control-preserving options. It does not
quantify the score regret of individual reference decisions.

The [deadline probe](data/large-model-20261012/chain-deadline-probe.json) checks
36 supervised NumPy decisions across opening, middle, late and exact-endgame
positions on 4×4, 5×5 and 10×10 boards. All finished below two seconds
(maximum 1.774 seconds), without a fallback. This checks responsiveness;
the fixed-work strength screen does not establish strength under that deadline.

Before a release, the smaller-board check uses 20 fresh paired games per board
on 4×4 and 5×5, seed 208001, with the ordinary Deep settings: 1.75-second
internal search, 8,192 ceiling and 12-edge root aid. Candidate responses use
the 1.9-second supervisor; reference search uses the same internal budget.
Require at least 50% score separately on each board and every candidate
decision below two seconds. This check runs while the large-board confirmation
continues and cannot substitute for it.

## Smaller-board Deep results

The completed checks pass their non-regression gates: 4×4 scores 13 wins,
4 draws and 3 losses (75%, paired interval 57.5–90%); 5×5 scores 14 wins
and 6 losses (70%, paired interval 50–90%). Mean margins are +2.7 and +5.7
boxes. All 955 candidate decisions finish below two seconds, maximum
1.781 seconds, with no supervisor fallback.

The [validated summary](data/large-model-20261012/chain-small-deep-summary.json)
and [full histories](data/large-model-20261012/chain-small-deep.json) retain
both sides' actual search counts. These timed results can vary with scheduling.
They compare against the retained constant-0.33 agent, a different baseline
from the earlier release's zero-first-play comparison. They do not establish
a 10×10 gain, and the large-board confirmation remains necessary.

## Large-board confirmation: rejected

Fresh confirmation retains 24 complete paired-seat games: 8 wins and 16 losses
(33.33%, paired interval 16.67–50%), mean margin −18.17 boxes. It fails the
promotion gate. The initial screen's improvement did not generalize to these
new openings, despite the successful smaller-board timed checks.

The planned 40-game run stops because the confidence gate is mathematically
unreachable. Counting all 16 unrecorded games as wins would yield 60% score,
but only 42.5% for the final paired-bootstrap lower bound. The required lower
bound is strictly above 50%. With the fixed bootstrap resample indices, any
other future outcome can only reduce that bound. No favorable partial score
is used for promotion.

The [receipt](data/large-model-20261012/chain-pruning-confirm.json) records this
optimistic bound, planned count and reconstructed metadata provenance;
[legal replay validation](data/large-model-20261012/chain-pruning-confirm-summary.json)
checks all 24 retained histories. [Unfinished opening-pair traces](data/large-model-20261012/chain-pruning-confirm-inflight.json)
are preserved separately. Only the matched benchmark process and its own
workers were stopped. Live search and weights remain unchanged.
