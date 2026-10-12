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
