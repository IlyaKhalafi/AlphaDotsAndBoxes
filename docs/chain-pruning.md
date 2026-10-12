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
