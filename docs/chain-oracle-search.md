# Chain reduction with exact endgame values

Protocol recorded before full-game testing, 2026-10-12.

Chain reduction alone fails fresh confirmation (8–16). Extending learned
values through mandatory captures also fails its screen (2–6), with the
remaining games unable to raise its planned 12-game score above 50%. The
next isolated search candidate combines the conservative chain reduction with
the existing exact degree-two endgame solver, without proof propagation.

Retain released weights on both sides. The candidate uses constant-0.33,
`chain_pruning=true`, `degree_two=true`, and `prove_results=false`.
The reference uses retained constant-0.33 search. Both use 1,024 simulations,
fresh trees and the common 12-edge root aid, pure NumPy and no move timer.
No capture-value wrapper, rollout, root halving or new model is enabled.

Use 12 paired-seat 10×10 games, fresh seed 24001. The earlier degree-two
experiment included proof propagation; this combination removes that option
and adds chain move reduction. It is a combined comparison against retained
search, not a causal estimate of either feature alone.

The oracle retains its existing cooperative caps: 50 ms at the root and
3 ms at a leaf. Only completed exact margins enter search; a timeout falls
back to the network. Oracle completion and timing may vary with scheduling
even though the simulation ceiling is fixed. The solver's bounded structural
cache persists within a worker and is independent of model weights.

A screen score above 50% freezes settings for 40 fresh games, seed 211001.
Promotion requires score above 50% and paired bootstrap lower bound above 50%.
The separate zero-first-play control uses 20 games, seed 311001, and requires
at least 50% score. Smaller-board and actual web-setting checks are required
before release. No strength improvement is assumed from exact endgame labels.
