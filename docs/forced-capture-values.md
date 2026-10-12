# Values after mandatory captures

Protocol recorded before full-game testing, 2026-10-12.

The chain-pruning screen was promising, but its first eight fresh confirmation
games scored only 3–5. Both the full confirmation and smaller-board tests
continue. No large-board improvement is claimed from that screen.

This independent follow-up retains the same conservative chain reduction and
extends a value prediction through single-choice captures. The network judges
the next genuine choice position. Extension stops at a two-box handout,
four-box loop handout, multiple capture regions or an ordinary choice. A
terminal continuation returns its exact outcome. No random continuation,
learned move sequence, new model or endgame oracle is added.

The enclosing search still records individual physical moves. It receives a
one-hot prior at a mandatory capture, otherwise the unchanged network policy.
Memoization avoids repeating an extension from the same state. Tests check
unchanged ordinary predictions, legal four-capture extension up to a two-box
handout, current-player value perspective, cache invalidation and exact
terminal outcomes.

Use retained weights on both sides, pure NumPy, 12 paired-seat 10×10 games,
1,024 simulations, fresh seed 21001. Candidate search uses constant-0.33 and
chain pruning, with `--candidate-forced-captures`; reference uses retained
constant-0.33 search. Both retain the 12-edge root aid and fresh trees, without
timers, degree-two oracle, proof propagation or rollouts. This compares the
combined reduction and value extension against the retained agent; it does
not isolate the extension against chain pruning alone.

A screen score above 50% freezes settings for 40 fresh games, seed 207001.
Promotion requires score above 50% and paired bootstrap lower bound above 50%.
The separate zero-first-play control uses 20 games, seed 307001, and requires
at least 50% score. Smaller-board and web-setting checks remain necessary.
