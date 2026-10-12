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

## Screen: rejected

The retained eight complete opening-paired games score 2 wins and 6 losses
(25%, paired interval 0–50%), with mean margin −18 boxes. The value extension
is exercised at 20,634 evaluated positions, advancing through 112,002
mandatory moves and reaching 61 exact terminal finishes. It does not improve
these full games.

The planned 12-game screen stops because even four further wins produce only
50% score, below the strictly-above-50% gate. The [receipt](data/large-model-20261012/forced-capture-screen.json)
records the stop bound and reconstruction provenance, with hashes verified
against the matching commit-cf2310f sources. [Legal replay validation](data/large-model-20261012/forced-capture-screen-summary.json)
checks all eight retained games. [Unfinished opening-pair traces](data/large-model-20261012/forced-capture-screen-inflight.json)
are preserved separately. This wrapper is not deployed.

The full suite passes 193 tests. The next search experiment uses
[exact chain-endgame values](chain-oracle-search.md) instead of this learned
continuation value; the taught-policy model remains a separate experiment.
