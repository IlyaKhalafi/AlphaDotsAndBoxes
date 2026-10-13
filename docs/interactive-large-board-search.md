# Speed up large-board search

2026-10-13: a live fresh 10×10 Deep browser request takes 47.68 seconds;
a separate API request takes 56.69 seconds. Both complete 8,192 simulations.
The longer search's confirmed strength gain does not make this latency suitable
for ordinary play.

The proposed UI time-limit reduction was withdrawn before deployment. Keep
the confirmed deeper-search profile and optimize its actual work instead.
Profile the cached search, reduce NumPy inference and traversal overhead,
then consider a distilled network with fresh strength tests. Faster moves
obtained only by cutting the search budget are not the desired solution.

## Next isolated speed experiment

Keep retained weights and the interactive deadline; test the already implemented
cached child selector against ordinary child selection on 12 paired-seat 10×10
games, seed 401001. Both use constant first-play reduction 0.33, an 8,192-work
ceiling, 1.75 seconds internally, fresh trees and the common 12-edge root aid.
The candidate has a supervised 1.9-second response deadline. The reference has
the same internal time cap; record its complete wall-clock responses too.
No root safety preference, pruning, structural solving, proofs, rollouts or tree
reuse is enabled. Use six CPU workers, one BLAS thread each, the immutable
`selection-v1` source snapshot and the retained immutable NumPy weights.

This tests whether the selector's lower overhead improves play within the short
budget. Its prior fixed-work parity and timing tests are not a timed-game
strength result. A screen score above 50% advances unchanged settings to
40 fresh games, seed 402001. Require score above 50%, paired-bootstrap lower
bound above 50%, and every candidate response below two seconds. Keep the
cached selector experimental until these gates and deployment checks pass.
