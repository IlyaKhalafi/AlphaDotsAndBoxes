# Delaying early gifts

Protocol fixed before full-game testing, 2026-10-12.

The rejected component iteration-25 confirmation contains 56 candidate and
40 retained-player moves that offer an immediate capture while a safe move
exists and no capture is already open. These are observable move properties,
not proven strategic errors: sacrificing a box can sometimes help control
the later game. Several gifts occur while dozens of safe alternatives remain.

The [26-game diagnostic](data/large-model-20261012/component-00025-confirm-diagnostic.json)
legally replays the published confirmation receipt. It also diagnoses 376
candidate and 478 reference endgame decisions, finding no outcome-reducing
move in the oracle's recognized domain. Margin mistakes still occur, and one
timed-out call is excluded. The diagnostic does not cover all earlier positions.

The next isolated search hypothesis delays those gifts. Optional
`SearchConfig.safe_moves` restricts expanded nodes to non-gifting actions only
when there are no available captures and at least one safe action. It leaves
capture and handout choices unrestricted. Exact root answers still consider
all legal actions. The heuristic cannot be combined with proof propagation;
its exclusions are not guaranteed to preserve an optimal move. It is disabled
by default and is not deployed or used in training.

If neither final self-play snapshot passes fresh confirmation, screen this
filter using retained weights on both sides, 12 paired-seat 10×10 games at
1,024 simulations each, seed 32001. Both sides use constant-0.33 search,
the cached selector and the common 12-edge aid. Only the candidate enables
safe-move filtering; no timers, structural aid, proofs, pruning, rollouts or
reuse are enabled. Freeze package and benchmark source before launch.

A score above 50% advances the unchanged settings to 40 fresh games,
seed 217001, requiring score above 50% and paired-bootstrap lower bound
strictly above 50%. A 20-game zero-first-play control uses seed 317001 and
requires at least 50%. Release also needs 20 games each on 4×4 and 5×5,
fresh seed 218001, with web deadlines (1.75 seconds internally, 1.9 seconds
for the supervised candidate), at least 50% score on each board and every
candidate response below two seconds. Keep the cached selector disabled
for timed release checks. The independent deeper-search arm continues with
its original settings. No new GPU training is required.

The package and scripts are frozen from commit `0e70928` in the
[source fingerprint](data/large-model-20261012/safe-moves-source-snapshot.json).
Both sides will read the same immutable retained NumPy snapshot, SHA-256
`21d8df7a60ba770e45a35d8405e009417904e250813fb3134403fb7707ae87b0`.
The full CPU suite passes 238 tests with two GPU tests skipped; the extended
diagnostic separately passes nine focused tests. The seeded tactical baseline
retains identical actions on four board shapes after sharing the safe-action
helper. At source freeze, full-game testing had not started for this arm.

The contingency is activated after the final component snapshot cannot clear
fresh confirmation (21–15 over 36 games, optimistic final lower bound 47.5%).
The safe-move screen starts with six CPU workers, the frozen source and
retained-model copy, and the exact settings specified above. Other prototypes
remain disabled and the live agent remains unchanged.

## Completed screen

The filter wins 9–3 (75%, paired 95% interval 58.33–91.67%), mean margin
+28.83 boxes. Both sides use the retained weights and exactly 1,024 simulations
on every neural decision. No structural solve, proof, rollout or reuse is enabled.
[Legally replayed summary and full history](data/large-model-20261012/safe-moves-screen-summary.json).

Candidate neural decisions average 1.143 seconds, versus 1.604 for the reference;
maxima are 5.422 and 5.004 seconds. This is an untimed strength screen, outside
the web deadline. Timings mix different played positions and are not a
same-position speed benchmark. The screen advances unchanged settings to the
specified 40-game fresh confirmation and 20-game zero-first-play control, using
six and two CPU workers. Neither the screen nor its interval replaces fresh
confirmation and actual web-setting checks. The live agent remains unchanged.

The [supervised deadline probe](data/large-model-20261012/safe-moves-deadline-probe.json)
passes all 36 decisions across opening, middle, late and exact-root positions
on 4×4, 5×5 and 10×10. Maximum response is 1.774 seconds with no fallback.
It uses the web deadlines and leaves the cached selector disabled. The
deadline tool accepts `--safe-moves` to reproduce this enabled-feature check.
Responsiveness alone does not establish strength or pass the release gates.
