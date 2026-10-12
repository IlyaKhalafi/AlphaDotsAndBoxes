# Checkpoints after 1,200 additional games

Protocol recorded before full-game testing, 2026-10-12.

The exact-policy teacher fails fresh confirmation, and the first-play floor
fails its full 12-game screen (5–7). The next model comparison freezes
iteration 25 of both ongoing self-play runs. Each snapshot contains 1,200
additional mixed-board games, rather than selecting the latest file while
matches run. These are independent candidate screens against retained weights.

Both runs use the recorded 4×4, 5×5, 8×8, 10×10 and 12×12 sampling list.
One adds component features; the other keeps the original encoding. Equal
game counts do not make this a controlled causal architecture comparison:
their trajectories differ and the component worker streams restarted at
iteration 13. No exact-policy fine-tuning is applied to either snapshot.

Use 12 paired-seat 10×10 games per candidate, 1,024 simulations, fresh
seeds 27001 (component) and 28001 (original encoding). Both sides use
constant-0.33 search with the validated cached selector and the common
12-edge root aid. No first-play floor, timer, structural solver, proof,
pruning, capture extension, rollout or tree reuse is enabled. Load the
immutable package and script snapshot used by the floor screen.

If both scores exceed 50%, select the higher score and prefer the component
snapshot on a tie. The frozen winner needs 40 fresh games, seed 213001,
with score above 50% and paired bootstrap lower bound above 50%. A separate
20-game zero-first-play control uses seed 313001 and requires at least 50%
score. Smaller-board, numerical-export and actual web-setting checks remain
necessary before release. Ongoing training keeps its existing eight-hour caps.

Frozen NumPy hashes are
`15e0230ea9cad3572c742aaf157a782f9bb9d3deb136e5daecca35b481712463`
(component) and
`373152b489f73aed95746281bf2b27cb147bd0982f815a4fc2d8405adfb1dad4`
(original encoding). Both screens have started with the above settings.

## Completed screens

The component snapshot wins 8–4 (66.67%, paired 95% interval 50–83.33%),
with mean margin +24 boxes. The original-encoding snapshot wins 7–5
(58.33%, interval 33.33–83.33%), with mean margin +20.5. The predeclared
rule selects the component snapshot for fresh confirmation; neither result
establishes a large-board improvement by itself.

[Legally replayed summaries](data/large-model-20261012/later-selfplay-screen-summary.json)
include both screens and the independent deeper-search screen. The selected
component snapshot now runs the specified 40-game confirmation and 20-game
zero-first-play control with unchanged weights and search settings. These
CPU evaluations use six and two workers respectively, after GPU training ends.
The deployed agent remains unchanged pending the release gates.

## Release checks

Before reading fresh confirmation outcomes, smaller-board validation is fixed
to 20 games each on 4×4 and 5×5, fresh seed 214001. Both sides use the web's
constant-0.33 search, fresh trees, 8,192-simulation ceiling and 1.75-second
internal deadline. Supervise the candidate with the web's 1.9-second deadline;
leave the optional cached selector disabled, matching the deployed setting.
Require at least 50% score on each board and every candidate response below
two seconds. These samples are regression checks, rather than proof of superiority
on smaller boards. A candidate that fails large-board confirmation is not released.

The selected snapshot already passes [NumPy parity](data/large-model-20261012/component-00025-export-parity.json)
on 738 legally reached positions across eight shapes, including 10×10 and
12×12 with padding. Maximum absolute logit/value errors are 3.82e−6 and
8.34e−7. The [deadline probe](data/large-model-20261012/component-00025-deadline-probe.json)
passes all 36 supervised decisions, maximum 1.761 seconds, with no fallback.

On the prior [reachable 14×14 loop diagnostic](data/large-model-20261012/component-00025-representation.json),
the component snapshot separates opposite-outcome states by 0.03898 in value,
compared with one float32 rounding step for the retained representation. Its
predictions remain negative for both states (−0.0475 and −0.0865), so this
demonstrates sensitivity to the additional information, not solved endgames
or full-game strength.

## Diagnostic: first 12 fresh games

The first 12 complete confirmation games finish 5–7. This partial result is
not a release decision. [Exact endgame checks](data/large-model-20261012/component-00025-endgame-diagnostic.json)
cover 183 candidate and 262 reference decisions where the chain-and-loop
oracle returns a complete answer. Each side makes five locally suboptimal
margin choices; none reduces the available win/draw/loss outcome. One timed-out
oracle call is excluded. These local regrets must not be added up as predicted
match losses.

At the first successful oracle answer in each game, 5–65 edges remain. Every
eventual candidate loss is already losing against perfect continuation at that
point. The diagnostic supports examining earlier decisions, rather than assuming
more exact play only in recognized late positions will repair these losses.
It says nothing about unrecognized positions or other openings.

`scripts/inspect_endgame_choices.py` can reproduce the first-12 subset from the
final confirmation receipt with `--limit 12`. The canonical game-subset hash
identifies that input independently of the containing receipt; oracle coverage
can vary with scheduling when a call reaches its 50-millisecond limit.
Eight focused tests cover legal replay, a known win-to-loss mistake, incomplete
histories, invalid budgets and timeout exclusion.

## Fresh confirmation: rejected

The retained 26 complete opening-paired games finish 13–13 (50%, paired 95%
interval 30.77–69.23%), with mean margin exactly zero. Even 14 further wins
would produce only a 50% lower confidence bound for the fixed final 40-game
bootstrap, which cannot pass the required strictly-above-50% gate. The run
therefore stops as a rejection. This bound uses all known complete pairs;
workers may finish them out of original index order. Their original indices
remain in the receipt, and every retained pair is legally replayed.

The dependent zero-first-play control stops at 4–4 over eight complete games
(50%, interval 12.5–87.5%, mean margin +18.75). It cannot promote a candidate
that fails the primary comparison. No smaller-board release matches are run.

[Validated summaries](data/large-model-20261012/component-00025-confirm-summary.json),
[primary receipt and stopping proof](data/large-model-20261012/component-00025-confirm.json),
[primary unfinished traces](data/large-model-20261012/component-00025-confirm-inflight.json),
[control receipt](data/large-model-20261012/component-00025-zero-control.json),
and [control unfinished traces](data/large-model-20261012/component-00025-zero-control-inflight.json)
preserve the evidence. Settings are reconstructed from the launch command and
source hashes verified against the immutable `fpu-floor-v1` snapshot; the exact
benchmark start timestamp is unavailable and recorded as null. Only these owned
benchmark processes and workers were stopped. The deployed weights remain the
retained model, and the [final-checkpoint protocol](final-selfplay-models.md) is
now activated while the independent deeper-search matches continue.
