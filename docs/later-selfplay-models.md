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
