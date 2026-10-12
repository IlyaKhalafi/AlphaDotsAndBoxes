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
