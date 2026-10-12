# Final checkpoints from the bounded runs

Protocol fixed before full-game testing, 2026-10-12.

If the selected component iteration-25 snapshot fails fresh confirmation,
screen the final saved checkpoints from the two completed training runs:
component iteration 34 (1,632 additional games, 218,590 positions) and original
encoding iteration 36 (1,728 games, 231,778 positions). These are the final
checkpoints, rather than intermediate snapshots selected by training loss.
Their different trajectories, game counts and component sampler restart mean
this is not a causal architecture comparison.

Each snapshot receives 12 paired-seat 10×10 games against retained weights,
1,024 simulations on both sides, seeds 30001 and 31001 respectively. Use
constant-0.33 search, the cached selector and the common 12-edge root aid,
without timers, pruning, structural aid, proofs, reuse or rollouts. Keep the
same immutable `fpu-floor-v1` package and benchmark source.

If both score above 50%, select the higher score, preferring the component
snapshot on a tie. The frozen winner needs 40 fresh games at seed 215001,
score above 50% and paired-bootstrap lower bound strictly above 50%. Its
separate 20-game zero-first-play control uses seed 315001 and requires at
least 50%. Release also requires NumPy export parity, actual web inference
checks and 20 games each on 4×4 and 5×5 at fresh seed 216001 with the web's
1.75-second internal and 1.9-second supervised candidate deadlines. Smaller
board scores must each be at least 50%, with all candidate responses below
two seconds. Keep the cached selector disabled in those timed release checks,
matching the deployed setting.

No further GPU training is started by this protocol. The independent
eightfold-search confirmation continues with its previously frozen settings.
Passing a short screen alone cannot change the live agent.

Frozen NumPy SHA-256 hashes:

- Component iteration 34: `bd6c888f47c5c75eb093dc16ef692e55792d10c9f74b9774949ca733b74544a7`.
- Original encoding iteration 36: `754f6a9d12440decdf67da322625f49777c82670103067ce5e2609412a97776b`.

The contingency is activated after iteration 25 fails confirmation at 13–13
and its confidence gate becomes unreachable. Both final-checkpoint screens
start with the frozen settings above, using three CPU workers each.

## Screen results and confirmation

The final component snapshot wins 8–4 (66.67%, paired 95% interval
33.33–91.67%), mean margin +15.67 boxes. The original-encoding snapshot
loses 5–7 (41.67%, interval 8.33–75%), mean margin −9.67. The component
snapshot advances under the recorded rule; the original-encoding snapshot is
rejected. [Legally replayed summaries and full histories](data/large-model-20261012/final-selfplay-screen-summary.json).

The component checkpoint starts the prescribed 40-game fresh confirmation
and 20-game zero-first-play control, using six and two CPU workers respectively.
Both use the immutable retained-model copy with the same published SHA-256,
so a future deployment cannot change the opponent for pending worker tasks.
Package and benchmark source remain the frozen `fpu-floor-v1` snapshot.
The live model remains unchanged pending confirmation and release checks.

The [NumPy parity check](data/large-model-20261012/component-00034-export-parity.json)
passes on 738 reachable states across eight board shapes, including 10×10
and 12×12 with seven padding nodes. Maximum absolute logit/value differences
are 4.05e−6 and 6.56e−7, within the recorded absolute/relative tolerances.
The [deadline probe](data/large-model-20261012/component-00034-deadline-probe.json)
passes all 36 supervised decisions on 4×4, 5×5 and 10×10, maximum 1.770
seconds with no fallback. These checks establish numerical agreement and
responsiveness, rather than strength under the web time limit.
