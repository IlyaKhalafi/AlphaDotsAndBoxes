# Component-value teaching experiment

Protocol recorded before teaching updates, 2026-10-12.

The first component model scored 79.17% in its 512-simulation screen. Its first
12 fresh confirmation games at 1,024 simulations have instead produced four
wins and eight losses. Confirmation continues; this checkpoint has not passed
the release gate. Training loss and the early screen cannot establish strength.

The next isolated change teaches exact values on larger chain/loop endgames.
The starting weights are component self-play iteration three, SHA-256
`7185d4f820cb2dba33ed9bfead8ce3496572ebde1403df03ffce15cd5ffa505e`.
Its trusted replay snapshot contains 21,561 positions, SHA-256
`661ac26130e77516ebb227ac58c7bff147ac2fb91c6f85649fc1a700a5037e08`.
Both snapshots are frozen independently of ongoing self-play.

Generate 2,048 distinct legally reached endgames, optionally followed by zero
through six optimal moves to include captures, with seed 41071. Use the same
board-size sampling list as the replay. The structural oracle supplies exact
final win/draw/loss labels, including scores already earned. Another 2,048
distinct earlier replay positions anchor the starting model's predictions.
No confirmation history or human move supplies a target.

Each proposed exact position gets a 50 ms cooperative oracle budget. Slow
multi-front continuations are skipped, and completed exact targets are retained
instead of solved again after cache eviction. Timeout counts are recorded;
cutoff values never become labels. This bounds generation work and makes the
resulting sampling bias explicit.

This first arm changes **value targets only**: policies on all positions remain
the frozen starting network's policies. RLlib uses 300 updates, batch size 512,
25% exact positions, learning rate 0.00003 and an 8 GiB allocator limit.
CUDA updates retain 50% pacing. Private Ray sessions use the common learner
initializer. The two existing self-play runs retain their settings and limits;
the added learner raises aggregate allocator ceilings from 54 to 62 GiB.

Save updates 25, 100 and 300. Screen each in that order using 12 paired-seat
10×10 games at **1,024** simulations, seeds 12001, 13001 and 14001. Both sides
use constant-0.33 search, fresh trees and the same 12-edge root aid. No time
limit, structural oracle, proof or reuse is enabled in these matches. The larger
screen budget matches confirmation after the first model's weaker fresh result.
Choose the highest score, preferring the earlier update on a tie. A candidate
must score above 50% to proceed.

The selected frozen snapshot needs 40 fresh games, seed 201001, against the
released constant-0.33 search, with score above 50% and paired bootstrap lower
bound above 50%. A separate 20-game zero-first-play control uses seed 301001
and requires at least 50% score. Smaller boards, numerical export and actual
web settings are checked before release. If this critic-only arm fails, a policy
teaching arm requires a separate recorded experiment and fresh seeds.

```bash
python scripts/teach.py --checkpoint runs/component-initial.npz \
  --replay runs/component-replay.pt --output runs/component-teaching.npz \
  --positions 2048 --seed 41071
python scripts/teach.py --checkpoint runs/component-initial.pt \
  --dataset runs/component-teaching.npz --output runs/component-critic \
  --steps 300 --batch-size 512 --gpu-memory-gb 8 --seed 41072
```

Replay `.pt` inputs are trusted local training state; ordinary deployment uses
NumPy checkpoints and does not load replay. The component dataset and actual
RLlib update path passed five teaching tests, including both graph encodings.

## Completed teaching and diagnostics

The dataset contains 2,048 exact positions and 2,048 anchors. Exact labels are
871 losses, 272 draws and 905 wins. Generation skipped 2,026 oracle timeouts;
the recorded 18.16 seconds measures dataset materialization, excluding position
proposal generation. [Dataset receipt](data/large-model-20261012/component-teaching-dataset.json).

All 300 RLlib updates completed in 304.60 seconds, with peak allocation
4,971.6 MiB. Final mixed value loss was 0.03165, mixed value MAE 0.07154,
and policy KL 0.00181. These mixed training targets include anchors and are
not game-strength measurements.
[All update records and settings](data/large-model-20261012/component-critic-progress.json).

A separate seed-41171 diagnostic samples 100 closed-component positions on each
of 4×4, 5×5, 10×10, 12×12 and 14×14. It excludes all canonical positions in
the supplied teaching dataset. Other historical training overlap is not measured.
On 10×10, raw value MAE is 0.882 for the retained model, 0.448 for the frozen
iteration-three starting model, and 0.391 / 0.362 / 0.323 after 25 / 100 / 300
teaching updates. Outcome classification at the recorded ±0.1 draw threshold
is 48%, 78%, 79%, 78% and 79% respectively. Value error improves without a
corresponding large classification improvement on this board.

On 12×12, 300 updates reduce value MAE from 0.588 to 0.262 and raise classification
from 62% to 86%; on 14×14 the corresponding values are 0.565 to 0.301 and 73% to
86%. Raw best-margin move selection regresses on some boards, so critic correction
is not assumed to improve full games. [Diagnostic receipt](data/large-model-20261012/component-critic-diagnostic.json).
Closed-chain opening regrets use the exact control recurrence, avoiding general
two-front enumeration. Exhaustive small-board action-value comparisons pass;
the full suite passes 155 tests.

Frozen NumPy hashes are `b59323dab2dc57583fb12c045012048279706238b369262058f8c00322419e42`
(25), `afc93186987f073707a0d2b31d7a84910b0b08490df49eb155a796302d6d68e5` (100), and
`b1c5bf3bb82a105d93bc766a3242fdcc2b532187ad43bb56821239d45ac4d8c9` (300).
Screens are launched in update order on the pre-recorded independent opening
ranges. Their fixed work budgets allow independent jobs to share available CPU
capacity; every selected model still needs fresh confirmation.

## Match screens: all rejected

The completed screens at 1,024 simulations produced:

| Teaching updates | Wins–draws–losses | Score | Paired 95% interval | Mean box margin |
| --- | --- | --- | --- | --- |
| 25 | 3–0–9 | 25% | 8.33–41.67% | −10.50 |
| 100 | 5–0–7 | 41.67% | 16.67–66.67% | −5.33 |
| 300 | 5–0–7 | 41.67% | 25–50% | +3.67 |

All retained nonexact decisions completed 1,024 simulations. Full legal replays,
scores and paired openings passed validation. None scores above 50%, so no
snapshot advances to confirmation and none replaces the served agent.
[Raw matches and validated summaries](data/large-model-20261012/critic-screen-summary.json).
The next hypothesis tests [game-rollout values](rollout-values.md), independently
of these taught weights.
