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
