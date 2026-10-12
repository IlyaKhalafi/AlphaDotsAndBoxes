# Larger-board search refinement

## Objective and protocol

Improve 10×10 play against the released constant first-play reduction 0.33,
using ordinary NumPy inference. The existing model's SHA-256 is
`21d8df7a60ba770e45a35d8405e009417904e250813fb3134403fb7707ae87b0`.
No neural weights change during the initial search experiments.

Two isolated first-play rules are screened:

- `policy_mass`: subtract `0.33 × sqrt(visited policy mass)` from the parent
  neural value. Visited policy mass sums the original priors of children with
  at least one visit. This starts with a smaller penalty and increases it as
  search explores more of the policy. The probability-mass factor is inspired
  by [Lc0's implementation](https://github.com/LeelaChessZero/lc0/blob/master/src/search/classic/search.cc),
  but this arm keeps the current agent's raw parent neural value rather than
  switching to the parent's backed-up mean.
- `branching`: subtract `0.33 × min(1, sqrt(54 / legal actions))`. The penalty
  shrinks on wide nodes and equals the released setting once at most 54 actions
  remain. The anchor is the 5×5 board after six opening moves.

Each screen uses 12 games (six opening pairs), 512 full simulations per nonexact
move, no timers, seed 4501, and swapped seats. Both arms use the same openings
for comparison; their observations are correlated and are not pooled as
independent evidence. Both references explicitly use constant reduction 0.33.
The other settings remain exploration constant 1.5, exact 12-edge roots, no
noise, no tree reuse, no leaf oracle/proof propagation and deterministic argmax.
Four single-threaded CPU workers per arm keep combined match work at eight
cores. Compiled kernels are disabled.

An arm above 50% screen score may advance; a screen alone does not qualify a
release. Confirmation uses 40 fresh games at 1,024 full simulations against the
released search, seed 4601 (opening seeds 4601–4620), with no move timer. Require
score above 50% and the lower 2.5th percentile of a 20,000-resample paired-opening
bootstrap (seed 4199) above 50%. Run a separate 20-game control against the older
zero-first-play search, seed 4701, requiring at least 50% score. Full legal
replays and actual work budgets must be checked. Confirmation settings are
fixed before the fresh matches, including any combined changes.

Changes will remain experimental until those checks pass. Before deployment,
check 4×4 and 5×5 behavior and actual large-board web settings. Larger-board
testing does not require a two-second move limit; the live service retains its
existing deadline until a concrete validated change is ready.

## Opening diagnostics

At 1,024 simulations on one independent 10×10 opening (diagnostic seed 4501 plus
board size), constant reduction 0.33 visited one of 214 root actions, with median
endpoint depth 71. Policy-mass scaling visited 92 (median depth 2, maximum 9);
branching scaling visited 183 (median depth 3, maximum 99). These are illustrative
positions, not strength measurements.

[Full allocation receipt](data/large-search-20261012/allocation-screen.json).

## Results

The branching screen completed with 7 wins, no draws and 5 losses: 58.33%
score, paired-bootstrap 95% interval 33.33–83.33%, mean box margin +5.33.
Every one of its 2,424 nonexact decisions completed 512 simulations. Candidate
mean time was 1.036 seconds and maximum 4.064 seconds; reference mean was 0.995
seconds and maximum 5.329 seconds. These trajectories differ, so elapsed times
are not a controlled speed comparison. All 2,568 post-opening moves replay
legally to the recorded scores.

[Full branching screen](data/large-search-20261012/branch-screen.json) and
[validated summary](data/large-search-20261012/screen-summary.json).

No larger-board improvement is claimed from these screens alone.

The policy-mass screen completed with 8 wins, no draws and 4 losses: 66.67%
score, paired-bootstrap 95% interval 50.00–83.33%, mean box margin +7.67.
All 2,424 nonexact decisions completed 512 simulations; all 2,568 post-opening
moves replay legally. Candidate mean time was 1.373 seconds and maximum 3.400;
reference mean was 1.250 seconds and maximum 5.220. Again, these are different
trajectories rather than a controlled timing comparison.

[Full policy-mass screen](data/large-search-20261012/mass-screen.json).

Policy-mass scaling is selected for the first confirmation because it has the
higher score in the completed isolated first-play screens. Before opening the
fresh results, its fixed settings are reduction 0.33, mode `policy_mass`, 1,024
simulations, exploration constant 1.5, exact 12-edge roots, and no structural
solver, proof propagation, reuse, noise or timer. The confirmation and older
reference control use seeds 4601 and 4701 as declared above. Four confirmation
workers and two control workers overlap the final two endgame screening pairs,
keeping active match work at eight cores. The independent endgame screen does
not change this candidate's confirmation settings.

## Exact endgame refinement

The structural solver now collapses collections of closed chains of length at
least three and even loops of length at least four into complete-turn choices:
take all offered boxes, or keep control by leaving two boxes in a chain or four
in a loop. It also recognizes one partially captured component. Short closed
chains and multiple open components retain the existing general exact solver.
This restricted recurrence follows the control alternatives in
[Buzzard and Ciere (2014)](https://math.colgate.edu/~integers/og8/og8.pdf).

Correctness checks compare exact margins and every optimal action against full
edge-subset minimax on all recognized 2×2 positions, random 3×3 positions, and
all 1,024 descendants of a mixed five-box chain/four-box loop. A dedicated
position checks that leaving two boxes can beat immediate captures. A synthetic
10×10 position containing 25 independent four-box loops is solved in 26
structural nodes, including all 100 optimal opening actions, in 0.020 seconds
with the deployment's pure NumPy environment. This is a correctness and speed
diagnostic, not a playing-strength result.

[Synthetic-position receipt](data/large-search-20261012/loony-diagnostic.json).

After the first-play screens, a separate endgame arm will use constant reduction
0.33 plus structural solving and proof propagation, against released constant
0.33 alone. Its screen uses 12 paired-seat 10×10 games, 512 simulations as a
ceiling, seed 4801, and no timers. Exact solutions can finish before that ceiling;
both actual simulations and solve hits must be reported. Any further combination
with a first-play rule requires its own fresh screen before confirmation.

The endgame arm completed with 6 wins and 6 losses, mean margin +3.83 boxes.
Each opening pair scored exactly 50%, making the empirical paired-bootstrap
interval degenerate at 50%; this does not establish statistical equivalence.
The candidate recorded 5,933 structural solve hits and 15,705 propagated proof
nodes on positions with more than 12 remaining edges, completing fewer than
512 simulations on 179 of 1,227 such decisions. All reference decisions in that
range completed 512 simulations. Exact game knowledge is additional work, so
this is a comparison at the same simulation ceiling, not identical computation.

[Full endgame screen](data/large-search-20261012/endgame-screen.json).

## Decision audit and next isolated screen

An offline audit uses up to 50 ms per structural solve on recorded positions
with 13–100 remaining edges. It checks full legal replays first and excludes
positions whose exact solve times out. On the policy-mass screen, 114 candidate
and 109 reference positions were solved; seven and five choices respectively
lost optimal box margin. None of these solved decisions changed a forced win
into a draw/loss. The endgame arm had no suboptimal choices in 55 solved
candidate positions; its reference had two in 41 solved positions, again with
no lost forced wins. Timeouts and different trajectories limit comparisons.
These observations suggest earlier decisions are the more useful next target.

[Full audit, including action-level regrets](data/large-search-20261012/endgame-audit.json).

A separate screen changes only the parent-value source of the policy-mass rule:
use the backed-up parent mean once it has visits, falling back to the network
value before the first visit. This is closer to the first-play equation used by
Lc0. The screen uses 12 paired-seat 10×10 games, 512 full simulations, seed 5201,
two CPU workers, and the released constant-0.33 reference. Oracle, proof, noise,
reuse and move timers remain off. Six active confirmation/control workers plus
this screen keep match work at eight cores. It does not alter the already
running raw-parent confirmation. If it advances, its own fresh confirmation
and older-reference control use seeds 5301 and 5401.

On the same diagnostic opening used earlier, the backed-mean policy-mass rule
visited 84 of 214 root actions at 1,024 simulations, median endpoint depth 4 and
maximum 11. Its raw-parent counterpart visited 92, median depth 2 and maximum 9.
Constant reduction with the backed mean still visited only one root action,
median depth 77. The broader/deeper allocation is a mechanism observation;
playing-strength results are assessed separately.

[Parent-value allocation diagnostic](data/large-search-20261012/allocation-parents.json).

The next isolated screen retains the candidate's tree between decisions instead
of starting a fresh planner on each move. It uses constant reduction 0.33 with
the network parent, no structural solver/proof/noise/timer, 512 new simulations
per nonexact move, 12 paired-seat 10×10 games and seed 5501. The reference retains
the released fresh-tree behavior at 512 simulations. Two workers start after
the backed-mean screen finishes. Inherited visits are reported separately from
new simulations; they represent reused prior work, not extra simulations in the
current decision. Fresh confirmation/control, if needed, use seeds 5601/5701.
This direct match path actually retains planners. The supervised web worker
still creates a new planner per request, so the benchmark rejects supervised
candidate reuse until that serving capability exists.

The backed-mean screen finished with 4 wins and 8 losses, 33.33% score,
paired-bootstrap interval 8.33–66.67%, mean margin −18.67 boxes. All 2,424
nonexact decisions completed 512 simulations and all histories replay legally.
It does not advance. The raw-parent candidate's confirmation remains unchanged;
the tree-reuse screen has started on its separately declared openings.

[Full backed-mean screen](data/large-search-20261012/mean-screen.json).

A further allocation diagnostic varies constant reduction without changing the
weights or simulation budget. At 1,024 simulations on the same 10×10 opening,
reductions 0.33, 0.66 and 0.90 reached median endpoint depths 71, 98 and 99
(maxima 102, 113 and 116). Each still visited only one root action and selected
the same move. Increasing the penalty deepens this example but does not resolve
its lack of alternatives. This diagnostic alone does not justify a match arm.

[Constant-penalty diagnostic](data/large-search-20261012/allocation-reductions.json).

## Root shortlist and sequential halving

The next acting experiment starts with the 16 highest-prior legal moves. It
allocates equal visits within a round, ranks the moves by log prior plus
`(50 + maximum root-child visits) × parent-perspective mean value`, then keeps
half the shortlist and repeats. Interior nodes retain constant-reduction 0.33
PUCT. The final policy selects the best evaluated finalist. This deterministic
root variant is inspired by
[Danihelka et al. (ICLR 2022)](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/gumbel-alphazero.pdf);
it does not implement Gumbel sampling or replace interior-node search. It is
currently an acting experiment: self-play training, proof propagation and tree
reuse are deliberately excluded from this mode.

On the same diagnostic opening at 1,024 simulations, it visited 16 root moves,
median endpoint depth 48.5, maximum 107, and selected a different move. This
addresses the allocation tradeoff directly, but says nothing yet about wins.
The optional root observer records fresh trees without enabling reuse.

[Full shortlist diagnostic](data/large-search-20261012/allocation-halving.json).

Its screen starts after the tree-reuse screen, with two workers, 12 paired-seat
10×10 games, 512 simulations, seed 5801, and released constant-0.33 search as
reference. Width 16 and the ranking constants above are fixed before the games.
No timers, noise, oracle, proof or reuse are enabled. If it advances, fresh
40-game confirmation and 20-game older-reference control use seeds 5901/6001
at 1,024 simulations. The raw-parent confirmation remains independent.
