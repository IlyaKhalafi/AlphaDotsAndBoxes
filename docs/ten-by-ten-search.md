# Untimed 10×10 comparison

## Protocol

This experiment tests whether the first-play change that improved 4×4 and 5×5
also transfers to 10×10 boxes (100 boxes and 220 edges). The earlier
[two-second experiment](two-second-search.md) checked 10×10 latency, not strength.

Both sides use the same `models/larger.npz` weights, SHA-256
`21d8df7a60ba770e45a35d8405e009417904e250813fb3134403fb7707ae87b0`.
The candidate uses first-play reduction 0.33; the reference uses the previous
zero value for unvisited children. Other search settings are shared: exploration
constant 1.5, exact 12-edge root endgames, no tree reuse, no degree-two leaf
solver or proof propagation, no noise, and deterministic maximum-policy ties.

Each nonexact decision completes **1,024 MCTS simulations**. Neither side has
a per-move time limit or a supervised response deadline. Exact roots return
directly. A simulation traverses the current search tree, evaluates one new
position (or a terminal result), and backs its value up through the tree;
it is not a complete rollout to the end of a randomly played game.

There are **20 games**, grouped into ten opening pairs. Each pair uses six
random opening moves and then exchanges candidate seats. Seed 4401 generates
opening seeds 4401–4410, independent of earlier screening and confirmation.
The checkpoint was trained on boards through 5×5; 10×10 is outside its training
sizes. This is a transfer test, not additional training.

Matches use NumPy with compiled kernels disabled, one BLAS thread per worker
and eight CPU workers. The live UI keeps its normal response deadline. Final
replays must reproduce all moves, turns and scores, and nonexact moves must
record the full shared simulation count. Report wins, draws, losses, box margins
and a 20,000-resample paired-opening bootstrap interval (seed 4199), including
negative results. This sample does not establish expert-human strength.

## Results

The run is in progress. No 10×10 superiority claim is made until it finishes.
Raw outcomes will be saved under `docs/data/ten-by-ten-20261012/`.

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
ALPHABOXES_PURE_NUMPY=1 python scripts/benchmark_engine.py \
  --candidate-engine mcts --search-options '{"fpu_reduction":0.33}' \
  --match-boards 10 --match-simulations 1024 --untimed --skip-latency \
  --games 20 --workers 8 --seed 4401 \
  --output runs/ten-by-ten.json --progress runs/ten-by-ten.jsonl \
  --move-progress-dir runs/ten-by-ten-moves
python scripts/summarize_engine.py --receipts runs/ten-by-ten.json \
  --output runs/ten-by-ten-summary.json
```

Elapsed times reflect CPU work and machine load; they do not truncate searches
or determine the move budget. Changing the simulation count can change the
relative strength of these search settings.
