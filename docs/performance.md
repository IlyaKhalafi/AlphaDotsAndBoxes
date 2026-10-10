# Self-play performance

Measurements on October 10, 2026 identified CPU self-play as the main wait in
the eight-hour campaign. Recent rounds used 191–245 seconds to generate 96
games, followed by roughly 22 seconds of learner updates. The VM exposes 18
CPU cores, and the original sampler already used twelve of them. Allocating
more GPU memory alone would not shorten that sampling stage.

The optimized path defers state creation until search visits a move, avoids
repeated network mode changes, caches board geometry, and encodes single search
leaves without training-batch setup. Independent games share neural calls.
Optional compiled kernels accelerate child scoring and the exact endgame
solver. Two GPU samplers reuse a captured call with fixed padded inputs; twelve
CPU samplers process batches alongside them. The model architecture is unchanged.

## Fixed-position measurements

Eight 5×5 positions, 512 simulations each, the same iteration-50 weights, one
PyTorch thread and three repeats. Values below are median wall-clock times.

| Search variant | Seconds | Speedup |
| --- | ---: | ---: |
| Original single-game CPU search | 6.582 | 1.00× |
| Optimized single-game CPU search | 4.929 | 1.34× |
| Eight-game CPU batching | 3.001 | 2.19× |
| CPU batching with compiled child scoring | 2.693 | 2.44× |
| Eight-game GPU batching | 2.316 | 2.84× |
| GPU batching with captured replay | 1.817 | 3.62× |

All tested visit policies match the original exactly. Maximum value differences
in the first GPU-batching test were below 5.5e−8. These positions do not exercise
the exact endgame aid. In a separate eight-position test with twelve edges left,
the Python solver took 0.2613 seconds and the compiled solver 0.00378 seconds:
69.2× faster, with identical score margins and optimal moves.

## Sampling and memory

A real Ray/RLlib trial with six GPU samplers took 229.5 seconds for 96 games;
captured replay reduced that to 207.3 seconds. These trials overlapped the main
CPU sampler. They demonstrate why a small inference benchmark cannot predict
whole-sampler speed on this shared GPU.

The mixed trial used twelve CPU samplers with eight games each and two GPU
samplers with sixteen games each. It generated 128 games and 5,860 positions in
112.65 seconds at 512 simulations. This is about 1.14 games per second, versus
roughly 0.4–0.5 in recent main-run rounds. The trial used seed 123 and fixed
iteration-50 weights; the main run uses seed 44 and evolving weights. Host load
also varied. Treat this as a local operational observation rather than a
controlled claim of a universal speedup.

Three disposable learner updates with batches of 40,960 completed successfully.
Peak allocation reached 65.39 GiB and peak reserved memory 69.01 GiB, below the
new 72 GiB learner cap. Each GPU sampler retains its own 1 GiB allocator cap;
CUDA contexts add some memory outside those allocator figures. The separate GPU
job stayed running. A sampled device utilization reached 99% during GPU self-play;
that reading includes the other job and is not a sustained utilization average.

The campaign resumed from full iteration-66 state: 5,056 games and 225,156
positions. The remaining budget was 16,121.48 seconds. The new configuration
collects 128 games per round and performs sixteen updates of 40,960 positions,
preserving the optimizer, replay, board distribution and 512-simulation search.
Sampler RNG streams restart when workers are recreated. The first two resumed
rounds generated 128 games each in 105.81 and 93.60 seconds, with 26.79 and
25.94 seconds of updates. Both completed all sixteen 40,960-position updates
and full resume saves, reaching 5,312 games and 236,410 positions. Their observed
sampling rate was 1.28 games per second versus 0.46 over the preceding eleven
rounds (2.80×). This is a before/after observation with evolving models and host
load. No playing-strength improvement is claimed from these speed measurements.

[Raw timings, policies, solver outcomes, trial metrics and memory receipt](data/selfplay-performance-20261010.json)
retain the measurements. The [training guide](training.md) describes building
optional kernels and running the reproducible benchmark. Ordinary deployment
still uses NumPy with a Python fallback and needs no compiler or training libraries.
