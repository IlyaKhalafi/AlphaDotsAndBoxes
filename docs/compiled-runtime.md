# Optional C++ runtime acceleration

2026-10-13. The retained width-96, depth-6 network now supports C++ kernels
for inference. The checkpoint, search settings and simulation budget stay the
same. Ordinary installation still uses NumPy and requires no compiler or
training framework.

## Build and run

From the source checkout, with a C++17 compiler and Python development headers:

```bash
python -m pip install -e '.[accelerate]'
python scripts/build_acceleration.py --kernels-only
adb serve --checkpoint models/larger.npz --extended-search --port 9003
```

The extension is detected automatically. `/api/health` reports
`"native_kernels": true` when it is enabled. The build tools are not imported
during play; PyTorch and Ray are not needed. To compare with the NumPy fallback:

```bash
ALPHABOXES_PURE_NUMPY=1 adb serve --checkpoint models/larger.npz --extended-search
```

For a separate installed runtime, build with the same Python ABI and point
`--output-dir` at its `site-packages` directory. The script copies only the
requested extension, using an atomic replacement. Portable build flags are
provided for Linux, macOS and Windows; this release is tested with GCC 13.3 on
Linux. On glibc x86-64, CPU features select the portable, AVX2 or
AVX-512 implementation at runtime. Other platforms retain the portable kernel.
No machine-specific compiler target is required. An absent or incompatible
extension uses the NumPy fallback.

## What runs in C++

Matrix multiplication and exponentials continue to use NumPy. C++ combines
the surrounding operations into fewer passes through memory:

- Grid neighbor sums and the self-node contribution.
- Bias addition, negation and clipping before the unchanged NumPy exponential;
  denominator addition and division afterwards.
- Bias, residual addition, row means, variance, normalization and affine weights.

This addresses inference, the main cost of cold large-board decisions. It also
avoids replacing NumPy's vectorized exponential with a slower scalar function.
The reduction follows NumPy's
[contiguous float32 summation order](https://github.com/numpy/numpy/blob/v2.3.3/numpy/_core/src/umath/loops_utils.h.src).
Fast-math and fused multiply-add contraction are disabled; the
[build receipt](data/large-model-20261012/cpp-build-receipt.json) confirms no
fused multiply-add instructions in the extension. Buffer bindings validate
shapes, contiguity and overlap before releasing the GIL. Per-call scratch arrays
keep concurrent network calls independent. Padding and structured features
remain supported.

## Measurements

Reference source is `94dd4ae`; C++ integration is `d3a4557`. Both use retained
checkpoint `21d8df7a…`, one BLAS thread and NumPy 2.5.3. The full-budget comparison
alternates reference/candidate order across two repeats of a three-move sequence,
starting six moves into the same 10×10 position, seed 61001. Both caches retain
32,768 predictions. Every decision completes 8,192 simulations, and policies,
values, chosen actions and complete visited-tree fingerprints match exactly.

| 10×10 decision | NumPy median | C++ median | Reduction |
| --- | --- | --- | --- |
| Cold first move | 32.69 s | 24.76 s | 24.3% |
| Second move | 27.69 s | 20.72 s | 25.2% |
| Third move | 13.28 s | 11.51 s | 13.3% |

The [full-budget receipt](data/large-model-20261012/cpp-runtime-full-budget.json)
contains individual timings, source fingerprints, search settings and policies.
Alternating inference calls retain 80 samples after five warmups. On an empty
10×10 board, the network alone falls from **4.224 to 2.803 ms** (33.6%).
Middle-position inference falls from 4.191 to 2.737 ms (34.7%). Additional
[board checks](data/large-model-20261012/cpp-runtime-board-checks.json) cover
5×5, 10×10 and 12×12 at 2,048 simulations and return identical search traces.
The [forced-fallback check](data/large-model-20261012/cpp-numpy-fallback-check.json)
disables the extension and verifies unchanged NumPy outputs and search traces.

Two repeats and a few positions on shared hardware do not establish a general
latency guarantee. Cached decisions contain more Python search work, so their
gain is smaller. The cold decision still exceeds two seconds; this is a code
speed improvement, not a new model or a claim of stronger play.

The complete suite passes **341 tests** with the compiled backend available.
Kernel tests include rectangular boards, arbitrary widths, padding, structured
features, signed zeros, clipped activations, unchanged inputs and rejection of
aliased or malformed buffers. The clean ordinary wheel was checked before
enabling acceleration: it contains no binary extension, PyTorch, Ray, Gymnasium
or Cython. Only the C++ kernel extension is added to the serving environment.
The [live API receipt](data/large-model-20261012/cpp-runtime-live-probe.json)
records port 9003's source and checkpoint checks and actual request timings.
On a fresh empty 10×10 board, extended Deep completes all 8,192 simulations in
**31.31, 12.07 and 12.28 seconds**, without deadline fallbacks. Ordinary 5×5 Deep
requests remain capped and take 1.766–1.768 seconds, completing 1,046, 1,354 and
1,715 simulations in the measured moves. These live checks have no concurrent
owned training or benchmark job; other shared-machine workloads remain present.

See the [3×3–10×10 timing comparison](board-size-latency.md) for measurements
on every square board size, including a fixed search budget and live UI timings.

To repeat the comparison against a trusted checkout of `94dd4ae`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
python scripts/benchmark_runtime.py \
  --reference-source /path/to/reference/src --sizes 10x10 \
  --simulations 8192 --search-repeats 2 --inference-repeats 80 \
  --output cpp-runtime-comparison.json
```
