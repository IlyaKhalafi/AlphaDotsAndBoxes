# Cache unvisited search choices

An instrumented 10×10, 1,024-simulation profile made 12.41 million sibling
scoring calls. Selection consumed 7.19 of 11.71 profiled seconds; the network
forward path consumed 3.73 seconds. Profiling overhead means these are not
ordinary decision timings. [Call profile](data/large-model-20261012/large-search-call-profile.json).

With constant first-play values, every unvisited child has the same value and
denominator. Its score is monotonic in its prior. Optional
`sparse_selection=true` caches prior groups, ranks visited children and the
leading unvisited groups, and preserves original dictionary order for equal
floating-point scores. It reads actual visits and values after every backup
and rebuilds root priority data after noise or reused-root prior updates.
Other first-play modes, proof propagation and small child sets retain the
ordinary selector. The default remains disabled.

Tests compare complete policies, values, root visits and reused noisy trees on
rectangular, 5×5 and 10×10 boards. Additional cases cover mixed current players,
cancelled backups and scores tied despite distinct tiny priors. The full suite
passes 207 tests.

[Alternating cold-cache probes](data/large-model-20261012/sparse-selection-probe.json)
use the same retained weights and 1,024 simulations. On one opening per board,
median times are 6.83 → 4.00 seconds on 10×10 and 14.12 → 7.87 on 12×12,
reductions of approximately 42% and 44%. 4×4 and 5×5 are roughly unchanged.
All policies, values and root visits match exactly. These are limited speed
probes, not a strength result or a promise of two-second full-budget searches.

The deeper-budget screen is operationally restarted before any complete opening
pair, with the same seed 19001, enabling this selector on both sides. Candidate
work remains 8,192 simulations against reference 1,024. Its confirmation and
control seeds and gates remain unchanged. The [original unfinished traces](data/large-model-20261012/deeper-search-screen-source-restart.json)
are retained separately; no outcomes are pooled or used to choose new openings.

The taught-model confirmation also restarts with the same seed 209001 before
any complete pair, keeping its original selector and all model/search settings.
[Original unfinished traces](data/large-model-20261012/policy-teaching-confirm-source-restart.json)
record the operational source-isolation restart. Both restarted evaluations
load a frozen copy of the package and benchmark script, so spawned workers
cannot import later edits. This corrects a provenance gap in evaluations that
hashed parent source but allowed workers to import live workspace files.
