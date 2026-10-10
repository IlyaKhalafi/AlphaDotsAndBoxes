# Model notes

All checkpoints use the same 272,648-parameter residual GIN: six blocks with 96
channels, shared edge policy and global value heads. They support different board
sizes without replacing a layer.

| File         | Training                                                                              | Role                              |
| ------------ | ------------------------------------------------------------------------------------- | --------------------------------- |
| `larger.pt`  | Larger-board iteration 200 + 320 new games, 13,934 new positions                 | Recommended experimental opponent |
| `agent.pt`   | 1,280 pure search-guided self-play games, 19,426 positions                            | Original bootstrap baseline       |
| `refined.pt` | 2,560 cumulative games, 38,810 positions; second phase enables exact 12-edge endgames | Comparison / further research     |

`larger.npz` contains the recommended model's weights for NumPy deployment;
`agent.npz` exports the original bootstrap baseline.
Serving it requires no PyTorch or RLlib. The `.pt` files are training-side and
legacy artifacts; export one with `adb export` to deploy it with NumPy.

The recommended model is deeper-self-play iteration 10, training seed 44, warm-started from larger-board iteration 200. This phase contributes 320 games and 13,934 positions; its selected snapshot uses 256 self-play simulations and exact 12-edge root endgames. It passed the pre-recorded NumPy deployment regression gate against the earlier preview on 4×4 and 5×5. [Fresh results and complete provenance](../docs/deep-run.md) describe the comparisons and their limits; expert-human strength is unproven.

Before larger-board training, bootstrap was retained because it scored slightly
higher against the stronger endgame opponent averaged across 3×3 and 4×4 in the
selection sample. Refinement improved small-board play and simple tactical
results but did not consistently improve the harder benchmark. The confidence
intervals overlap; this selection is a conservative heuristic, not proof that
one model is universally better.

The original two runs use training seed 42 and mix 1×2, 2×2, 2×3, and 3×3 boards. The 4×4
and 2×5 boards are outside the training size set. The first phase uses 32 PUCT
simulations; the second uses 64 and an exact late-game solver. No human games
are training data. The longer `strong.json` campaign has not been run.

```bash
adb serve --checkpoint models/larger.npz
adb serve --checkpoint models/agent.npz
adb export --checkpoint models/refined.pt --output models/refined.npz
adb serve --checkpoint models/refined.npz
```

The UI enables an exact 12-edge endgame aid for every graph checkpoint. CLI evaluation
disables that aid unless `--exact-threshold` is explicitly set. Pure policy,
learned search, and solver-aided results are separated in
[the experiment report](../docs/results.md).

Training checkpoints contain tensors, a format version, model settings, and plain
training metadata. They are loaded using `torch.load(..., weights_only=True)`.
Deployment checkpoints contain float32 NumPy arrays and JSON metadata, loaded
with `allow_pickle=False`; the source training checkpoint hash is retained.
They contain no optimizer, replay, external service configuration, or credentials.
Hashes are recorded in the evaluation receipts in `docs/data/`.

These are experimental models. Expert-human strength and reliable transfer to
arbitrarily large boards have not been established. They inherit the repository's
MIT license.
