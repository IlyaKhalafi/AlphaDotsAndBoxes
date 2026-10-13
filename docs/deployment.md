# Deploy the game

The standard installation needs NumPy, FastAPI, and Uvicorn. The graph network,
search, and game rules run on CPU. PyTorch, Ray, RLlib, and Gymnasium are absent
from the deployment dependency set.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
adb serve --checkpoint models/larger.npz --host 0.0.0.0 --port 9003
```

Open port 9003 on the host or place the service behind your hosting platform's
reverse proxy. `/api/health` identifies the inference backend and checkpoint hash.
The bundled Docker image runs the same command on port 8000 as a non-root user:

```bash
docker build -t alpha-dots-and-boxes .
docker run --rm -p 9003:8000 alpha-dots-and-boxes
```

## Use your own trained model

On the training machine, install the optional tools and export weights:

```bash
python -m pip install -e '.[train]'
adb export --checkpoint runs/larger/latest.pt --output models/custom.npz
```

Copy only the `.npz` file to the deployment machine. It contains float32 arrays
and JSON metadata, uses `allow_pickle=False`, and records its source checkpoint
hash. No optimizer or training code is needed to load it.

```bash
adb serve --checkpoint models/custom.npz --host 0.0.0.0 --port 9003
```

For Docker, mount that file read-only and override the image's command:

```bash
docker run --rm -p 9003:8000 \
  -v "$PWD/models/custom.npz:/app/models/custom.npz:ro" alpha-dots-and-boxes \
  adb serve --checkpoint models/custom.npz --host 0.0.0.0 --port 8000
```

Legacy `.pt` weights remain supported when the training extra is installed.
NumPy export preserves the architecture and weights; floating-point differences
can change search tie decisions. Tests compare masked logits and values across
square, rectangular, and padded graphs. Evaluate exported weights when reporting
deployed playing strength.

The bootstrap export was checked on 206 states across six board shapes, with
seven extra padding nodes. Maximum absolute error was 1.43×10⁻⁶ for logits and
3.88×10⁻⁷ for values. The [parity receipt](data/numpy-bootstrap-parity.json)
records both checkpoint hashes and software versions. A fresh installation and
a non-root Docker container both completed graph search without training
frameworks installed; the clean installation also finished a 3×3 self-play game.
The [larger-board export receipt](data/numpy-larger-parity.json) checks the same
206 states against the recommended model's source weights.

## Hosting behavior

Use one server process: games live in its memory, expire after an hour of
inactivity, and reset on a restart. The service allows 128 sessions. Multiple
replicas need sticky routing or a shared session store; neither is included.

For NumPy checkpoints, CPU search runs in a warm child process. Agent and hint
requests have a **1.9-second response deadline**, including waiting for another
game's search. Internal thinking is capped at 1.75 seconds. If the worker stalls,
the caller returns a legal tactical move and replaces its own worker on the next
request. Waiting requests receive only the search time left in their response
budget. Replays mark fallback moves and omit an unavailable value estimate.
Concurrent agent/hint requests on the same game receive a 409 response instead
of waiting for a stale move. OS scheduling and network transport can still affect
observed HTTP latency; [measured timings](two-second-search.md) include 10×10
positions and clean installations. Legacy PyTorch serving uses the older search
path; export `.npz` weights for supervised NumPy deployment.

The UI offers Quick (0.15 s), Standard (0.5 s), and Deep (up to the internal cap).
Opt into a separate longer-search worker with `adb serve --checkpoint
models/larger.npz --extended-search`. On graph-engine boards with at least
100 boxes, the UI labels Deep **up to 60 s** and sends `extended: true`.
This requests 8,192 simulations with a 60-second response deadline, including
queueing. Ordinary requests keep the short deadline and use a separate worker.
The long worker retains up to 32,768 exact network predictions to reuse searched
continuations; the ordinary worker retains 4,096. Cold openings can still be slow.
Extended requests on smaller boards or the alpha–beta engine receive 422.
Replays report actual simulation counts and `full_budget_completed`; a
time-limited partial search is not the complete fixed-work benchmark.
See [the strength and timing report](deeper-large-board-search.md).

Agent/hint requests accept `time_limit` from 0.05 to 2 seconds and `engine`
(`mcts` or `alpha_beta`). Timed MCTS has an 8,192-traversal ceiling; requests with
only `simulations` retain their 8–512-traversal ceiling within the deadline.
Alpha–beta is experimental. The released MCTS uses first-play reduction 0.33,
with exact root solving at 12 remaining edges. The experimental degree-two
solver and proof propagation are disabled in the released agent.

`/api/health` reports the checkpoint hash, search setting and deadline. Replays
record actual completed traversals or alpha–beta depth, nodes, cutoffs and table
hits. The CLI sets one OpenBLAS/OMP thread unless explicitly overridden.

For move-by-move debugging, `GET /api/games/{id}/replay` exports the active
game's moves, players, agent search settings, checkpoint metadata, and final
position. Replaying those actions reconstructs extra turns correctly. Undo
removes the discarded continuation from the export. Save the response before
the session expires or the server restarts; games are not stored on disk.

The runtime is free of external model APIs and credentials. Training and
development use `.[train]` and `.[dev]` respectively; recording the UI uses the
optional `.[media]` tools.
