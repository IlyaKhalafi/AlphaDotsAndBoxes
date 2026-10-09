# Deploy the game

The standard installation needs NumPy, FastAPI, and Uvicorn. The graph network,
search, and game rules run on CPU. PyTorch, Ray, RLlib, and Gymnasium are absent
from the deployment dependency set.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
adb serve --checkpoint models/agent.npz --host 0.0.0.0 --port 9003
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

## Hosting behavior

Use one server process: games live in its memory, expire after an hour of
inactivity, and reset on a restart. The service allows 128 simultaneous sessions
and serializes CPU search. Multiple replicas need sticky routing or a shared
session store; neither is included. Large boards and Deep thinking cost more CPU
time. The CLI defaults to one OpenBLAS/OMP thread unless you set those environment
variables explicitly.

The runtime is free of external model APIs and credentials. Training and
development use `.[train]` and `.[dev]` respectively; recording the UI uses the
optional `.[media]` tools.
