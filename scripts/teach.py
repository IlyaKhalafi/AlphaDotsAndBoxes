"""Build exact teaching data or fine-tune the unchanged network with RLlib."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def main():
    from alphaboxes.teaching import build_component_dataset, build_dataset, fine_tune

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sources", nargs="+", type=Path)
    mode.add_argument("--replay", type=Path, help="Trusted local replay for component teaching")
    mode.add_argument("--dataset", type=Path)
    parser.add_argument("--teach-policy", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--positions", type=int, default=2048)
    parser.add_argument("--exact-fraction", type=float, default=0.25)
    parser.add_argument("--gpu-memory-gb", type=float, default=28)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.sources or args.replay:
        if args.output.suffix != ".npz":
            parser.error("Dataset output must have a .npz suffix.")
        if args.replay:
            result = build_component_dataset(
                args.checkpoint,
                args.replay,
                args.output,
                args.seed if args.seed is not None else 41071,
                args.positions,
            )
        else:
            result = build_dataset(
                args.checkpoint,
                args.sources,
                args.output,
                args.seed if args.seed is not None else 3310,
            )
        print(json.dumps(result), flush=True)
    else:
        print(
            fine_tune(
                args.checkpoint,
                args.dataset,
                args.output,
                teach_policy=args.teach_policy,
                steps=args.steps,
                batch_size=args.batch_size,
                device=args.device,
                seed=args.seed if args.seed is not None else 3312,
                exact_fraction=args.exact_fraction,
                gpu_memory_gb=args.gpu_memory_gb,
            )
        )


if __name__ == "__main__":
    main()
