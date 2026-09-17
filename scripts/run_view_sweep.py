#!/usr/bin/env python3
"""How does held-out quality depend on the number of training views?

Fixes one held-out camera per scene and trains on the first ``K`` of the
remaining views for ``K`` in a sweep.  This isolates the few-shot
generalization question ("which primitive degrades slowest when views are
scarce?") from the standard leave-one-out protocol in ``run_loo.py``.

Usage::

    python scripts/run_view_sweep.py --scene corner --held-out 0 \
        --counts 3,5,7 --iterations 3000 --output results/views
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gvst.config import SceneSpec, TrainConfig, scaled_train_config
from gvst.eval.loo import run_fold


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="corner")
    p.add_argument("--held-out", type=int, default=0)
    p.add_argument("--counts", default="3,5,7")
    p.add_argument("--methods", default="3dgs,2dgs,triangle")
    p.add_argument("--views", type=int, default=8)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--init-points", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="results/views")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output, exist_ok=True)
    counts = [int(c) for c in args.counts.split(",")]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    spec = SceneSpec(
        name=args.scene,
        description="",
        views=args.views,
        width=args.image_size,
        height=args.image_size,
        init_points=args.init_points,
    )
    records = []
    with open(os.path.join(args.output, "records.jsonl"), "a") as fout:
        for count in counts:
            train_indices = [i for i in range(args.views) if i != args.held_out][:count]
            for method in methods:
                cfg = scaled_train_config(
                    TrainConfig(iterations=args.iterations, sh_degree=2, seed=args.seed)
                ).with_method(method)
                record = run_fold(
                    spec.name,
                    spec,
                    fold=count,
                    held_out=args.held_out,
                    method=method,
                    cfg=cfg,
                    device=args.device,
                    seed=args.seed,
                    eval_every=max(1, args.iterations // 10),
                    log=True,
                    save_dir=os.path.join(args.output, "raw", spec.name, method, f"k{count}"),
                    train_indices=train_indices,
                ).to_dict()
                record["n_train"] = count
                records.append(record)
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                print(
                    f"{spec.name}/{method}: {count} train views -> "
                    f"held-out PSNR {record['psnr']:.2f} SSIM {record['ssim']:.3f}",
                    flush=True,
                )
    print(f"wrote {len(records)} records to {os.path.join(args.output, 'records.jsonl')}")


if __name__ == "__main__":
    main()
