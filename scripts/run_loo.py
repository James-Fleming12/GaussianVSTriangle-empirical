#!/usr/bin/env python3
"""Run the leave-one-out benchmark on the synthetic scenes.

Examples
--------
Smoke test (seconds, CPU)::

    python scripts/run_loo.py --quick --device cpu --output results/quick

Full benchmark (GPU)::

    python scripts/run_loo.py --output results/main \
        --scenes plane,sphere,corner,sheets --methods 3dgs,2dgs,triangle \
        --views 8 --folds 0,1,2 --iterations 3000 --image-size 64
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gvst.config import BenchmarkConfig, SceneSpec, TrainConfig, scaled_train_config
from gvst.eval.loo import run_benchmark

SCENE_DESCRIPTIONS = {
    "plane": "flat textured plane (oblique surfaces)",
    "sphere": "curved surface + specular highlight",
    "corner": "room corner (sharp edges / occlusion)",
    "sheets": "thin disconnected sheets",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenes", default="plane,sphere,corner,sheets")
    p.add_argument("--methods", default="3dgs,2dgs,triangle")
    p.add_argument("--views", type=int, default=8, help="total cameras per scene")
    p.add_argument("--folds", default="0,1,2", help="held-out camera indices, comma separated, or 'all'")
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--init-points", type=int, default=3000)
    p.add_argument("--sh-degree", type=int, default=2)
    p.add_argument(
        "--lambda-dist",
        type=float,
        default=None,
        help="override 2DGS depth-distortion weight (default: method default, i.e. 0)",
    )
    p.add_argument(
        "--lambda-normal",
        type=float,
        default=None,
        help="override 2DGS normal-consistency weight (default: method default, i.e. 0)",
    )
    p.add_argument("--dist-from", type=int, default=None, help="override 2DGS distortion warm-up iteration")
    p.add_argument("--normal-from", type=int, default=None, help="override 2DGS normal warm-up iteration")
    p.add_argument("--seeds", default="0", help="comma separated benchmark seeds")
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--output", default="results/main")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--quick", action="store_true", help="tiny configuration for a smoke run")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.quick:
        args.iterations = min(args.iterations, 60)
        args.views = min(args.views, 4)
        args.image_size = min(args.image_size, 24)
        args.init_points = min(args.init_points, 120)
        args.eval_every = min(args.eval_every, 20)
        args.folds = "0" if args.folds == "0,1,2" else args.folds
        args.device = "cpu" if not torch.cuda.is_available() else args.device

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    specs = []
    for name in scenes:
        if name not in SCENE_DESCRIPTIONS:
            raise SystemExit(f"unknown scene {name!r}; choose from {list(SCENE_DESCRIPTIONS)}")
        specs.append(
            SceneSpec(
                name=name,
                description=SCENE_DESCRIPTIONS[name],
                views=args.views,
                width=args.image_size,
                height=args.image_size,
                init_points=args.init_points,
                seed=0,
            )
        )

    folds = None if args.folds == "all" else [int(f) for f in args.folds.split(",")]
    cfg = TrainConfig(
        sh_degree=args.sh_degree,
        iterations=args.iterations,
        lambda_dist=0.0 if args.lambda_dist is None else args.lambda_dist,
        lambda_normal=0.0 if args.lambda_normal is None else args.lambda_normal,
    )
    cfg = scaled_train_config(cfg)
    if args.dist_from is not None:
        cfg.dist_from_iter = args.dist_from
    if args.normal_from is not None:
        cfg.normal_from_iter = args.normal_from
    bench = BenchmarkConfig(
        scenes=specs,
        methods=methods,
        train_config=cfg,
        folds=folds,
        seeds=tuple(int(s) for s in args.seeds.split(",")),
        output_dir=args.output,
        device=args.device,
        eval_every=args.eval_every,
        save_images=not args.no_images,
    )
    print(f"benchmark: {len(specs)} scene(s) x {len(methods)} method(s) x "
          f"{len(folds) if folds else args.views} fold(s); {args.iterations} iters, "
          f"{args.image_size}px on {args.device}")
    records = run_benchmark(bench, log=True)
    print(f"\nwrote {len(records)} records to {os.path.join(args.output, 'records.jsonl')}")


if __name__ == "__main__":
    main()
