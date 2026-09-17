#!/usr/bin/env python3
"""Iteration-2 robustness / training-diagnostics experiments.

These runs extend the main leave-one-out benchmark (``scripts/run_loo.py``)
along the axes needed to judge whether Triangle Splatting's held-out win is
robust and *where its training actually breaks*:

* ``scenes``      LOO on three new stress scenes (intersecting sheets,
                  high-frequency texture, textureless geometry), with geometry
                  metrics;
* ``geometry``    geometry metrics for the four original scenes (fold 0);
* ``densify``     Triangle Splatting densification ablation
                  (MCMC vs deterministic vs none);
* ``longtrain``   3k vs 10k iterations, to expose held-out overfitting;
* ``resolution``  64 vs 128 px, to expose hard-window aliasing;
* ``sh``          SH-degree-0 appearance ablation (colour capacity).

Every run records per-checkpoint training diagnostics (zero-gradient fraction,
coverage, triangle degeneracy) and, where enabled, analytic Chamfer/F-score/
normal-consistency.  Runs are resumable: completed configurations in an
experiment's ``records.jsonl`` are skipped.

Usage::

    python scripts/run_iter2.py --experiments scenes,geometry,densify \
        --device cuda --output results/iter2
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

SCENE_DESCRIPTIONS = {
    "plane": "flat textured plane",
    "sphere": "curved surface + specular",
    "corner": "room corner",
    "sheets": "thin disconnected sheets",
    "intersect": "mutually intersecting sheets",
    "hf": "high-frequency textured plane",
    "solid": "textureless geometry",
}

# Each experiment is a dict of overrides applied to a common base configuration.
EXPERIMENTS = {
    "scenes": dict(
        scenes=("intersect", "hf", "solid"),
        methods=("3dgs", "2dgs", "triangle"),
        folds=(0, 1, 2),
        iterations=3000,
        image_size=64,
        compute_geometry=True,
    ),
    "geometry": dict(
        scenes=("plane", "sphere", "corner", "sheets"),
        methods=("3dgs", "2dgs", "triangle"),
        folds=(0,),
        iterations=3000,
        image_size=64,
        compute_geometry=True,
    ),
    "densify": dict(
        scenes=("corner", "sheets", "intersect"),
        methods=("triangle",),
        folds=(0, 1, 2),
        iterations=3000,
        image_size=64,
        compute_geometry=False,
        densify_modes=("mcmc", "deterministic", "none"),
    ),
    "longtrain": dict(
        scenes=("corner", "sphere"),
        methods=("3dgs", "2dgs", "triangle"),
        folds=(0,),
        iterations=10000,
        image_size=64,
        compute_geometry=False,
    ),
    # paired baselines for the ablations (same seed / init as their ablation)
    "longtrain3k": dict(
        scenes=("corner", "sphere"),
        methods=("3dgs", "2dgs", "triangle"),
        folds=(0,),
        iterations=3000,
        image_size=64,
        compute_geometry=False,
    ),
    "resolution": dict(
        scenes=("hf", "corner"),
        methods=("3dgs", "2dgs", "triangle"),
        folds=(0,),
        iterations=3000,
        image_size=128,
        compute_geometry=False,
    ),
    "resolution64": dict(
        scenes=("hf", "corner"),
        methods=("3dgs", "2dgs", "triangle"),
        folds=(0,),
        iterations=3000,
        image_size=64,
        compute_geometry=False,
    ),
    "sh": dict(
        scenes=("sphere", "hf"),
        methods=("3dgs", "2dgs", "triangle"),
        folds=(0,),
        iterations=3000,
        image_size=64,
        compute_geometry=False,
        sh_degree=0,
    ),
    "sh2": dict(
        scenes=("sphere", "hf"),
        methods=("3dgs", "2dgs", "triangle"),
        folds=(0,),
        iterations=3000,
        image_size=64,
        compute_geometry=False,
        sh_degree=2,
    ),
}


def _key(record: dict) -> tuple:
    return (
        record.get("scene"),
        record.get("method"),
        record.get("fold"),
        record.get("iterations"),
        record.get("image_size"),
        record.get("sh_degree"),
        record.get("tri_densify_mode", "mcmc"),
    )


def _load_done(path: str) -> set:
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(_key(json.loads(line)))
    return done


def run_experiment(name: str, base_dir: str, device: str, log: bool = True) -> int:
    spec = EXPERIMENTS[name]
    out_dir = os.path.join(base_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "records.jsonl")
    done = _load_done(path)

    modes = spec.get("densify_modes", ("mcmc",))
    new = 0
    with open(path, "a") as fout:
        for s_i, scene_name in enumerate(spec["scenes"]):
            for fold in spec["folds"]:
                for method in spec["methods"]:
                    for mode in modes:
                        if method != "triangle" and mode != "mcmc":
                            continue
                        seed = 1000 * s_i + 10 * fold
                        probe = {
                            "scene": scene_name,
                            "method": method,
                            "fold": fold,
                            "iterations": spec["iterations"],
                            "image_size": spec["image_size"],
                            "sh_degree": spec.get("sh_degree", 2),
                            "tri_densify_mode": mode if method == "triangle" else "mcmc",
                        }
                        if _key(probe) in done:
                            continue
                        scene_spec = SceneSpec(
                            name=scene_name,
                            description=SCENE_DESCRIPTIONS.get(scene_name, ""),
                            views=8,
                            width=spec["image_size"],
                            height=spec["image_size"],
                            init_points=3000,
                            seed=0,
                        )
                        cfg = TrainConfig(
                            sh_degree=spec.get("sh_degree", 2),
                            iterations=spec["iterations"],
                            seed=seed,
                            collect_diagnostics=True,
                        )
                        cfg = scaled_train_config(cfg).with_method(method)
                        cfg.tri_densify_mode = mode
                        result = run_fold(
                            scene_name,
                            scene_spec,
                            fold,
                            fold,  # held-out camera == fold index
                            method,
                            cfg,
                            device=device,
                            seed=seed,
                            eval_every=200,
                            log=False,
                            save_dir=None,
                            compute_geometry=spec.get("compute_geometry", False),
                        )
                        record = result.to_dict()
                        record.update(
                            experiment=name,
                            iterations=spec["iterations"],
                            image_size=spec["image_size"],
                            sh_degree=spec.get("sh_degree", 2),
                            tri_densify_mode=mode if method == "triangle" else "mcmc",
                        )
                        fout.write(json.dumps(record) + "\n")
                        fout.flush()
                        new += 1
                        if log:
                            print(
                                f"[{name}] {scene_name}/{method}/fold{fold}/{mode} "
                                f"PSNR {record['psnr']:.2f} SSIM {record['ssim']:.3f} "
                                f"({record['seconds']:.0f}s, {record['n_primitives']} prims)"
                            )
    return new


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", default="scenes,geometry,densify,longtrain,resolution,sh")
    p.add_argument("--output", default="results/iter2")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    names = [n.strip() for n in args.experiments.split(",") if n.strip()]
    for name in names:
        if name not in EXPERIMENTS:
            raise SystemExit(f"unknown experiment {name!r}; choose from {list(EXPERIMENTS)}")
    total = 0
    for name in names:
        total += run_experiment(name, args.output, args.device, log=True)
    print(f"\ndone: {total} new run(s) written under {args.output}")


if __name__ == "__main__":
    main()
