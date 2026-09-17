"""Leave-one-out (LOO) evaluation protocol.

Protocol (per synthetic scene):

1.  Fix a set of ``V`` orbit cameras and render exact ground truth for each.
2.  For every fold ``f``: hold out camera ``f``; train on the other ``V-1``
    views only.  The initialization point cloud is back-projected from the
    *training* views only, so the held-out view never reaches the model.
3.  Evaluate PSNR / SSIM on the held-out view, and keep the training curves
    (held-out and training PSNR over wall-clock time) so that convergence
    speed, not just final quality, can be compared.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch

from ..config import BenchmarkConfig, SceneSpec
from ..primitives import METHODS, build_model
from ..scene.synthetic import make_scene
from ..train.trainer import Trainer
from .metrics import depth_rmse, evaluate


@dataclass
class FoldResult:
    scene: str
    method: str
    fold: int
    held_out: int
    n_train: int
    psnr: float
    ssim: float
    depth_rmse: float
    train_psnr: float
    train_ssim: float
    n_primitives: int
    seconds: float
    peak_memory_mb: float
    seconds_to_20db: Optional[float]
    history: list = field(default_factory=list)
    seed: int = 0
    image_size: int = 0
    train_views: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _scene_ground_truth(scene, cameras: list) -> list:
    return [scene.render(cam) for cam in cameras]


def _seconds_to(history: list, key: str, threshold: float) -> Optional[float]:
    for entry in history:
        if entry.get(key, -1e9) >= threshold:
            return float(entry["seconds"])
    return None


def run_fold(
    scene_name: str,
    spec: SceneSpec,
    fold: int,
    held_out: int,
    method: str,
    cfg,
    device: str = "cuda",
    seed: int = 0,
    eval_every: int = 200,
    log: bool = False,
    save_dir: Optional[str] = None,
    train_indices: Optional[list] = None,
) -> FoldResult:
    """Train on the given training views and evaluate on ``held_out``."""
    scene = make_scene(scene_name, seed=spec.seed).to(device)
    cameras = spec_cameras(spec, scene)
    gts = _scene_ground_truth(scene, cameras)

    if train_indices is None:
        train_idx = [i for i in range(spec.views) if i != held_out]
    else:
        train_idx = [i for i in train_indices if i != held_out]
    train_cameras = [cameras[i] for i in train_idx]
    train_targets = [gts[i]["image"].detach() for i in train_idx]
    eval_camera = cameras[held_out]
    eval_target = gts[held_out]["image"].detach()
    # the compositing background must match the scene's sky colour
    background = torch.tensor(scene.background, device=device)

    points, colors = scene.init_points(train_cameras, spec.init_points, seed=seed)
    extent = scene.extent(train_cameras)
    model = build_model(method, points, colors, cfg, extent, seed=seed).to(device)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(model, cfg, background=background, device=device, log=log)
    fit = trainer.fit(
        train_cameras,
        eval_camera=eval_camera,
        eval_target=eval_target,
        eval_every=eval_every,
        train_targets=train_targets,
    )

    heldout_image = fit.pop("heldout_image")
    heldout_depth = fit.pop("heldout_depth")
    metrics = evaluate(heldout_image, eval_target.cpu())
    depth_err = depth_rmse(heldout_depth, gts[held_out]["depth"].detach().cpu(), gts[held_out]["mask"].detach().cpu())

    # final training-view quality: mean over all training cameras
    from ..render.raster import render as render_fn

    model.eval()
    train_psnrs, train_ssims = [], []
    with torch.no_grad():
        for cam, tgt in zip(train_cameras, train_targets):
            out = render_fn(model, cam, background=background, need_weights=False, need_depth_maps=False)
            m = evaluate(out.image, tgt)
            train_psnrs.append(m["psnr"])
            train_ssims.append(m["ssim"])
    train_psnr = float(sum(train_psnrs) / max(1, len(train_psnrs)))
    train_ssim = float(sum(train_ssims) / max(1, len(train_ssims)))
    peak_mb = 0.0
    if device.startswith("cuda"):
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

    result = FoldResult(
        scene=scene_name,
        method=method,
        fold=fold,
        held_out=held_out,
        n_train=len(train_cameras),
        psnr=metrics["psnr"],
        ssim=metrics["ssim"],
        depth_rmse=depth_err,
        train_psnr=train_psnr,
        train_ssim=train_ssim,
        n_primitives=fit["n_primitives"],
        seconds=fit["seconds"],
        peak_memory_mb=peak_mb,
        seconds_to_20db=_seconds_to(fit["history"], "heldout_psnr", 20.0),
        history=fit["history"],
        seed=seed,
        image_size=spec.width,
        train_views=train_idx,
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _save_images(save_dir, heldout_image, eval_target.cpu(), heldout_depth)
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(result.to_dict(), f, indent=2)
    return result


def _save_images(save_dir: str, pred: torch.Tensor, gt: torch.Tensor, depth: torch.Tensor | None = None) -> None:
    try:
        from PIL import Image

        def to_img(t):
            a = (t.detach().cpu().permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
            return Image.fromarray(a)

        to_img(pred).save(os.path.join(save_dir, "heldout_pred.png"))
        to_img(gt).save(os.path.join(save_dir, "heldout_gt.png"))
        if depth is not None:
            d = depth.detach().cpu().numpy()
            d = (d - d.min()) / max(d.max() - d.min(), 1e-6)
            Image.fromarray((d * 255).astype("uint8")).save(os.path.join(save_dir, "heldout_depth.png"))
    except Exception:  # pragma: no cover - images are a convenience
        pass


def spec_cameras(spec: SceneSpec, scene):
    return scene.cameras(
        n=spec.views,
        width=spec.width,
        height=spec.height,
        fov_deg=spec.fov_deg,
        radius=spec.radius,
    )


def run_benchmark(bench: BenchmarkConfig, log: bool = False) -> list:
    device = bench.device
    cfg = bench.train_config
    os.makedirs(bench.output_dir, exist_ok=True)
    records_path = os.path.join(bench.output_dir, "records.jsonl")
    records = []
    with open(records_path, "a") as fout:
        for s_i, spec in enumerate(bench.scenes):
            scene = make_scene(spec.name, seed=spec.seed).to(device)
            cameras = spec_cameras(spec, scene)
            gts = _scene_ground_truth(scene, cameras)
            del scene, cameras, gts
            folds = bench.folds if bench.folds is not None else list(range(spec.views))
            for fold, held_out in enumerate(folds):
                if held_out >= spec.views:
                    raise ValueError(f"fold {held_out} out of range for {spec.name} ({spec.views} views)")
                for m_i, method in enumerate(bench.methods):
                    if method not in METHODS:
                        raise KeyError(f"unknown method {method!r}")
                    for seed_i, base_seed in enumerate(bench.seeds):
                        seed = int(base_seed + 1000 * s_i + 10 * fold)
                        method_cfg = cfg.with_method(method)
                        method_cfg.seed = seed
                        save_dir = os.path.join(
                            bench.output_dir, "raw", spec.name, method, f"fold{fold}_seed{seed_i}"
                        )
                        result = run_fold(
                            spec.name,
                            spec,
                            fold,
                            held_out,
                            method,
                            method_cfg,
                            device=device,
                            seed=seed,
                            eval_every=bench.eval_every,
                            log=log,
                            save_dir=save_dir if bench.save_images else None,
                        )
                        record = result.to_dict()
                        records.append(record)
                        fout.write(json.dumps(record) + "\n")
                        fout.flush()
                        if log:
                            print(
                                f"  -> {spec.name}/{method}/fold{fold} "
                                f"PSNR {record['psnr']:.2f} SSIM {record['ssim']:.3f} "
                                f"({record['seconds']:.1f}s, {record['n_primitives']} prims)"
                            )
    return records
