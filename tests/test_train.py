"""Trainer integration tests: optimization, checkpoints, protocol integrity."""

import json

import torch

from gvst.config import SceneSpec, TrainConfig, scaled_train_config
from gvst.eval.loo import run_fold
from gvst.eval.metrics import psnr
from gvst.primitives import build_model
from gvst.train.trainer import Trainer


def _mini_setup(method="3dgs", iters=40, seed=0, views=4, size=16):
    from gvst.scene.synthetic import make_scene

    scene = make_scene("corner")
    cams = scene.cameras(n=views, width=size, height=size, fov_deg=45.0, radius=3.0)
    gts = [scene.render(c)["image"] for c in cams]
    pts, cols = scene.init_points(cams[: views - 1], 150, seed=seed)
    cfg = scaled_train_config(TrainConfig(iterations=iters, sh_degree=1, seed=seed)).with_method(method)
    cfg.densify_until_iter = 0  # optimization-only tests; densification has its own tests
    model = build_model(method, pts, cols, cfg, scene.extent(cams[: views - 1]), seed=seed)
    return scene, cams, gts, model, cfg


def test_training_reduces_loss_and_updates_params():
    scene, cams, gts, model, cfg = _mini_setup("3dgs", iters=150)
    trainer = Trainer(model, cfg, background=torch.tensor([0.1, 0.1, 0.12]), device="cpu")

    def view_psnr():
        from gvst.render.raster import render

        with torch.no_grad():
            out = render(model, cams[0], background=torch.tensor([0.1, 0.1, 0.12]))
        return psnr(out.image, gts[0])

    before = {k: p.detach().clone() for k, p in model.p.items()}
    psnr_before = view_psnr()
    fit = trainer.fit(cams[:3], eval_camera=cams[3], eval_target=gts[3], train_targets=gts[:3], eval_every=50)
    assert len(fit["history"]) >= 2
    assert view_psnr() > psnr_before
    assert any(not torch.allclose(before[k], p.detach()) for k, p in model.p.items())
    assert torch.isfinite(fit["heldout_image"]).all()


def test_heldout_camera_is_never_used_for_initialization(monkeypatch):
    """The protocol must not leak the held-out view into the model."""
    from gvst.scene import synthetic

    recorded = {}
    original = synthetic.Scene.init_points

    def spy(self, cameras, n_points, seed=0):
        recorded["centers"] = [c.t.detach().clone() for c in cameras]
        return original(self, cameras, n_points, seed=seed)

    monkeypatch.setattr(synthetic.Scene, "init_points", spy)
    spec = SceneSpec(name="corner", description="", views=4, width=12, height=12, init_points=60)
    cfg = scaled_train_config(TrainConfig(iterations=10, sh_degree=1)).with_method("3dgs")
    run_fold("corner", spec, fold=0, held_out=2, method="3dgs", cfg=cfg, device="cpu", seed=0, eval_every=5)

    from gvst.scene.synthetic import make_scene

    scene = make_scene("corner")
    cams = scene.cameras(n=4, width=12, height=12, fov_deg=45.0, radius=3.0)
    held_out_center = cams[2].t
    assert len(recorded["centers"]) == 3
    assert all(not torch.allclose(c, held_out_center) for c in recorded["centers"])


def test_run_fold_struct():
    spec = SceneSpec(name="plane", description="", views=4, width=12, height=12, init_points=80)
    cfg = scaled_train_config(TrainConfig(iterations=15, sh_degree=1)).with_method("2dgs")
    res = run_fold("plane", spec, fold=1, held_out=1, method="2dgs", cfg=cfg, device="cpu", seed=0, eval_every=5)
    assert res.n_train == 3
    assert res.held_out == 1
    assert 1 not in res.train_views
    assert len(res.train_views) == 3
    assert res.psnr > 5.0
    assert res.history[-1]["iteration"] == 14
    assert isinstance(json.loads(json.dumps(res.to_dict())), dict)


def test_run_fold_is_deterministic():
    spec = SceneSpec(name="plane", description="", views=4, width=12, height=12, init_points=80)
    cfg = scaled_train_config(TrainConfig(iterations=10, sh_degree=1, seed=0)).with_method("3dgs")
    a = run_fold("plane", spec, 0, 0, "3dgs", cfg, device="cpu", seed=0, eval_every=5)
    b = run_fold("plane", spec, 0, 0, "3dgs", cfg, device="cpu", seed=0, eval_every=5)
    assert a.psnr == b.psnr
    assert a.n_primitives == b.n_primitives


def test_train_view_subset_is_respected():
    spec = SceneSpec(name="plane", description="", views=6, width=12, height=12, init_points=80)
    cfg = scaled_train_config(TrainConfig(iterations=8, sh_degree=1)).with_method("triangle")
    res = run_fold(
        "plane", spec, fold=3, held_out=0, method="triangle", cfg=cfg, device="cpu", seed=0,
        eval_every=4, train_indices=[1, 2, 3],
    )
    assert res.n_train == 3
    assert res.train_views == [1, 2, 3]
