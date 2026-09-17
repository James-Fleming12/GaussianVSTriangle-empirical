"""Iteration-2 additions: stress scenes, geometry metrics, diagnostics, ablations."""

import torch

from gvst.config import TrainConfig, scaled_train_config
from gvst.eval.geometry import evaluate_geometry, sample_triangle_surface
from gvst.primitives import build_model
from gvst.scene.synthetic import SCENES, make_scene


def test_new_scenes_render_and_sample():
    for name in ("intersect", "hf", "solid"):
        assert name in SCENES
        scene = make_scene(name)
        cam = scene.cameras(n=1, width=16, height=16, fov_deg=45.0)[0]
        out = scene.render(cam)
        assert out["image"].shape == (3, 16, 16)
        assert out["mask"].any()
        clouds = scene.sample_surface(2000, seed=0)
        assert clouds["points"].shape == (2000, 3)
        assert clouds["normals"].shape == (2000, 3)
        assert torch.isfinite(clouds["points"]).all()
        # sampled normals are unit length
        assert torch.allclose(clouds["normals"].norm(dim=1), torch.ones(2000), atol=1e-4)


def test_surface_samples_lie_on_analytic_shapes():
    scene = make_scene("solid").to("cpu")
    clouds = scene.sample_surface(3000, seed=0)
    pts = clouds["points"]
    on_sphere = (pts.norm(dim=1) - 1.0).abs() < 1e-4
    on_floor = (pts[:, 1] + 1.0).abs() < 1e-4
    assert bool((on_sphere | on_floor).all())


def _toy_model(method, n=8, seed=0):
    torch.manual_seed(seed)
    pts = torch.rand(n, 3) * 0.5
    cols = torch.rand(n, 3)
    cfg = scaled_train_config(TrainConfig(iterations=20, sh_degree=1, seed=seed)).with_method(method)
    return build_model(method, pts, cols, cfg, scene_extent=10.0, seed=seed), cfg


def test_triangle_diagnostics_fields():
    model, cfg = _toy_model("triangle")
    model.setup_optimizer(cfg)
    model._reset_diagnostics()
    # fabricate one backward step's worth of stats
    model._diag_grad_zero, model._diag_grad_total = 2, 8
    model._diag_cov_sum, model._diag_cov_count = 4.0, 8
    d = model.diagnostics()
    assert d["zero_grad_frac"] == 0.25
    assert 0.0 <= d["coverage_frac"] <= 1.0
    for key in ("area_mean", "degenerate_frac", "sliver_frac", "min_angle_mean_deg", "aspect_mean"):
        assert key in d
    # diagnostics reset after being read
    assert model._diag_grad_total == 0


def test_gaussian_diagnostics_fields():
    for method in ("3dgs", "2dgs"):
        model, cfg = _toy_model(method)
        model.setup_optimizer(cfg)
        model._reset_diagnostics()
        model._diag_grad_zero, model._diag_grad_total = 1, 8
        d = model.diagnostics()
        assert "scale_mean" in d and "tiny_frac" in d and "opacity_mean" in d


def test_triangle_densify_mode_none_is_a_noop():
    model, cfg = _toy_model("triangle", n=6)
    cfg.tri_densify_mode = "none"
    model.setup_optimizer(cfg)
    model._stats = {
        "importance": torch.full((6,), 1.0),
        "image_size": torch.full((6,), 100.0),
        "coverage": torch.full((6,), 3.0),
    }
    n0 = model.num_primitives
    model.densify(0)
    assert model.num_primitives == n0


def test_triangle_none_mode_skips_final_prune():
    model, cfg = _toy_model("triangle", n=4)
    cfg.tri_densify_mode = "none"
    model.setup_optimizer(cfg)
    model._stats = {
        "importance": torch.zeros(4),
        "image_size": torch.zeros(4),
        "coverage": torch.zeros(4),
    }
    model.prune_final()
    assert model.num_primitives == 4


def test_triangle_densify_deterministic_grows_within_cap():
    model, cfg = _toy_model("triangle", n=8)
    cfg.tri_densify_mode = "deterministic"
    cfg.tri_max_shapes = 20
    cfg.tri_growth = 2.0
    cfg.tri_densify_from_iter = 0
    cfg.tri_densify_until_iter = 100
    cfg.tri_densification_interval = 1
    cfg.tri_importance_threshold = -1.0
    model.setup_optimizer(cfg)
    model._stats = {
        "importance": torch.arange(8, dtype=torch.float32),
        "image_size": torch.full((8,), 2.0),  # small -> clone
        "coverage": torch.full((8,), 3.0),
    }
    model.densify(0)
    assert model.num_primitives > 8
    assert model.num_primitives <= 20


def test_geometry_metric_identical_clouds():
    pts = torch.rand(500, 3)
    nrm = torch.nn.functional.normalize(torch.randn(500, 3), dim=1)
    cloud = {"points": pts, "normals": nrm}
    out = evaluate_geometry(cloud, cloud, threshold=0.01, max_points=1000)
    assert out["chamfer"] < 1e-3
    assert abs(out["fscore"] - 1.0) < 1e-6
    assert abs(out["normal_consistency"] - 1.0) < 1e-5


def test_geometry_metric_penalises_offset_cloud():
    pts = torch.rand(500, 3)
    nrm = torch.nn.functional.normalize(torch.randn(500, 3), dim=1)
    shifted = {"points": pts + 0.5, "normals": nrm}
    cloud = {"points": pts, "normals": nrm}
    out = evaluate_geometry(shifted, cloud, threshold=0.01, max_points=1000)
    assert out["chamfer"] > 0.1
    assert out["fscore"] == 0.0


def test_sample_triangle_surface_shapes():
    model, cfg = _toy_model("triangle", n=5)
    model.setup_optimizer(cfg)
    soup = sample_triangle_surface(model, 400, seed=0)
    assert soup["points"].shape == (400, 3)
    assert soup["normals"].shape == (400, 3)
    assert torch.allclose(soup["normals"].norm(dim=1), torch.ones(400), atol=1e-4)
