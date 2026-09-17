"""Per-primitive projection / alpha semantics for the three methods."""

import torch

from gvst.primitives import build_model
from gvst.render.raster import render


def _single(method, cfg, point=(0.0, 0.0, 0.0), extent=1.0, seed=0):
    pts = torch.tensor([point])
    cols = torch.tensor([[0.5, 0.5, 0.5]])
    model = build_model(method, pts, cols, cfg, extent, seed=seed)
    return model


def test_all_methods_project_finite(axis_camera, tiny_cfg):
    for method in ("3dgs", "2dgs", "triangle"):
        model = _single(method, tiny_cfg)
        out = render(model, axis_camera, background=torch.zeros(3))
        assert torch.isfinite(out.image).all(), method
        assert out.image.shape == (3, 16, 16)
        assert out.acc.max() <= 1.0 + 1e-5


def test_3dgs_alpha_peaks_at_center(axis_camera, tiny_cfg):
    model = _single("3dgs", tiny_cfg)
    with torch.no_grad():
        model.p["opacity"].fill_(2.0)  # sigmoid -> ~0.88
        model.p["scaling"].fill_(torch.log(torch.tensor(0.3)))
    out = render(model, axis_camera, background=torch.zeros(3))
    center_alpha = out.acc[int(axis_camera.cy), int(axis_camera.cx)]
    corner_alpha = out.acc[0, 0]
    assert center_alpha > corner_alpha


def test_2dgs_ray_hits_center_at_correct_depth(axis_camera, tiny_cfg):
    model = _single("2dgs", tiny_cfg)
    with torch.no_grad():
        model.p["opacity"].fill_(2.0)
        model.p["scaling"].fill_(torch.log(torch.tensor(0.3)))
    model.start_view(axis_camera)
    uv = torch.tensor([[axis_camera.cx, axis_camera.cy]])  # exact principal point
    screen = model.screen_chunk(uv, axis_camera)
    assert abs(screen["depth"][0, 0].item() - 3.0) < 1e-4
    assert abs(screen["alpha"][0, 0].item() - torch.sigmoid(torch.tensor(2.0)).item()) < 1e-4


def test_triangle_window_bounded_by_geometry(axis_camera, tiny_cfg):
    """I(s)=1 at the incenter, 0 on the boundary and outside (Sec. 3.1)."""
    model = _single("triangle", tiny_cfg)
    with torch.no_grad():
        model.p["vertices"] = torch.nn.Parameter(
            torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        )
        model.p["opacity"].fill_(0.0)  # sigmoid -> 0.5
    model.start_view(axis_camera)
    opacity = float(torch.sigmoid(torch.tensor(0.0)))

    world_incenter = torch.tensor([[0.293, 0.293, 0.0]])
    uv_incenter, _ = axis_camera.project(world_incenter)
    s = model.screen_chunk(uv_incenter, axis_camera)
    assert abs(s["alpha"][0, 0].item() - opacity) < 2e-2

    edge_mid = axis_camera.project(torch.tensor([[0.5, 0.5, 0.0]]))[0]
    s_edge = model.screen_chunk(edge_mid, axis_camera)
    assert s_edge["alpha"][0, 0].item() < 1e-4

    outside = axis_camera.project(torch.tensor([[2.0, 2.0, 0.0]]))[0]
    s_out = model.screen_chunk(outside, axis_camera)
    assert s_out["alpha"][0, 0].item() == 0.0


def test_triangle_sigma_is_positive_and_learnable(tiny_cfg):
    model = _single("triangle", tiny_cfg)
    assert (model.sigma() > 0).all()
    area = model.triangle_area()
    assert area.item() > 0


def test_triangle_area_zero_for_degenerate(tiny_cfg):
    model = _single("triangle", tiny_cfg)
    with torch.no_grad():
        model.p["vertices"] = torch.nn.Parameter(torch.zeros(1, 3, 3))
    assert model.triangle_area().item() == 0.0


def test_gradients_flow_to_all_parameters(axis_camera, tiny_cfg):
    for method in ("3dgs", "2dgs", "triangle"):
        model = _single(method, tiny_cfg)
        gt = torch.ones(3, 16, 16)
        out = render(model, axis_camera, background=torch.zeros(3))
        ((out.image - gt) ** 2).mean().backward()
        for name, p in model.p.items():
            assert p.grad is not None, f"{method}:{name}"
            assert torch.isfinite(p.grad).all(), f"{method}:{name}"
