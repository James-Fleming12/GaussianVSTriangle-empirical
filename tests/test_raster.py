"""Compositing, ordering, depth/normal regularizer sanity checks."""

import torch

from gvst.render.raster import _composite, distortion_loss, normal_consistency
from gvst.scene.synthetic import make_scene
from gvst.train.losses import ssim
from gvst.eval.metrics import psnr


def test_single_opaque_primitive_renders_its_color():
    alpha = torch.ones(1, 4) * 0.999999
    depth = torch.ones(1, 4) * 2.0
    colors = torch.tensor([[1.0, 0.0, 0.0]])
    bg = torch.zeros(3)
    image, acc, w = _composite(alpha, depth, colors, bg)
    assert torch.allclose(image, torch.tensor([[1.0, 0.0, 0.0]]).expand(4, 3), atol=1e-4)
    assert torch.allclose(acc, torch.full((4,), 0.999999), atol=1e-5)


def test_depth_ordering_front_wins():
    """The first (closest) opaque primitive determines the colour."""
    alpha = torch.ones(2, 4) * 0.999999
    depth = torch.tensor([[1.0] * 4, [2.0] * 4])  # already sorted front-to-back
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    image, _, _ = _composite(alpha, depth, colors, torch.zeros(3))
    assert torch.allclose(image[:, 0], torch.ones(4), atol=1e-3)
    assert torch.allclose(image[:, 1], torch.zeros(4), atol=1e-3)


def test_transmittance_decreases_with_depth():
    alpha = torch.full((3, 1), 0.5)
    depth = torch.tensor([[1.0], [2.0], [3.0]])
    colors = torch.zeros(3, 3)
    _, acc, w = _composite(alpha, depth, colors, torch.zeros(3))
    assert w[0, 0] > w[1, 0] > w[2, 0]
    assert acc[0] < 1.0


def test_distortion_loss_zero_for_coplanar_and_positive_otherwise():
    weights = torch.full((2, 4), 0.5)
    same = torch.ones(2, 4) * 2.0
    assert distortion_loss(weights, same).item() == 0.0
    diff = torch.tensor([[2.0] * 4, [2.5] * 4])
    assert distortion_loss(weights, diff).item() > 0


def test_normal_consistency_identical_normals():
    weights = torch.full((2, 8), 0.5)
    normals = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    n_map = torch.zeros(2, 4, 3)
    n_map[..., 1] = 1.0
    assert abs(normal_consistency(weights, normals, n_map).item()) < 1e-6
    n_map[..., 1] = -1.0
    assert abs(normal_consistency(weights, normals, n_map).item() - 2.0) < 1e-6


def test_metrics_identical_and_noisy():
    a = torch.rand(3, 8, 8)
    assert psnr(a, a) > 60
    assert abs(ssim(a, a).item() - 1.0) < 1e-5
    noisy = (a + 0.3 * torch.randn_like(a)).clamp(0, 1)
    assert psnr(noisy, a) < psnr(a, a)
    assert ssim(noisy, a) < 1.0


def test_render_end_to_end_smoke(tiny_scene, tiny_cameras, tiny_cfg):
    from gvst.primitives import build_model
    from gvst.render.raster import render

    cams = tiny_cameras
    gts = [tiny_scene.render(c)["image"] for c in cams]
    pts, cols = tiny_scene.init_points(cams[:3], 60, seed=0)
    model = build_model("3dgs", pts, cols, tiny_cfg, tiny_scene.extent(cams[:3]), seed=0)
    out = render(model, cams[3], background=torch.tensor(tiny_scene.background))
    assert out.image.shape == gts[3].shape
    assert torch.isfinite(out.image).all()
    loss = (out.image - gts[3]).abs().mean()
    loss.backward()
    assert all(p.grad is not None for p in model.p.values())
