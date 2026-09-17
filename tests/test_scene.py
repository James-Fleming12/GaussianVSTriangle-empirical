"""Synthetic ground-truth tests: ray casting, shading, initialization."""

import torch

from gvst.scene.synthetic import Quad, Sphere, make_scene


def test_sphere_intersection_depth_is_exact():
    sphere = Sphere((0.0, 0.0, 0.0), 1.0, lambda uv: torch.ones(uv.shape[0], 3))
    origin = torch.tensor([[0.0, 0.0, 3.0]])
    direction = torch.tensor([[0.0, 0.0, -1.0]])
    hit = sphere.intersect(origin, direction)
    assert hit.valid.item()
    assert abs(hit.t.item() - 2.0) < 1e-5
    assert torch.allclose(hit.normal, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-5)


def test_sphere_miss_is_invalid():
    sphere = Sphere((0.0, 0.0, 0.0), 0.5, lambda uv: torch.ones(uv.shape[0], 3))
    origin = torch.tensor([[2.0, 0.0, 3.0]])
    direction = torch.tensor([[0.0, 0.0, -1.0]])
    assert not sphere.intersect(origin, direction).valid.any()


def test_plane_hit_and_bounds():
    quad = Quad((-1.0, -1.0, -1.0), (2.0, 0, 0), (0, 2.0, 0), lambda uv: torch.ones(uv.shape[0], 3))
    origin = torch.tensor([[0.0, 0.0, 3.0], [5.0, 0.0, 3.0]])
    direction = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    hit = quad.intersect(origin, direction)
    assert hit.valid[0] and not hit.valid[1]
    assert abs(hit.t[0].item() - 4.0) < 1e-5
    assert torch.allclose(hit.normal[0], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5)


def test_scene_render_shapes_and_range():
    scene = make_scene("corner")
    cams = scene.cameras(n=3, width=16, height=12, fov_deg=45.0)
    out = scene.render(cams[0])
    assert out["image"].shape == (3, 12, 16)
    assert out["depth"].shape == (12, 16)
    assert out["normal"].shape == (3, 12, 16)
    assert out["image"].min() >= 0 and out["image"].max() <= 1
    assert out["mask"].any()
    # depth is positive where a surface was hit and zero elsewhere
    assert (out["depth"][out["mask"]] > 0).all()
    assert (out["depth"][~out["mask"]] == 0).all()


def test_render_depths_match_raycast():
    scene = make_scene("sphere")
    cam = scene.cameras(n=1, width=16, height=16, fov_deg=45.0)[0]
    out = scene.render(cam, supersample=1)
    uv = cam.pixel_grid()
    origins, dirs = cam.ray(uv)
    hit = scene.intersect(origins, dirs)
    # rendered depth is the camera-space depth (-z_cam); hit.t is the Euclidean
    # distance along the (unit) ray, so scale by |dir_z| in camera space
    d_cam_z = (dirs @ cam.R_cw)[:, 2].abs()
    assert torch.allclose(
        out["depth"].reshape(-1)[hit.valid],
        (hit.t * d_cam_z)[hit.valid],
        atol=1e-4,
    )


def test_init_points_lie_near_surfaces():
    scene = make_scene("sphere")
    cams = scene.cameras(n=3, width=16, height=16, fov_deg=45.0)
    pts, cols = scene.init_points(cams, 200, seed=1)
    assert pts.shape == (200, 3)
    assert cols.shape == (200, 3)
    # every point must be within a small tolerance of the unit sphere
    assert (pts.norm(dim=1) - 1.0).abs().max() < 0.05


def test_init_points_only_use_given_cameras():
    scene = make_scene("sphere")
    cams = scene.cameras(n=4, width=16, height=16, fov_deg=45.0)
    pts_a, _ = scene.init_points(cams[:2], 100, seed=5)
    pts_b, _ = scene.init_points(cams[:2], 100, seed=5)
    pts_c, _ = scene.init_points(cams[2:], 100, seed=5)
    assert torch.allclose(pts_a, pts_b)  # deterministic given seed
    assert not torch.allclose(pts_a, pts_c)  # different views -> different points


def test_extent_positive(tiny_scene, tiny_cameras):
    assert tiny_scene.extent(tiny_cameras) > 0
