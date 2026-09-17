"""Camera convention tests.

Convention under test (see ``gvst/scene/cameras.py``):

* ``R_cw`` camera->world with columns ``[right, up, backward]``,
  ``t`` the camera centre in world space,
* camera looks down ``-z``; pixel ``(u, v)`` has its origin top-left,
  ``u`` right and ``v`` down,
* ``u = cx + fx x/d``, ``v = cy - fy y/d`` with ``d = -z_cam``.
"""

import torch

from gvst.scene.cameras import Camera, field_of_view_camera, look_at, orbit_cameras


def test_center_projects_to_principal_point(axis_camera):
    uv, depth = axis_camera.project(torch.zeros(3))
    assert torch.allclose(uv, torch.tensor([axis_camera.cx, axis_camera.cy]), atol=1e-5)
    assert torch.allclose(depth, torch.tensor(3.0), atol=1e-5)


def test_projection_signs(axis_camera):
    right = axis_camera.project(torch.tensor([1.0, 0.0, 0.0]))[0]
    up = axis_camera.project(torch.tensor([0.0, 1.0, 0.0]))[0]
    assert right[0] > axis_camera.cx  # +x world -> larger u
    assert right[1] == axis_camera.cy
    assert up[1] < axis_camera.cy  # +y world -> smaller v (v grows down)
    assert up[0] == axis_camera.cx


def test_world_camera_roundtrip(axis_camera):
    p = torch.randn(7, 3)
    assert torch.allclose(axis_camera.camera_to_world(axis_camera.world_to_camera(p)), p, atol=1e-5)


def test_rays_intersect_camera_center(axis_camera):
    uv = torch.tensor([[0.0, 0.0], [15.0, 15.0], [8.0, 3.0]])
    origins, dirs = axis_camera.ray(uv)
    assert torch.allclose(origins, axis_camera.center.expand_as(origins), atol=1e-6)
    # move along each ray to the exact camera depth -z = 2 and recover the pixel
    d_cam = dirs @ axis_camera.R_cw
    pts = origins + (2.0 / (-d_cam[:, 2]))[:, None] * dirs
    uv_back, depth = axis_camera.project(pts)
    assert torch.allclose(uv_back, uv, atol=1e-4)
    assert torch.allclose(depth, torch.full_like(depth, 2.0), atol=1e-4)


def test_orbit_cameras_look_at_target():
    target = torch.tensor([0.5, -0.2, 0.1])
    cams = orbit_cameras(n=5, radius=3.0, target=target, fov_deg=40.0, width=12, height=12)
    for cam in cams:
        uv, depth = cam.project(target)
        assert abs(uv[0].item() - cam.cx) < 1e-4
        assert abs(uv[1].item() - cam.cy) < 1e-4
        assert depth.item() > 0


def test_look_at_orthonormal():
    R, t = look_at(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0.0, 0.0, 0.0]))
    eye = R @ R.T
    assert torch.allclose(eye, torch.eye(3), atol=1e-5)
    assert torch.allclose(t, torch.tensor([1.0, 2.0, 3.0]))
