"""Pinhole cameras with an explicit, tested convention.

Convention (matches the tests in the sibling ``3D-and-2DGS-Implementations``
repository):

* ``R_cw`` is camera-to-world, columns ``[right, up, backward]``.
* ``t`` is the camera centre in world space.
* The camera looks down ``-z_cam`` (OpenGL-style), ``+x_cam`` is image right,
  ``+y_cam`` is image up.
* Pixel ``(u, v)`` has origin at the top-left, ``u`` grows right and ``v``
  grows down; ``u = cx + fx * x / d`` and ``v = cy - fy * y / d`` with
  ``d = -z_cam`` the positive depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def look_at(eye: Tensor, target: Tensor, world_up: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Return ``(R_cw, t)`` for a camera at ``eye`` looking at ``target``."""
    if world_up is None:
        world_up = torch.tensor([0.0, 1.0, 0.0], dtype=eye.dtype, device=eye.device)
    forward = target - eye
    forward = forward / forward.norm().clamp_min(1e-8)
    right = torch.cross(forward, world_up, dim=0)
    if right.norm() < 1e-6:  # degenerate: use another reference axis
        alt = torch.tensor([1.0, 0.0, 0.0], dtype=eye.dtype, device=eye.device)
        right = torch.cross(forward, alt, dim=0)
    right = right / right.norm().clamp_min(1e-8)
    up = torch.cross(right, forward, dim=0)
    R_cw = torch.stack([right, up, -forward], dim=1)
    return R_cw, eye.clone()


@dataclass
class Camera:
    R_cw: Tensor  # [3,3]
    t: Tensor  # [3]
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def device(self) -> torch.device:
        return self.t.device

    def to(self, device) -> "Camera":
        return Camera(self.R_cw.to(device), self.t.to(device), self.fx, self.fy, self.cx, self.cy, self.width, self.height)

    # ---- transforms -------------------------------------------------
    def world_to_camera(self, p_world: Tensor) -> Tensor:
        """``p_world [..., 3] -> p_cam [..., 3]``."""
        return (p_world - self.t) @ self.R_cw

    def camera_to_world(self, p_cam: Tensor) -> Tensor:
        return p_cam @ self.R_cw.T + self.t

    @property
    def center(self) -> Tensor:
        return self.t

    # ---- projection -------------------------------------------------
    def project(self, p_world: Tensor) -> tuple[Tensor, Tensor]:
        """Project points, returning ``(uv [..., 2], depth [...])``.

        Pixels are in ``(u, v)`` (top-left origin, v down).  Points with
        non-positive depth produce ``inf``/``nan``-free values by clamping.
        """
        pc = self.world_to_camera(p_world)
        depth = -pc[..., 2]
        safe = depth.clamp_min(1e-6)
        u = self.cx + self.fx * pc[..., 0] / safe
        v = self.cy - self.fy * pc[..., 1] / safe
        return torch.stack([u, v], dim=-1), depth

    def ray(self, uv: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(origins [3], directions [..., 3])`` for pixels ``uv [..., 2]``."""
        x = (uv[..., 0] - self.cx) / self.fx
        y = -(uv[..., 1] - self.cy) / self.fy
        z = -torch.ones_like(x)
        d_cam = torch.stack([x, y, z], dim=-1)
        d_cam = d_cam / d_cam.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        d_world = d_cam @ self.R_cw.T
        return self.t, d_world

    def pixel_grid(self) -> Tensor:
        """Pixel centres ``[H*W, 2]`` in row-major order."""
        v, u = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float32, device=self.device),
            torch.arange(self.width, dtype=torch.float32, device=self.device),
            indexing="ij",
        )
        return torch.stack([u.reshape(-1), v.reshape(-1)], dim=-1)

    def scaled(self, factor: int) -> "Camera":
        """Supersampled copy: same frustum, ``factor`` x resolution."""
        return Camera(
            self.R_cw,
            self.t,
            self.fx * factor,
            self.fy * factor,
            (self.cx + 0.5) * factor - 0.5,
            (self.cy + 0.5) * factor - 0.5,
            self.width * factor,
            self.height * factor,
        )


def field_of_view_camera(
    eye: Tensor,
    target: Tensor,
    fov_deg: float,
    width: int,
    height: int,
    world_up: Tensor | None = None,
) -> Camera:
    """Perspective camera with horizontal FOV derived from ``width``."""
    R_cw, t = look_at(eye, target, world_up)
    fx = (width / 2.0) / float(torch.tan(torch.tensor(0.5 * fov_deg * torch.pi / 180.0)))
    fy = fx
    return Camera(R_cw, t, fx, fy, width / 2.0 - 0.5, height / 2.0 - 0.5, width, height)


def orbit_cameras(
    n: int,
    radius: float,
    target: Tensor,
    fov_deg: float,
    width: int,
    height: int,
    elevations_deg: tuple = (20.0, 25.0, 30.0),
    azimuth_offset_deg: float = -90.0,
    device=None,
) -> list[Camera]:
    """Place ``n`` cameras on a ring (azimuth sweep) with cycling elevation.

    The first camera sits on the ``-y`` side (azimuth ``-90`` plus offset) so
    that the scene is viewed from a natural upright orientation.
    """
    device = target.device if device is None else device
    target = target.to(device)
    cams = []
    for i in range(n):
        az = torch.deg2rad(torch.tensor(azimuth_offset_deg + 360.0 * i / n, device=device))
        el = torch.deg2rad(torch.tensor(float(elevations_deg[i % len(elevations_deg)]), device=device))
        x = radius * torch.cos(el) * torch.cos(az)
        y = radius * torch.sin(el)
        z = radius * torch.cos(el) * torch.sin(az)
        eye = target + torch.stack([x, y, z])
        cams.append(field_of_view_camera(eye, target, fov_deg, width, height))
    return cams
