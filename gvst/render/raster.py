"""Unified differentiable splat compositing (pure PyTorch).

All three representations share one compositing engine.  A model only has to
expose, for a *chunk of pixels*:

* ``alpha [N,p]``  : per-pixel opacity (already multiplied by the primitive
  opacity),
* ``depth [N,p]``  : per-pixel ray parameter (positive, in world units),
* and (optionally) ``normal [N,3]`` : world-space, camera-facing normal.

Everything else -- front-to-back ordering, transmittance, the background
composite, expected depth, the 2DGS depth-distortion regularizer and the
normal-consistency regularizer -- is implemented once here, which is what makes
the comparison controlled.

Reference equations:

* alpha compositing: 3DGS Eq. (3), 2DGS Eq. (12), Triangle Splatting Sec. 3.1
* depth distortion: 2DGS Eq. (13)
* normal consistency: 2DGS Eq. (14-15); Triangle Splatting uses the same term
  on the triangle face normal against the depth-gradient normal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..scene.cameras import Camera
from .sh import eval_sh

_EPS = 1e-6


@dataclass
class RenderOutput:
    image: Tensor  # [3,H,W]
    depth: Tensor  # [H,W] expected depth
    acc: Tensor  # [H,W] accumulated opacity
    weights: Optional[Tensor] = None  # [N,P] blending weights (sorted order)
    depth_maps: Optional[Tensor] = None  # [N,P] per-primitive ray depths (sorted order)
    pixel_order: Optional[Tensor] = None  # [N] primitive order used
    normals: Optional[Tensor] = None  # [N,3] per-primitive camera-facing world normals

    def train_psnr(self, target: Tensor) -> Tensor:
        mse = ((self.image - target) ** 2).mean()
        return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))


def colors_from_sh(features: Tensor, dirs: Tensor, degree: int) -> Tensor:
    """SH colour, shifted/clamped exactly like the reference CUDA renderers."""
    return (eval_sh(features, dirs, degree) + 0.5).clamp(0.0, 1.0)


def _composite(alpha: Tensor, depth: Tensor, colors: Tensor, bg: Tensor) -> tuple:
    """Front-to-back compositing for already-depth-sorted primitives.

    ``alpha [N,p]``, ``depth [N,p]``, ``colors [N,3]``; returns
    ``(image [p,3], acc [p], weights [N,p])``.
    """
    a = alpha.clamp(0.0, 1.0 - 1e-6)
    log_t = torch.log1p(-a)
    cum = torch.cumsum(log_t, dim=0)
    cum_exclusive = torch.cat([torch.zeros_like(cum[:1]), cum[:-1]], dim=0)
    w = a * torch.exp(cum_exclusive)
    image = torch.einsum("np,nc->pc", w, colors)
    acc = w.sum(dim=0)
    image = image + (1.0 - acc).unsqueeze(-1) * bg
    return image, acc, w


def _expected_depth(weights: Tensor, depth: Tensor) -> Tensor:
    acc = weights.sum(dim=0)
    out = (weights * depth).sum(dim=0) / acc.clamp_min(_EPS)
    return torch.where(acc > _EPS, out, torch.zeros_like(out))


def render(
    model,
    camera: Camera,
    background: Tensor | None = None,
    need_weights: bool = False,
    need_depth_maps: bool = False,
    chunk_pixels: int = 4096,
) -> RenderOutput:
    """Render ``model`` from ``camera`` with the shared compositor."""
    n_prims = model.num_primitives
    h, w = camera.height, camera.width
    p = h * w
    device = camera.device

    if background is None:
        background = torch.zeros(3, device=device)
    background = background.to(device)

    model.start_view(camera)
    colors = model.colors_for_view(camera)  # [N,3]
    order = model.sort_order(camera)  # [N] front-to-back
    sorted_colors = colors[order]

    pixels = camera.pixel_grid()  # [P,2]

    image = torch.zeros(p, 3, device=device)
    acc = torch.zeros(p, device=device)
    depth_map = torch.zeros(p, device=device)
    weights_all = torch.zeros(n_prims, p, device=device) if need_weights else None
    depth_maps_all = torch.zeros(n_prims, p, device=device) if need_depth_maps else None
    normals = None

    for start in range(0, p, chunk_pixels):
        chunk = pixels[start : start + chunk_pixels]
        screen = model.screen_chunk(chunk, camera)
        if normals is None and screen.get("normal") is not None:
            normals = screen["normal"][order]
        a = screen["alpha"][order]
        z = screen["depth"][order]
        img_c, acc_c, w_c = _composite(a, z, sorted_colors, background)
        image[start : start + chunk.shape[0]] = img_c
        acc[start : start + chunk.shape[0]] = acc_c
        depth_map[start : start + chunk.shape[0]] = _expected_depth(w_c, z)
        if need_weights:
            weights_all[:, start : start + chunk.shape[0]] = w_c
        if need_depth_maps:
            depth_maps_all[:, start : start + chunk.shape[0]] = z

    out = RenderOutput(
        image=image.reshape(h, w, 3).permute(2, 0, 1),
        depth=depth_map.reshape(h, w),
        acc=acc.reshape(h, w),
        weights=weights_all,
        depth_maps=depth_maps_all,
        pixel_order=order,
        normals=normals,
    )
    return out


def distortion_loss(weights: Tensor, depth: Tensor) -> Tensor:
    """2DGS Eq. (13): ``sum_ij w_i w_j |z_i - z_j|``, mean over pixels.

    ``weights`` / ``depth`` must be in front-to-back (ascending depth) order
    along dim 0; cumulative moment identities avoid the O(N^2) double sum.
    """
    s = torch.cumsum(weights, dim=0) - weights
    m = torch.cumsum(weights * depth, dim=0) - weights * depth
    term = 2.0 * weights * (depth * s - m)
    return term.sum(dim=0).mean()


def depth_to_world_points(camera: Camera, depth: Tensor) -> Tensor:
    """Back-project a depth map (camera depth units) to world points [H,W,3]."""
    h, w = camera.height, camera.width
    uv = camera.pixel_grid().reshape(h, w, 2)
    _, dirs = camera.ray(uv)
    return camera.center.view(1, 1, 3) + depth.unsqueeze(-1) * dirs


def _gradients(field: Tensor) -> tuple:
    """Central differences, one-sided at the border, along H (dim0) and W (dim1)."""
    dx = torch.zeros_like(field)
    dy = torch.zeros_like(field)
    dx[:, 1:-1] = (field[:, 2:] - field[:, :-2]) * 0.5
    dx[:, 0] = field[:, 1] - field[:, 0]
    dx[:, -1] = field[:, -1] - field[:, -2]
    dy[1:-1, :] = (field[2:, :] - field[:-2, :]) * 0.5
    dy[0, :] = field[1, :] - field[0, :]
    dy[-1, :] = field[-1, :] - field[-2, :]
    return dx, dy


def depth_normal_map(camera: Camera, depth: Tensor, mask: Tensor | None = None) -> Tensor:
    """Normal map from the gradient of a depth map (2DGS Eq. 15).

    Normals are computed in world space and oriented towards the camera.
    """
    points = depth_to_world_points(camera, depth)
    dx, dy = _gradients(points)
    n = torch.cross(dx, dy, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    to_cam = camera.center.view(1, 1, 3) - points
    flip = (n * to_cam).sum(-1, keepdim=True) < 0
    n = torch.where(flip, -n, n)
    if mask is not None:
        n = torch.where(mask.unsqueeze(-1), n, torch.zeros_like(n))
    return n


def normal_consistency(weights: Tensor, normals: Tensor, normal_map: Tensor, mask: Tensor | None = None) -> Tensor:
    """``mean(1 - <render_normal, depth_normal>)``, both unit length.

    ``weights [N,p]`` and ``normals [N,3]`` are aligned; ``normal_map [H,W,3]``.
    This is the form used by both the 2DGS and Triangle Splatting codebases.
    """
    h, w = normal_map.shape[:2]
    p = h * w
    render_n = torch.einsum("np,nc->pc", weights, normals)
    render_n = render_n / render_n.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    target = normal_map.reshape(p, 3)
    err = 1.0 - (render_n * target).sum(-1)
    if mask is not None:
        err = err * mask.reshape(p)
        return err.sum() / mask.sum().clamp_min(1.0)
    return err.mean()


def median_depth(weights: Tensor, depth: Tensor) -> Tensor:
    """Depth at which the cumulative blending weight crosses 0.5."""
    cdf = torch.cumsum(weights, dim=0)
    idx = (cdf >= 0.5).float().argmax(dim=0)
    return depth.gather(0, idx.unsqueeze(0)).squeeze(0)
