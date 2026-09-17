"""Geometry metrics for the primitive comparison.

PSNR/SSIM can be won by appearance fitting even when the reconstructed surface
is wrong (floaters, cracks, interpenetration).  This module extracts a surface
point cloud per method and compares it to an *analytic* ground-truth surface
sample with:

* symmetric Chamfer distance (L1),
* F-score at a scene-relative threshold (2% of the camera-ring extent),
* normal consistency (nearest-neighbour cosine agreement).

For Triangle Splatting the cloud is the explicit triangle soup (all primitives,
so floaters and stray triangles are penalised).  For 3DGS/2DGS -- which have no
explicit surface -- the cloud is the rendered expected depth back-projected from
each camera.  A rendered-depth cloud is also computed for triangles so the two
extraction modes can be compared directly.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..render.raster import depth_normal_map, depth_to_world_points, render


def sample_triangle_surface(model, n: int, seed: int = 0) -> dict:
    """Area-weighted barycentric samples on all triangles (world points + normals)."""
    v = model.p["vertices"].detach()
    generator = torch.Generator(device=v.device).manual_seed(seed)
    cross = torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=1)
    area = 0.5 * cross.norm(dim=1)
    normals = cross / cross.norm(dim=1, keepdim=True).clamp_min(1e-12)
    total = float(area.sum())
    if total <= 0 or v.shape[0] == 0:
        return {"points": v.new_zeros(0, 3), "normals": v.new_zeros(0, 3)}
    sel = torch.multinomial(area / area.sum(), n, replacement=True, generator=generator)
    u = torch.rand(n, 1, device=v.device, generator=generator)
    w = torch.rand(n, 1, device=v.device, generator=generator)
    flip = (u + w) > 1.0
    u = torch.where(flip, 1.0 - u, u)
    w = torch.where(flip, 1.0 - w, w)
    points = v[sel, 0] * (1.0 - u - w) + v[sel, 1] * u + v[sel, 2] * w
    return {"points": points, "normals": normals[sel]}


@torch.no_grad()
def rendered_surface_points(model, cameras: list, background: Tensor) -> dict:
    """Back-project rendered expected depth from each camera into a point cloud."""
    pts, nrm = [], []
    with torch.no_grad():
        for cam in cameras:
            out = render(model, cam, background=background, need_weights=False, need_depth_maps=False)
            mask = out.acc > 0.5
            if int(mask.sum()) == 0:
                continue
            p = depth_to_world_points(cam, out.depth)
            nmap = depth_normal_map(cam, out.depth)
            pts.append(p[mask])
            nrm.append(nmap[mask])
    if not pts:
        return {"points": background.new_zeros(0, 3), "normals": background.new_zeros(0, 3)}
    return {"points": torch.cat(pts, 0), "normals": torch.cat(nrm, 0)}


def _nearest(points: Tensor, reference: Tensor, chunk: int = 2048) -> tuple:
    """Nearest-neighbour distance + index from every ``points`` to ``reference``."""
    if points.shape[0] == 0 or reference.shape[0] == 0:
        return points.new_zeros(points.shape[0]), torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
    dists = torch.empty(points.shape[0], device=points.device)
    idxs = torch.empty(points.shape[0], dtype=torch.long, device=points.device)
    for start in range(0, points.shape[0], chunk):
        d = torch.cdist(points[start : start + chunk], reference)
        v, i = d.min(dim=1)
        dists[start : start + chunk] = v
        idxs[start : start + chunk] = i
    return dists, idxs


def evaluate_geometry(pred: dict, gt: dict, threshold: float, max_points: int = 6000) -> dict:
    """Chamfer / F-score / normal consistency between two point clouds."""
    p, n_p = pred["points"], pred["normals"].float()
    g, n_g = gt["points"], gt["normals"].float()
    if p.shape[0] > max_points:
        sel = torch.randperm(p.shape[0], device=p.device)[:max_points]
        p, n_p = p[sel], n_p[sel]
    if g.shape[0] > max_points:
        sel = torch.randperm(g.shape[0], device=g.device)[:max_points]
        g, n_g = g[sel], n_g[sel]
    if p.shape[0] == 0 or g.shape[0] == 0:
        return {"chamfer": float("nan"), "fscore": 0.0, "normal_consistency": float("nan")}

    d_p2g, i_p2g = _nearest(p, g)
    d_g2p, i_g2p = _nearest(g, p)
    chamfer = 0.5 * (d_p2g.mean() + d_g2p.mean())
    precision = (d_p2g < threshold).float().mean()
    recall = (d_g2p < threshold).float().mean()
    fscore = (2 * precision * recall / (precision + recall).clamp_min(1e-9)).item()
    nc_p = (n_p * n_g[i_p2g]).abs().sum(-1).mean()
    nc_g = (n_g * n_p[i_g2p]).abs().sum(-1).mean()
    return {
        "chamfer": float(chamfer),
        "fscore": float(fscore),
        "normal_consistency": float(0.5 * (nc_p + nc_g)),
    }


def geometry_for_model(
    model,
    method: str,
    cameras: list,
    background: Tensor,
    gt_cloud: dict,
    extent: float,
    n_soup: int = 6000,
    seed: int = 0,
) -> dict:
    """Full geometry report for one trained model against an analytic GT cloud."""
    threshold = 0.02 * extent
    out: dict = {}
    rendered = rendered_surface_points(model, cameras, background)
    out.update(
        {f"geom_{k}": v for k, v in evaluate_geometry(rendered, gt_cloud, threshold).items()}
    )
    if method == "triangle":
        soup = sample_triangle_surface(model, n_soup, seed=seed)
        soup = {k: v.to(background.device) for k, v in soup.items()}
        out.update(
            {f"geom_soup_{k}": v for k, v in evaluate_geometry(soup, gt_cloud, threshold).items()}
        )
    return out
