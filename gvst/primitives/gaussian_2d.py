"""2D Gaussian Splatting (Huang et al., 2024).

Paper mapping:

* flat oriented disk ``H = [s_u t_u, s_v t_v, 0, p]``      Eq. (4-5)
* ray-splat intersection gives local ``(u, v)``            Eq. (7-10)
* object-space low-pass filter                            Eq. (11)
* alpha blending / rasterization                          Eq. (12)
* depth distortion                                        Eq. (13)
* normal consistency                                      Eq. (14-15)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .base import SplatModel, inverse_sigmoid, knn_mean_distance, quat_to_rot


class Gaussian2D(SplatModel):
    method = "2dgs"

    def __init__(self, points: Tensor, colors: Tensor, sh_degree: int, scene_extent: float, init_opacity: float = 0.1):
        super().__init__(sh_degree)
        n = points.shape[0]
        dist = knn_mean_distance(points)
        self.scene_extent = scene_extent
        self.p["means"] = nn.Parameter(points.clone())
        self.p["features"] = nn.Parameter(torch.zeros(n, (sh_degree + 1) ** 2, 3))
        with torch.no_grad():
            self.p["features"][:, 0] = (colors - 0.5) / 0.28209479177387814
        self.p["opacity"] = nn.Parameter(inverse_sigmoid(torch.full((n, 1), float(init_opacity))))
        self.p["scaling"] = nn.Parameter(torch.log(dist.clamp_min(1e-6))[:, None].repeat(1, 2))
        self.p["rotation"] = nn.Parameter(torch.zeros(n, 4))
        with torch.no_grad():
            self.p["rotation"][:, 0] = 1.0
        self._lr_names = {
            "features": "feature_lr",
            "opacity": "opacity_lr",
            "scaling": "scaling_lr",
            "rotation": "rotation_lr",
            "means": "position_lr_init",
        }
        self._max_radii2d: Tensor | None = None

    # ---- activations --------------------------------------------------
    @property
    def num_primitives(self) -> int:
        return self.p["means"].shape[0]

    @property
    def primitive_centers(self) -> Tensor:
        return self.p["means"]

    @property
    def opacity(self) -> Tensor:
        return torch.sigmoid(self.p["opacity"]).squeeze(-1)

    def scale(self) -> Tensor:
        return torch.exp(self.p["scaling"])

    def tangents(self) -> tuple:
        R = quat_to_rot(self.p["rotation"])
        return R[:, :, 0], R[:, :, 1], R[:, :, 2]

    def covariance_world(self) -> Tensor:
        tu, tv, _ = self.tangents()
        su, sv = self.scale()[:, 0], self.scale()[:, 1]
        a = (su[:, None] * tu)[:, :, None]
        b = (sv[:, None] * tv)[:, :, None]
        return a @ a.transpose(1, 2) + b @ b.transpose(1, 2)

    # ---- rendering ----------------------------------------------------
    def start_view(self, camera) -> None:
        tu, tv, n = self.tangents()
        su, sv = self.scale()[:, 0], self.scale()[:, 1]
        centers = self.p["means"]
        to_cam = camera.center[None, :] - centers
        flip = (n * to_cam).sum(-1, keepdim=True) < 0
        n = torch.where(flip, -n, n)  # camera-facing normal

        pc = camera.world_to_camera(centers)
        depth = -pc[:, 2]
        safe = depth.clamp_min(0.05)
        fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
        mu2d = torch.stack([cx + fx * pc[:, 0] / safe, cy - fy * pc[:, 1] / safe], dim=-1)
        if centers.requires_grad and torch.is_grad_enabled():
            mu2d.retain_grad()
            grad_ref = mu2d
        else:
            grad_ref = None

        # screen-space extent (3 sigma) for max_screen_size pruning
        J = torch.zeros(centers.shape[0], 2, 3, device=centers.device, dtype=centers.dtype)
        J[:, 0, 0] = fx / safe
        J[:, 0, 2] = fx * pc[:, 0] / safe**2
        J[:, 1, 1] = -fy / safe
        J[:, 1, 2] = -fy * pc[:, 1] / safe**2
        W = camera.R_cw.T
        cov_cam = W @ self.covariance_world() @ W.T
        cov2d = J @ cov_cam @ J.transpose(1, 2)
        cov2d[:, 0, 0] += 0.3
        cov2d[:, 1, 1] += 0.3
        tr = cov2d[:, 0, 0] + cov2d[:, 1, 1]
        det = (cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2).clamp_min(0.0)
        disc = (tr * tr / 4 - det).clamp_min(0.0).sqrt()
        lambda_max = tr / 2 + disc
        radii2d = 3.0 * lambda_max.clamp_min(0.0).sqrt()

        self._view_ctx = {
            "tu": tu,
            "tv": tv,
            "n": n,
            "su": su,
            "sv": sv,
            "centers": centers,
            "mu2d": mu2d,
            "depth": depth,
            "valid": depth > 0.1,
            "radii2d": radii2d,
            "opacity": torch.sigmoid(self.p["opacity"]).squeeze(-1),
            "grad_ref": grad_ref,
        }

    def screen_chunk(self, pixels: Tensor, camera) -> dict:
        ctx = self._view_ctx
        n = ctx["n"]
        centers = ctx["centers"]
        tu, tv = ctx["tu"], ctx["tv"]
        su, sv = ctx["su"], ctx["sv"]
        opacity = ctx["opacity"]
        p = pixels.shape[0]

        _, dirs = camera.ray(pixels)  # [p,3]
        denom = torch.einsum("pc,nc->np", dirs, n)  # [N,p] ray . splat-normal
        offset = torch.einsum("nc,nc->n", centers - camera.center, n)[:, None]
        denom_safe = denom.abs().clamp_min(1e-8) * torch.where(denom >= 0, 1.0, -1.0)
        t = offset / denom_safe  # [N,p]
        hit = camera.center[None, None, :] + t[..., None] * dirs[None]
        q = hit - centers[:, None, :]
        u = torch.einsum("npc,nc->np", q, tu) / su[:, None]
        v = torch.einsum("npc,nc->np", q, tv) / sv[:, None]
        g = torch.exp(-0.5 * (u * u + v * v))

        dist2 = ((pixels[None] - ctx["mu2d"][:, None, :]) ** 2).sum(-1)
        sigma_lp2 = 0.5  # 2DGS: sigma = sqrt(2)/2
        g_lp = torch.exp(-0.5 * dist2 / sigma_lp2)
        g = torch.maximum(g, g_lp)

        valid = (t > 0.05) & ctx["valid"][:, None]
        alpha = opacity[:, None] * g
        alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
        alpha = torch.where(alpha < 1e-5, torch.zeros_like(alpha), alpha)
        depth = torch.where(valid, t.clamp_min(0.0), torch.zeros_like(t))
        return {"alpha": alpha, "depth": depth, "normal": n}

    # ---- adaptive density control (2DGS Sec. 6.1 / 3DGS Sec. 5.2) -----
    def accumulate_densification_stats(self, render_out=None) -> None:
        super().accumulate_densification_stats(render_out)
        r = self._view_ctx.get("radii2d")
        if r is None:
            return
        r = r.detach()
        if self._max_radii2d is None or self._max_radii2d.shape[0] != r.shape[0]:
            self._max_radii2d = r.clone()
        else:
            self._max_radii2d = torch.maximum(self._max_radii2d, r)

    def densify(self, iteration: int) -> None:
        cfg = self._cfg
        if self.optimizer is None:
            return
        in_window = cfg.densify_from_iter <= iteration < cfg.densify_until_iter
        do_densify = in_window and cfg.densification_interval > 0 and iteration % cfg.densification_interval == 0
        if do_densify and self.mean_grad is not None:
            with torch.no_grad():
                grads = self.mean_grad
                scale_max = self.scale().max(dim=1).values
                threshold = cfg.percent_dense * self.scene_extent
                clone_mask = (grads >= cfg.densify_grad_threshold) & (scale_max <= threshold)
                split_mask = (grads >= cfg.densify_grad_threshold) & (scale_max > threshold)
                prune_mask = self.opacity < cfg.min_opacity
                if self._max_radii2d is not None:
                    prune_mask = prune_mask | (self._max_radii2d > cfg.max_screen_radius)
                # clones keep their original; splits replace it by two children
                keep = ~(split_mask | prune_mask)
                if self.num_primitives >= cfg.max_primitives:
                    clone_mask = torch.zeros_like(clone_mask)
                    split_mask = torch.zeros_like(split_mask)
                n_new = int(clone_mask.sum()) + 2 * int(split_mask.sum())
                if self.num_primitives + n_new > cfg.max_primitives:
                    room = max(0, cfg.max_primitives - self.num_primitives + int(prune_mask.sum()))
                    clone_idx = clone_mask.nonzero().squeeze(-1)[:room]
                    room -= clone_idx.numel()
                    split_idx = split_mask.nonzero().squeeze(-1)[: max(0, room // 2)]
                    clone_mask = torch.zeros_like(clone_mask)
                    split_mask = torch.zeros_like(split_mask)
                    clone_mask[clone_idx] = True
                    split_mask[split_idx] = True
                append = self._build_append(clone_mask, split_mask)
            self.rebuild_params(keep, append)
            self.reset_densification_stats()
            self._max_radii2d = None
        # opacity reset (official 2DGS resets after each densification round)
        if in_window and cfg.opacity_reset_interval > 0 and iteration > 0 and iteration % cfg.opacity_reset_interval == 0:
            with torch.no_grad():
                op = torch.sigmoid(self.p["opacity"])
                self.p["opacity"].data.copy_(inverse_sigmoid(op.clamp_max(0.01)))

    def _build_append(self, clone_mask: Tensor, split_mask: Tensor) -> dict:
        append = {}
        with torch.no_grad():
            tu, tv, _ = self.tangents()
            su, sv = self.scale()[:, 0], self.scale()[:, 1]
            idx_c = clone_mask.nonzero().squeeze(-1)
            idx_s = split_mask.nonzero().squeeze(-1)
            parts = {k: [] for k in self.p.keys()}
            if idx_c.numel():
                for k in self.p.keys():
                    parts[k].append(self.p[k].data[idx_c])
            if idx_s.numel():
                ns = idx_s.numel()
                means = self.p["means"].data[idx_s]
                eps = torch.randn(2 * ns, 2, device=means.device)
                su_rep = su[idx_s].repeat(2)
                sv_rep = sv[idx_s].repeat(2)
                tu_rep = tu[idx_s].repeat(2, 1)
                tv_rep = tv[idx_s].repeat(2, 1)
                offset = eps[:, 0:1] * su_rep[:, None] * tu_rep + eps[:, 1:2] * sv_rep[:, None] * tv_rep
                parts["means"].append(means.repeat(2, 1) + offset)
                parts["scaling"].append(
                    (self.p["scaling"].data[idx_s] - torch.log(torch.tensor(1.6, device=means.device))).repeat(2, 1)
                )
                parts["rotation"].append(self.p["rotation"].data[idx_s].repeat(2, 1))
                parts["features"].append(self.p["features"].data[idx_s].repeat(2, 1, 1))
                parts["opacity"].append(self.p["opacity"].data[idx_s].repeat(2, 1))
            for k, chunk_list in parts.items():
                if chunk_list:
                    append[k] = torch.cat(chunk_list, dim=0)
        return append
