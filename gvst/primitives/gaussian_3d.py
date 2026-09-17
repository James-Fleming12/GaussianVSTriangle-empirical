"""3D Gaussian Splatting (Kerbl et al., 2023).

Paper mapping:

* covariance ``Sigma = R S S^T R^T``                  Sec. 3 / Eq. (1-2)
* screen projection ``Sigma' = J W Sigma W^T J^T``    Sec. 4 / Eq. (3-6)
* alpha blending                                    Sec. 4 / Eq. (3)
* adaptive density control (clone / split / prune)  Sec. 5.2
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..render.raster import colors_from_sh
from .base import (
    SplatModel,
    inverse_sigmoid,
    knn_mean_distance,
    quat_to_rot,
)


class Gaussian3D(SplatModel):
    method = "3dgs"

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
        self.p["scaling"] = nn.Parameter(torch.log(dist.clamp_min(1e-6))[:, None].repeat(1, 3))
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
        # official ADC uses the view-space x,y positional gradient only
        self._grad_dims = (0, 2)

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

    def covariance_world(self) -> Tensor:
        R = quat_to_rot(self.p["rotation"])
        S = self.scale()
        M = R * S[:, None, :]
        return M @ M.transpose(1, 2)

    # ---- training diagnostics ----------------------------------------
    def _diagnostic_param_name(self) -> str:
        return "means"

    def _extra_diagnostics(self) -> dict:
        with torch.no_grad():
            s = self.scale()
            s_max = s.max(dim=1).values
            op = self.opacity
            return {
                "scale_min": float(s.min()),
                "scale_max": float(s_max.max()),
                "scale_mean": float(s_max.mean()),
                "tiny_frac": float((s_max < 1e-3).float().mean()),
                "opacity_mean": float(op.mean()),
                "opacity_p10": float(op.quantile(0.1)),
            }

    # ---- rendering ----------------------------------------------------
    def start_view(self, camera) -> None:
        ctx = {}
        pc = camera.world_to_camera(self.p["means"])
        if self.p["means"].requires_grad and torch.is_grad_enabled():
            pc.retain_grad()
            ctx["grad_ref"] = pc
        depth = -pc[:, 2]
        safe = depth.clamp_min(0.05)
        fx, fy, cx, cy = camera.fx, camera.fy, camera.cx, camera.cy
        mu2d = torch.stack([cx + fx * pc[:, 0] / safe, cy - fy * pc[:, 1] / safe], dim=-1)
        # Jacobian of the perspective projection (OpenGL-like, y flipped)
        n = self.num_primitives
        J = torch.zeros(n, 2, 3, device=pc.device, dtype=pc.dtype)
        J[:, 0, 0] = fx / safe
        J[:, 0, 2] = fx * pc[:, 0] / safe**2
        J[:, 1, 1] = -fy / safe
        J[:, 1, 2] = -fy * pc[:, 1] / safe**2
        W = camera.R_cw.T
        cov_cam = W @ self.covariance_world() @ W.T
        cov2d = J @ cov_cam @ J.transpose(1, 2)
        cov2d[:, 0, 0] += 0.3  # low-pass filter, 3DGS Sec. 4
        cov2d[:, 1, 1] += 0.3
        det = (cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2).clamp_min(1e-12)
        inv00 = cov2d[:, 1, 1] / det
        inv01 = -cov2d[:, 0, 1] / det
        inv11 = cov2d[:, 0, 0] / det
        ctx.update(
            pc=pc,
            depth=depth,
            valid=depth > 0.1,
            mu2d=mu2d,
            inv=torch.stack([torch.stack([inv00, inv01], -1), torch.stack([inv01, inv11], -1)], -2),
            opacity=torch.sigmoid(self.p["opacity"]).squeeze(-1),
        )
        self._view_ctx = ctx

    def screen_chunk(self, pixels: Tensor, camera) -> dict:
        ctx = self._view_ctx
        mu = ctx["mu2d"]
        diff = pixels[None, :, :] - mu[:, None, :]  # [N,p,2]
        inv = ctx["inv"]
        e = (
            diff[..., 0] ** 2 * inv[:, None, 0, 0]
            + 2 * diff[..., 0] * diff[..., 1] * inv[:, None, 0, 1]
            + diff[..., 1] ** 2 * inv[:, None, 1, 1]
        )
        alpha = ctx["opacity"][:, None] * torch.exp(-0.5 * e)
        p = pixels.shape[0]
        valid = ctx["valid"][:, None]
        alpha = torch.where(valid, alpha, torch.zeros_like(alpha))
        alpha = torch.where(alpha < 1e-5, torch.zeros_like(alpha), alpha)
        depth = ctx["depth"][:, None].expand(-1, p)
        depth = torch.where(valid, depth, torch.zeros_like(depth))
        return {"alpha": alpha, "depth": depth}

    # ---- adaptive density control (Sec. 5.2) --------------------------
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
                prune_mask = (self.opacity < cfg.min_opacity) | (scale_max > 0.1 * self.scene_extent)
                # clones keep their original (official 3DGS appends copies); splits
                # replace the original by two children
                keep = ~(split_mask | prune_mask)
                if self.num_primitives >= cfg.max_primitives:
                    # budget reached: freeze growth (pruning still applies) so the
                    # optimizer can settle instead of churning primitives
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
        # opacity reset (official code resets while still inside the densify window)
        if in_window and cfg.opacity_reset_interval > 0 and iteration > 0 and iteration % cfg.opacity_reset_interval == 0:
            with torch.no_grad():
                op = torch.sigmoid(self.p["opacity"])
                self.p["opacity"].data.copy_(inverse_sigmoid(op.clamp_max(0.01)))

    def _build_append(self, clone_mask: Tensor, split_mask: Tensor) -> dict:
        append = {}
        with torch.no_grad():
            R = quat_to_rot(self.p["rotation"])
            S = self.scale()
            M = R * S[:, None, :]
            idx_c = clone_mask.nonzero().squeeze(-1)
            idx_s = split_mask.nonzero().squeeze(-1)
            parts = {k: [] for k in self.p.keys()}
            if idx_c.numel():
                for k in self.p.keys():
                    parts[k].append(self.p[k].data[idx_c])
            if idx_s.numel():
                ns = idx_s.numel()
                means = self.p["means"].data[idx_s]
                eps = torch.randn(2 * ns, 3, device=means.device)
                Ms = M[idx_s].repeat(2, 1, 1)
                sampled = means.repeat(2, 1) + torch.bmm(Ms, eps.unsqueeze(-1)).squeeze(-1)
                parts["means"].append(sampled)
                parts["scaling"].append((self.p["scaling"].data[idx_s] - torch.log(torch.tensor(1.6, device=means.device))).repeat(2, 1))
                parts["rotation"].append(self.p["rotation"].data[idx_s].repeat(2, 1))
                parts["features"].append(self.p["features"].data[idx_s].repeat(2, 1, 1))
                parts["opacity"].append(self.p["opacity"].data[idx_s].repeat(2, 1))
            for k, chunk_list in parts.items():
                if chunk_list:
                    append[k] = torch.cat(chunk_list, dim=0)
        return append
