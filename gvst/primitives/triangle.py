"""Triangle Splatting (Held et al., 2025).

Paper / official-code mapping:

* triangle primitive: 3 learnable vertices, opacity, sharpness ``sigma``
                                                        Sec. 3.1
* window function ``I(p) = ReLU(phi(p)/phi(s))^sigma``    Eq. (1)
* screen-space SDF ``phi = max_i (n_i . p + d_i)``        Sec. 3.1
* probabilistic (MCMC-style) densification by midpoint
  subdivision (4 children) or in-plane cloning (2 children) Sec. 3.2
* pruning by max blending weight / view coverage          Sec. 3.2
* size regularization (official code uses ``1/area``)     Sec. 3.3
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .base import SplatModel, fibonacci_directions, inverse_sigmoid, knn_mean_distance, random_rotation_matrices


class TriangleModel(SplatModel):
    method = "triangle"

    def __init__(
        self,
        points: Tensor,
        colors: Tensor,
        sh_degree: int,
        scene_extent: float,
        size_multiplier: float = 2.23,
        init_opacity: float = 0.28,
        init_sigma: float = 1.16,
        generator: torch.Generator | None = None,
    ):
        super().__init__(sh_degree)
        n = points.shape[0]
        dist = knn_mean_distance(points)
        self.scene_extent = scene_extent
        base_dirs = fibonacci_directions(3).to(points.device)  # [3,3]
        R = random_rotation_matrices(n, generator).to(points.device)
        radii = (size_multiplier * dist).clamp_min(1e-6)
        verts = points[:, None, :] + torch.einsum("nij,kj->nki", R, base_dirs) * radii[:, None, None]
        self.p["vertices"] = nn.Parameter(verts)
        self.p["features"] = nn.Parameter(torch.zeros(n, (sh_degree + 1) ** 2, 3))
        with torch.no_grad():
            self.p["features"][:, 0] = (colors - 0.5) / 0.28209479177387814
        self.p["opacity"] = nn.Parameter(inverse_sigmoid(torch.full((n, 1), float(init_opacity))))
        self.p["sigma"] = nn.Parameter(torch.full((n, 1), float(torch.log(torch.tensor(init_sigma - 0.01)))))
        self._lr_names = {
            "features": "feature_lr",
            "opacity": "tri_opacity_lr",
            "sigma": "tri_sigma_lr",
            "vertices": "tri_position_lr_init",
        }
        self._stats = None
        self._densify_round = 0

    # ---- activations --------------------------------------------------
    @property
    def num_primitives(self) -> int:
        return self.p["vertices"].shape[0]

    @property
    def primitive_centers(self) -> Tensor:
        return self.p["vertices"].mean(dim=1)

    @property
    def opacity(self) -> Tensor:
        return torch.sigmoid(self.p["opacity"]).squeeze(-1)

    def sigma(self) -> Tensor:
        return 0.01 + torch.exp(self.p["sigma"]).squeeze(-1)

    def triangle_area(self) -> Tensor:
        v = self.p["vertices"]
        cross = torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=1)
        return 0.5 * cross.norm(dim=1)

    def triangle_quality(self) -> dict:
        """Per-triangle shape statistics used to detect degenerate geometry."""
        v = self.p["vertices"]
        e0 = (v[:, 1] - v[:, 0]).norm(dim=1)
        e1 = (v[:, 2] - v[:, 1]).norm(dim=1)
        e2 = (v[:, 0] - v[:, 2]).norm(dim=1)
        edges = torch.stack([e0, e1, e2], dim=1)
        longest = edges.max(dim=1).values
        area = self.triangle_area()
        semi = edges.sum(dim=1) * 0.5
        inradius = area / semi.clamp_min(1e-12)
        aspect = longest / (2.0 * inradius).clamp_min(1e-12)
        # minimum interior angle via the law of cosines
        ang = []
        for i in range(3):
            a = v[:, (i + 1) % 3] - v[:, i]
            b = v[:, (i + 2) % 3] - v[:, i]
            cos = (a * b).sum(-1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)
            ang.append(torch.acos(cos.clamp(-1.0, 1.0)))
        min_angle = torch.stack(ang, dim=1).min(dim=1).values
        return {"area": area, "aspect": aspect, "min_angle": min_angle}

    # ---- training diagnostics ----------------------------------------
    def _diagnostic_param_name(self) -> str:
        return "vertices"

    def _extra_diagnostics(self) -> dict:
        with torch.no_grad():
            q = self.triangle_quality()
            area, aspect, ang = q["area"], q["aspect"], q["min_angle"]
            area_eps = (1e-3 * float(self.scene_extent)) ** 2
            op = self.opacity
            sig = self.sigma()
            return {
                "area_mean": float(area.mean()),
                "area_min": float(area.min()),
                "degenerate_frac": float((area < max(area_eps, 1e-10)).float().mean()),
                "sliver_frac": float((aspect > 15.0).float().mean()),
                "min_angle_mean_deg": float(ang.mean()) * 180.0 / 3.141592653589793,
                "sharp_frac": float((ang < 0.0174533).float().mean()),
                "aspect_mean": float(aspect.mean()),
                "opacity_mean": float(op.mean()),
                "opacity_p10": float(op.quantile(0.1)),
                "sigma_mean": float(sig.mean()),
            }

    # ---- rendering ----------------------------------------------------
    def start_view(self, camera) -> None:
        verts = self.p["vertices"]
        uvs, depths = camera.project(verts)  # [N,3,2], [N,3]
        valid = (depths > 0.1).all(dim=1)

        # enforce positive screen-space winding so outward edge normals are consistent
        area = 0.5 * (
            (uvs[:, 1, 0] - uvs[:, 0, 0]) * (uvs[:, 2, 1] - uvs[:, 0, 1])
            - (uvs[:, 2, 0] - uvs[:, 0, 0]) * (uvs[:, 1, 1] - uvs[:, 0, 1])
        )
        flip = area < 0
        q = uvs.clone()
        if flip.any():
            q[flip] = uvs[flip][:, [0, 2, 1]]

        e0 = q[:, 1] - q[:, 0]
        e1 = q[:, 2] - q[:, 1]
        e2 = q[:, 0] - q[:, 2]
        l0, l1, l2 = e0.norm(dim=1), e1.norm(dim=1), e2.norm(dim=1)
        eps = 1e-6
        normals = torch.stack(
            [
                torch.stack([e0[:, 1], -e0[:, 0]], -1),
                torch.stack([e1[:, 1], -e1[:, 0]], -1),
                torch.stack([e2[:, 1], -e2[:, 0]], -1),
            ],
            dim=1,
        )
        normals = normals / normals.norm(dim=-1, keepdim=True).clamp_min(eps)
        offsets = -(normals * q).sum(-1)  # [N,3]
        denom = (l0 + l1 + l2).clamp_min(eps)
        incenter = (l1[:, None] * q[:, 0] + l2[:, None] * q[:, 1] + l0[:, None] * q[:, 2]) / denom[:, None]
        phi_s = torch.einsum("nc,nc->n", normals[:, 0], incenter) + offsets[:, 0]
        for i in range(1, 3):
            phi_s = torch.maximum(phi_s, torch.einsum("nc,nc->n", normals[:, i], incenter) + offsets[:, i])
        phi_s = -phi_s.abs().clamp_min(1e-4)  # strictly negative (inside)

        # per-view statistics for probabilistic densification
        image_size = (q - q.mean(dim=1, keepdim=True)).norm(dim=-1).max(dim=1).values
        inradius = -phi_s

        # world-space plane normal (camera-facing) for the normal-consistency loss
        v_cam = camera.world_to_camera(verts)
        n3 = torch.cross(v_cam[:, 1] - v_cam[:, 0], v_cam[:, 2] - v_cam[:, 0], dim=1)
        n3 = n3 / n3.norm(dim=-1, keepdim=True).clamp_min(eps)
        n3_world = n3 @ camera.R_cw.T
        to_cam = camera.center[None, :] - self.primitive_centers
        n3_world = torch.where(((n3_world * to_cam).sum(-1, keepdim=True) < 0), -n3_world, n3_world)

        self._view_ctx = {
            "q": q,
            "normals": normals,
            "offsets": offsets,
            "incenter": incenter,
            "phi_s": phi_s,
            "valid": valid,
            "depth": depths.mean(dim=1),
            "sigma": self.sigma(),
            "opacity": torch.sigmoid(self.p["opacity"]).squeeze(-1),
            "image_size": image_size,
            "inradius": inradius,
            "normal3": n3_world,
            "v0_cam": v_cam[:, 0],
        }

    def screen_chunk(self, pixels: Tensor, camera) -> dict:
        ctx = self._view_ctx
        p = pixels.shape[0]
        # signed distances to the three edges: [N,3,p]
        l = torch.einsum("pc,nkc->nkp", pixels, ctx["normals"]) + ctx["offsets"][:, :, None]
        phi = l.max(dim=1).values
        # log-space window: ReLU(phi/phi_s)^sigma, clamped for finite gradients
        ratio = (phi / ctx["phi_s"][:, None]).clamp(1e-6, 1.0)
        window = torch.exp(ctx["sigma"][:, None] * torch.log(ratio))
        alpha = ctx["opacity"][:, None] * window
        alpha = torch.where(ctx["valid"][:, None], alpha, torch.zeros_like(alpha))
        alpha = torch.where(alpha < 1e-5, torch.zeros_like(alpha), alpha)

        # per-pixel plane depth (only meaningful where the pixel is inside the triangle)
        d_cam = camera.ray(pixels)[1] @ camera.R_cw
        n3 = self._view_ctx["normal3"]
        denom = torch.einsum("pc,nc->np", d_cam, n3)
        denom_safe = denom.abs().clamp_min(1e-6) * torch.where(denom >= 0, 1.0, -1.0)
        t = (ctx["v0_cam"] * n3).sum(-1)[:, None] / denom_safe
        t = torch.where((ctx["valid"][:, None] & (denom.abs() > 1e-6)), t.clamp_min(0.0), torch.zeros_like(t))
        return {"alpha": alpha, "depth": t, "normal": n3}

    # ---- probabilistic densification (Sec. 3.2) -----------------------
    def accumulate_densification_stats(self, render_out=None) -> None:
        n = self.num_primitives
        if self._stats is None:
            self._stats = {
                "importance": torch.zeros(n, device=self.p["vertices"].device),
                "image_size": torch.zeros(n, device=self.p["vertices"].device),
                "coverage": torch.zeros(n, device=self.p["vertices"].device),
            }
        if render_out is not None and render_out.weights is not None:
            imp = render_out.weights.detach().max(dim=1).values
            self._stats["importance"] = torch.maximum(self._stats["importance"], imp)
        size = self._view_ctx.get("image_size")
        if size is not None:
            self._stats["image_size"] = torch.maximum(self._stats["image_size"], size.detach())
        inradius = self._view_ctx.get("inradius")
        if inradius is not None:
            self._stats["coverage"] = self._stats["coverage"] + (inradius.detach() > 1.0).float()

    def _dead_mask(self, iteration: int) -> Tensor:
        cfg = self._cfg
        dead = (self._stats["importance"] < cfg.tri_importance_threshold) | (
            self.opacity <= cfg.tri_opacity_dead
        )
        if iteration > 1000:
            dead = dead | (self._stats["coverage"] < 2)
        if bool(dead.all()) and dead.numel() > 0:
            # never prune the whole representation: keep the most important shape
            dead = dead.clone()
            dead[int(self._stats["importance"].argmax())] = False
        return dead

    def densify(self, iteration: int) -> None:
        """Probabilistic densification, then periodic pruning after ``until``."""
        cfg = self._cfg
        if self._stats is None or self.optimizer is None:
            return
        if cfg.tri_densify_mode == "none":
            return
        interval = cfg.tri_densification_interval
        if interval <= 0 or iteration % interval != 0:
            return
        dead = self._dead_mask(iteration)

        if not (cfg.tri_densify_from_iter <= iteration < cfg.tri_densify_until_iter):
            # post-densification phase: prune only (same criterion as the reference code)
            if dead.any():
                self.rebuild_params(~dead, None)
                self._stats = None
            return

        n = self.num_primitives
        target = min(cfg.tri_max_shapes, int(cfg.tri_growth * n))
        num_add = max(0, target - n) + int(dead.sum())
        if num_add <= 0:
            return

        with torch.no_grad():
            probs = self.opacity if self._densify_round % 2 == 0 else 1.0 / self.sigma()
            probs = probs.masked_fill(dead, 0.0).clamp_min(0.0)
            nz = int((probs > 0).sum())
            if nz:
                k = min(num_add, nz)
                if cfg.tri_densify_mode == "deterministic":
                    sel = torch.topk(probs, k).indices
                else:
                    sel = torch.multinomial(probs, k, replacement=False)
            else:
                sel = torch.empty(0, dtype=torch.long, device=probs.device)
            if sel.numel() and self._stats["image_size"][sel].numel():
                big = self._stats["image_size"][sel] > cfg.tri_split_size
                costs = torch.where(big, torch.full_like(sel, 3.0), torch.ones_like(sel, dtype=torch.float32))
                cum = torch.cumsum(costs, dim=0)
                reached = (cum >= num_add).nonzero(as_tuple=True)[0]
                cutoff = int(reached[0].item()) + 1 if reached.numel() else sel.numel()
                sel = sel[:cutoff]
            self._densify_round += 1
            append = self._build_append(sel)
        keep = ~dead
        keep[sel] = False
        self.rebuild_params(keep, append)
        self._stats = None

    def _build_append(self, sel: Tensor) -> dict:
        cfg = self._cfg
        if sel.numel() == 0:
            return {}
        verts = self.p["vertices"].data[sel]
        big = self._stats["image_size"][sel] > cfg.tri_split_size
        parts = {k: [] for k in self.p.keys()}

        idx_big = sel[big]
        idx_small = sel[~big]

        if idx_big.numel():
            v = self.p["vertices"].data[idx_big]
            A, B, C = v[:, 0], v[:, 1], v[:, 2]
            M_ab, M_ac, M_bc = (A + B) / 2, (A + C) / 2, (B + C) / 2
            children = torch.stack(
                [torch.stack([A, M_ab, M_ac], 1), torch.stack([B, M_ab, M_bc], 1),
                 torch.stack([C, M_ac, M_bc], 1), torch.stack([M_ab, M_ac, M_bc], 1)],
                dim=0,
            )  # [4, nb, 3, 3]
            parts["vertices"].append(children.permute(1, 0, 2, 3).reshape(-1, 3, 3))
            parts["features"].append(self.p["features"].data[idx_big].repeat_interleave(4, dim=0))
            parts["opacity"].append(self.p["opacity"].data[idx_big].repeat_interleave(4, dim=0))
            parts["sigma"].append(self.p["sigma"].data[idx_big].repeat_interleave(4, dim=0))
        if idx_small.numel():
            v = self.p["vertices"].data[idx_small]
            n = v.shape[0]
            bbox = v.max(dim=1).values - v.min(dim=1).values
            noise = (torch.rand(n, 1, 3, device=v.device) - 0.5) * (bbox * cfg.tri_max_noise_factor).unsqueeze(1)
            nrm = torch.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0], dim=1)
            nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-8)
            noise = noise - (noise * nrm.unsqueeze(1)).sum(-1, keepdim=True) * nrm.unsqueeze(1)
            parts["vertices"].append(torch.cat([v, v + noise], dim=0))
            o = self.opacity[idx_small].clamp(0, 1 - 1e-6)
            o_child = inverse_sigmoid(1.0 - (1.0 - o).pow(0.5))
            parts["features"].append(self.p["features"].data[idx_small].repeat(2, 1, 1))
            parts["opacity"].append(torch.cat([o_child[:, None], o_child[:, None]], dim=0))
            parts["sigma"].append(self.p["sigma"].data[idx_small].repeat(2, 1))
        return {k: torch.cat(v, dim=0) for k, v in parts.items() if v}

    def prune_final(self) -> None:
        if self._stats is None or self.optimizer is None:
            return
        if self._cfg.tri_densify_mode == "none":
            # ablation: no growth and no pruning, so the initialisation is kept
            self._stats = None
            return
        dead = self._dead_mask(self._cfg.iterations)
        if dead.any():
            self.rebuild_params(~dead, None)
        self._stats = None
