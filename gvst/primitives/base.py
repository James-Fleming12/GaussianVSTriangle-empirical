"""Shared machinery for all three primitives.

The models keep their optimizable tensors in an ``nn.ParameterDict`` (``self.p``)
so that all of them can reuse the same optimizer-state surgery when primitives
are cloned, split or pruned (the same bookkeeping the reference implementations
perform by hand).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..render.raster import colors_from_sh


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def quat_to_rot(q: Tensor) -> Tensor:
    """``q [N,4]`` (w,x,y,z, unnormalized) -> ``R [N,3,3]``."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    o = torch.ones_like(w)
    zz = torch.zeros_like(w)
    R = torch.stack(
        [
            o - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), o - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), o - 2 * (x * x + y * y),
        ],
        dim=-1,
    )
    return R.reshape(-1, 3, 3)


def inverse_sigmoid(x: Tensor) -> Tensor:
    return torch.log(x.clamp(1e-6, 1 - 1e-6) / (1 - x.clamp(1e-6, 1 - 1e-6))).clamp(-20, 20)


def knn_mean_distance(points: Tensor, k: int = 3) -> Tensor:
    """Mean Euclidean distance to the ``k`` nearest neighbours (excluding self)."""
    n = points.shape[0]
    if n < 2:
        return torch.ones(n, device=points.device, dtype=points.dtype)
    d = torch.cdist(points, points)
    d[torch.arange(n), torch.arange(n)] = float("inf")
    k = min(k, n - 1)
    vals, _ = d.topk(k, largest=False, dim=1)
    return vals.mean(dim=1)


def fibonacci_directions(n: int) -> Tensor:
    dirs = []
    for i in range(n):
        z = 1.0 - 2.0 * i / max(n - 1, 1)
        r = math.sqrt(max(1.0 - z * z, 0.0))
        theta = math.pi * (3.0 - math.sqrt(5.0)) * i
        dirs.append([r * math.cos(theta), r * math.sin(theta), z])
    return torch.tensor(dirs, dtype=torch.float32)


def random_rotation_matrices(n: int, generator: Optional[torch.Generator] = None) -> Tensor:
    """Uniform random rotations ``[n,3,3]`` via quaternion sampling."""
    q = torch.randn(n, 4, generator=generator)
    return quat_to_rot(q)


# ---------------------------------------------------------------------------
# base model
# ---------------------------------------------------------------------------


class SplatModel(nn.Module):
    """Base class implementing optimizer surgery and the render interface."""

    method: str = "base"

    def __init__(self, sh_degree: int):
        super().__init__()
        self.max_sh_degree = int(sh_degree)
        self.active_sh_degree = int(sh_degree)
        self.scene_extent = 1.0
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.p = nn.ParameterDict()
        self._lr_names: dict = {}
        self._grad_accum: Optional[Tensor] = None
        self._grad_count = 0
        self._view_ctx: dict = {}
        self._generator: Optional[torch.Generator] = None

    # ---- required properties -----------------------------------------
    @property
    def num_primitives(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def primitive_centers(self) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def screen_chunk(self, pixels: Tensor, camera) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    # ---- colors / sorting --------------------------------------------
    def colors_for_view(self, camera) -> Tensor:
        centers = self.primitive_centers
        view = centers - camera.center
        dirs = view / view.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return colors_from_sh(self.p["features"], dirs, self.active_sh_degree)

    def start_view(self, camera) -> None:
        """Per-view setup; subclasses store tensors whose gradients drive ADC."""
        self._view_ctx = {}

    def sort_order(self, camera) -> Tensor:
        centers = self.primitive_centers
        pc = camera.world_to_camera(centers)
        depth = -pc[:, 2]
        return torch.argsort(depth)

    # ---- optimizer ---------------------------------------------------
    def setup_optimizer(self, cfg) -> None:
        self.scene_extent = float(self.scene_extent)
        groups = []
        for name, lr_key in self._lr_names.items():
            groups.append({"params": [self.p[name]], "lr": float(getattr(cfg, lr_key)), "name": name})
        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self._cfg = cfg

    def update_learning_rate(self, iteration: int) -> None:
        if self.optimizer is None:
            return
        for group, lr_key in zip(self.optimizer.param_groups, self._lr_names.values()):
            if lr_key == "position_lr_init":
                lr = self._position_lr(iteration) * self.scene_extent
            elif lr_key == "tri_position_lr_init":
                lr = self._tri_position_lr(iteration)
            else:
                lr = float(getattr(self._cfg, lr_key))
            group["lr"] = lr

    def _position_lr(self, iteration: int) -> float:
        """3DGS exponential position schedule (geometry-aware, delay 0.01)."""
        cfg = self._cfg
        if iteration < 0:
            return cfg.position_lr_init
        alpha = min(1.0, iteration / max(1, cfg.position_lr_max_steps))
        delay_rate = cfg.position_lr_delay_mult + (1.0 - cfg.position_lr_delay_mult) * math.sin(
            0.5 * math.pi * alpha
        )
        log_lerp = math.exp(math.log(cfg.position_lr_init) * (1 - alpha) + math.log(cfg.position_lr_final) * alpha)
        return delay_rate * log_lerp

    def _tri_position_lr(self, iteration: int) -> float:
        cfg = self._cfg
        alpha = min(1.0, iteration / max(1, cfg.iterations))
        return cfg.tri_position_lr_final + (cfg.tri_position_lr_init - cfg.tri_position_lr_final) * (1 - alpha)

    # ---- optimizer-state-preserving add/remove -----------------------
    def rebuild_params(self, keep_mask: Optional[Tensor], append: Optional[dict] = None) -> None:
        """Replace parameters by ``old[keep_mask] (+ append)`` preserving Adam state."""
        append = append or {}
        for group in self.optimizer.param_groups:
            name = group["name"]
            old = group["params"][0]
            kept = old.detach() if keep_mask is None else old.detach()[keep_mask]
            if name in append:
                new_t = torch.cat([kept, append[name].detach()], dim=0).contiguous()
            else:
                new_t = kept.contiguous()
            new_param = nn.Parameter(new_t)
            state = self.optimizer.state.pop(old, None)
            if state is not None:
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        v = state[key] if keep_mask is None else state[key][keep_mask]
                        if name in append:
                            v = torch.cat([v, torch.zeros_like(append[name])], dim=0)
                        state[key] = v.contiguous()
                self.optimizer.state[new_param] = state
            group["params"][0] = new_param
            self.p[name] = new_param
        self._grad_accum = None

    # ---- dense-gradient statistics (3DGS / 2DGS ADC) -----------------
    def reset_densification_stats(self) -> None:
        self._grad_accum = None
        self._grad_count = 0

    def accumulate_densification_stats(self, render_out=None) -> None:
        ref = self._view_ctx.get("grad_ref")
        if ref is None or ref.grad is None:
            return
        g = ref.grad.detach()
        dims = getattr(self, "_grad_dims", None)
        if dims is not None:
            g = g[..., dims[0] : dims[1]]
        if g.dim() > 1:
            g = g.norm(dim=-1)
        if self._grad_accum is None or self._grad_accum.shape[0] != g.shape[0]:
            self._grad_accum = g.clone()
        else:
            self._grad_accum = self._grad_accum + g
        self._grad_count += 1

    @property
    def mean_grad(self) -> Optional[Tensor]:
        if self._grad_accum is None:
            return None
        return self._grad_accum / max(1, self._grad_count)

    # ---- densification hooks -----------------------------------------
    def densify(self, iteration: int) -> None:
        pass

    def prune_final(self) -> None:
        pass

    # ---- misc --------------------------------------------------------
    @torch.no_grad()
    def set_sh_degree(self, degree: int) -> None:
        self.active_sh_degree = min(int(degree), self.max_sh_degree)
