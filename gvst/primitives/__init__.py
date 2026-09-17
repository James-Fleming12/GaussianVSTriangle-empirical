"""Primitive models: 3DGS, 2DGS, Triangle Splatting."""

from __future__ import annotations

import torch

from .base import SplatModel
from .gaussian_2d import Gaussian2D
from .gaussian_3d import Gaussian3D
from .triangle import TriangleModel

METHODS = ("3dgs", "2dgs", "triangle")


def build_model(method: str, points, colors, cfg, scene_extent: float, seed: int = 0) -> SplatModel:
    gen = torch.Generator().manual_seed(seed)
    if method == "3dgs":
        return Gaussian3D(points, colors, cfg.sh_degree, scene_extent, init_opacity=cfg.init_opacity_3dgs)
    if method == "2dgs":
        return Gaussian2D(points, colors, cfg.sh_degree, scene_extent, init_opacity=cfg.init_opacity_2dgs)
    if method == "triangle":
        return TriangleModel(
            points,
            colors,
            cfg.sh_degree,
            scene_extent,
            size_multiplier=cfg.tri_size_init,
            init_opacity=cfg.tri_set_opacity,
            init_sigma=cfg.tri_set_sigma,
            generator=gen,
        )
    raise KeyError(f"unknown method {method!r}")
