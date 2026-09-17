"""Training / benchmark configuration.

Paper hyperparameters are kept as defaults here and are referenced where they
come from.  For the small synthetic benchmark the *schedules* (densification
windows, regularization warm-ups, SH degree ramp) are linearly rescaled to the
benchmark iteration budget so that every method sees the same budget; the
rescaling is applied by :func:`scaled_train_config`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass
class TrainConfig:
    method: str = "3dgs"  # "3dgs" | "2dgs" | "triangle"
    iterations: int = 3_000
    seed: int = 0
    sh_degree: int = 2

    # ---- shared photometric loss ----
    lambda_dssim: float = 0.2  # 3DGS Eq. (7)

    # ---- shared optimizer learning rates (3DGS / 2DGS) ----
    position_lr_init: float = 1.6e-4
    position_lr_final: float = 1.6e-6
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 2.5e-3
    opacity_lr: float = 5e-2
    scaling_lr: float = 5e-3
    rotation_lr: float = 1e-3

    # ---- adaptive density control (3DGS Sec. 5.2 / 2DGS Sec. 6.1) ----
    densify_from_iter: int = 500
    densify_until_iter: int = 2_500
    densification_interval: int = 100
    densify_grad_threshold: float = 2e-4
    percent_dense: float = 0.01
    opacity_reset_interval: int = 3_000
    min_opacity: float = 5e-3  # 3DGS uses 0.005; 2DGS uses 0.05
    max_primitives: int = 8_000
    max_screen_radius: float = 20.0  # 2DGS prunes splats larger than this

    # ---- 2DGS regularizers (Eq. 13-16) ----
    # The 2DGS reference evaluation for bounded synthetic (NeRF-synthetic)
    # scenes uses lambda_dist=0 and lambda_normal=0 (scripts/nerf_eval.py);
    # the regularizers target geometry (DTU: 1000, T&T: 100/10) and are
    # destabilizing at this small scale / short budget.  Kept configurable so
    # the effect can be measured separately (see README).
    lambda_dist: float = 0.0
    lambda_normal: float = 0.0
    dist_from_iter: int = 300
    normal_from_iter: int = 750

    # ---- Triangle Splatting (official code + paper Table 5) ----
    tri_position_lr_init: float = 1.8e-3
    tri_position_lr_final: float = 1.8e-5
    tri_sigma_lr: float = 8e-4
    tri_opacity_lr: float = 1.4e-2
    tri_lambda_opacity: float = 5.5e-3
    tri_lambda_size: float = 1e-8
    tri_lambda_normals: float = 1e-4
    tri_lambda_dist: float = 0.0
    tri_densify_from_iter: int = 500
    tri_densify_until_iter: int = 25_000
    tri_densification_interval: int = 500
    tri_growth: float = 1.3
    tri_opacity_dead: float = 0.014
    tri_importance_threshold: float = 0.022
    tri_split_size: float = 24.0
    tri_max_shapes: int = 8_000
    tri_iteration_mesh: int = 5_000
    tri_set_opacity: float = 0.28
    tri_set_sigma: float = 1.16
    tri_size_init: float = 2.23
    tri_max_noise_factor: float = 1.5

    # ---- initialization ----
    init_opacity_3dgs: float = 0.1  # 3DGS initializes to 0.1
    init_opacity_2dgs: float = 0.1

    def with_method(self, method: str) -> "TrainConfig":
        cfg = replace(self, method=method)
        if method == "2dgs":
            cfg.min_opacity = 0.05  # 2DGS prunes below 0.05 (Sec. 6.1)
        return cfg


def scaled_train_config(cfg: TrainConfig) -> TrainConfig:
    """Adapt the paper schedules to ``cfg.iterations``.

    The papers train for 30k iterations; the benchmark runs 3k.  Total steps
    are reduced, but *absolute* cadences that the optimizer dynamics depend on
    (densify every 100/500 iterations, opacity reset every 3000, regularization
    warm-ups) are kept at their paper values.  Densification windows keep the
    papers' fractions of the run so a short run still exercises the whole
    pipeline (initialization -> densify -> refine).
    """
    t = cfg.iterations
    cfg = replace(
        cfg,
        position_lr_max_steps=t,
        # 3DGS/2DGS: densify from 500 to 15000/30000 of the run, every 100 iters
        densify_from_iter=min(500, max(1, t // 6)),
        densify_until_iter=int(0.50 * t),
        densification_interval=100,
        # 2DGS regularizer warm-ups (paper absolute: 3000 / 7000)
        dist_from_iter=3000,
        normal_from_iter=7000,
        # Triangle Splatting: densify from 500 to 25000/30000, every 500 iters;
        # normal loss enters at 5000/30000 of the run
        tri_densify_from_iter=min(500, max(1, t // 6)),
        tri_densify_until_iter=int(0.833 * t),
        tri_densification_interval=500,
        tri_iteration_mesh=int(0.167 * t),
    )
    return cfg


@dataclass
class SceneSpec:
    name: str
    description: str
    views: int = 8
    width: int = 64
    height: int = 64
    fov_deg: float = 45.0
    radius: float = 3.0
    elevations_deg: tuple = (20.0, 25.0, 30.0)
    init_points: int = 3000
    seed: int = 0


@dataclass
class BenchmarkConfig:
    scenes: list = field(default_factory=list)
    methods: tuple = ("3dgs", "2dgs", "triangle")
    train_config: TrainConfig = field(default_factory=TrainConfig)
    folds: Optional[list] = None  # held-out view indices; None -> all views
    seeds: tuple = (0,)
    output_dir: str = "results"
    device: str = "cuda"
    eval_every: int = 200
    save_images: bool = True
