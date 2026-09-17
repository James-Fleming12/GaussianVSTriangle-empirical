"""Image-quality metrics (PSNR / SSIM), implemented in pure PyTorch.

The SSIM matches the implementation used by the 3DGS / 2DGS / Triangle
Splatting codebases (11x11 Gaussian window, sigma=1.5), so numbers are
comparable with the papers' pipelines.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..train.losses import ssim


def psnr(pred: Tensor, target: Tensor) -> float:
    mse = ((pred - target) ** 2).mean().clamp_min(1e-12)
    return float(10.0 * torch.log10(1.0 / mse))


def ssim_value(pred: Tensor, target: Tensor) -> float:
    return float(ssim(pred, target))


def evaluate(pred: Tensor, target: Tensor) -> dict:
    return {"psnr": psnr(pred, target), "ssim": ssim_value(pred, target)}


def depth_rmse(pred_depth: Tensor, gt_depth: Tensor, gt_mask: Tensor | None = None) -> float:
    """RMSE between rendered and ground-truth depth on pixels that hit geometry."""
    if gt_mask is not None:
        sel = gt_mask.reshape(-1).bool()
        if int(sel.sum()) == 0:
            return float("nan")
        pred = pred_depth.reshape(-1)[sel]
        gt = gt_depth.reshape(-1)[sel]
    else:
        pred = pred_depth.reshape(-1)
        gt = gt_depth.reshape(-1)
    return float(((pred - gt) ** 2).mean().sqrt())
