"""Training losses shared by all methods (plus the paper-specific terms)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    return (pred - target).abs().mean()


def _gaussian_window(window_size: int = 11, sigma: float = 1.5) -> Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return g[:, None] @ g[None, :]


def ssim(pred: Tensor, target: Tensor, window_size: int = 11, sigma: float = 1.5) -> Tensor:
    """Standard structural similarity used throughout the 3DGS/2DGS papers."""
    c = pred.shape[0]
    window = _gaussian_window(window_size, sigma).to(pred.device, pred.dtype)
    window = window.expand(c, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    mu1 = F.conv2d(pred.unsqueeze(0), window, padding=pad, groups=c)
    mu2 = F.conv2d(target.unsqueeze(0), window, padding=pad, groups=c)
    sigma1 = F.conv2d((pred * pred).unsqueeze(0), window, padding=pad, groups=c) - mu1 * mu1
    sigma2 = F.conv2d((target * target).unsqueeze(0), window, padding=pad, groups=c) - mu2 * mu2
    sigma12 = F.conv2d((pred * target).unsqueeze(0), window, padding=pad, groups=c) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1 * mu1 + mu2 * mu2 + c1) * (sigma1 + sigma2 + c2)
    )
    return ssim_map.mean()


def dssim(pred: Tensor, target: Tensor) -> Tensor:
    return 1.0 - ssim(pred, target)


def photometric_loss(pred: Tensor, target: Tensor, lambda_dssim: float = 0.2) -> Tensor:
    """3DGS Eq. (7): ``(1-lambda) L1 + lambda (1-SSIM)``."""
    return (1.0 - lambda_dssim) * l1_loss(pred, target) + lambda_dssim * dssim(pred, target)


def triangle_size_loss(area: Tensor) -> Tensor:
    """Triangle Splatting size term (official code: ``1 / mean area``)."""
    if area.numel() == 0:
        return area.sum() * 0.0
    return 1.0 / area.mean().clamp_min(1e-8)
