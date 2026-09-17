"""Real spherical harmonics, following the 3DGS convention (Kerbl et al. 2023).

Coefficients are stored as ``[N, K, 3]`` with ``K = (degree + 1)^2``.  The
evaluation returns colours in the same "SH space" as the 3DGS shader, i.e.
``rgb = clamp_min(sh_eval + 0.5, 0)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (
    0.5900435899266435,
    2.890611442640554,
    0.4570457994644658,
    0.3731763325901154,
    0.4570457994644658,
    1.445305721320277,
    0.5900435899266435,
)


def sh_num_coeffs(degree: int) -> int:
    return (degree + 1) ** 2


def eval_sh(sh: Tensor, dirs: Tensor, degree: int) -> Tensor:
    """Evaluate SH ``sh [N,K,3]`` along unit ``dirs [N,3] -> [N,3]``."""
    n = sh.shape[0]
    result = SH_C0 * sh[:, 0]
    if degree >= 1:
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
        result = result - SH_C1 * y[:, None] * sh[:, 1] + SH_C1 * z[:, None] * sh[:, 2] - SH_C1 * x[:, None] * sh[:, 3]
    if degree >= 2:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = (
            result
            + SH_C2[0] * xy[:, None] * sh[:, 4]
            + SH_C2[1] * yz[:, None] * sh[:, 5]
            + SH_C2[2] * (2 * zz - xx - yy)[:, None] * sh[:, 6]
            + SH_C2[3] * xz[:, None] * sh[:, 7]
            + SH_C2[4] * (xx - yy)[:, None] * sh[:, 8]
        )
    if degree >= 3:
        result = (
            result
            + SH_C3[0] * y * (3 * xx - yy)[:, None] * sh[:, 9]
            + SH_C3[1] * xy * z[:, None] * sh[:, 10]
            + SH_C3[2] * y * (4 * zz - xx - yy)[:, None] * sh[:, 11]
            + SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy)[:, None] * sh[:, 12]
            + SH_C3[4] * x * (4 * zz - xx - yy)[:, None] * sh[:, 13]
            + SH_C3[5] * z * (xx - yy)[:, None] * sh[:, 14]
            + SH_C3[6] * x * (xx - 3 * yy)[:, None] * sh[:, 15]
        )
    assert result.shape[0] == n
    return result


def rgb_to_sh(rgb: Tensor) -> Tensor:
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh: Tensor) -> Tensor:
    return sh * SH_C0 + 0.5
