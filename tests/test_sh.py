"""Spherical-harmonics tests (3DGS convention)."""

import math

import torch

from gvst.render.sh import SH_C0, eval_sh, rgb_to_sh, sh_to_rgb


def test_degree_zero_is_constant():
    rgb = torch.tensor([[0.2, 0.5, 0.9]])
    sh = rgb_to_sh(rgb)
    assert sh.shape == (1, 3)  # [N, C] before reshaping to [N, K, C]
    sh = sh[:, None, :]
    value = eval_sh(sh, torch.tensor([[1.0, 0.0, 0.0]]), degree=0) + 0.5  # colour shader
    assert torch.allclose(value, rgb, atol=1e-5)


def test_rgb_sh_roundtrip():
    rgb = torch.rand(5, 3)
    back = sh_to_rgb(rgb_to_sh(rgb))
    assert torch.allclose(rgb, back, atol=1e-6)


def test_degree_one_basis_values():
    """Y_1^0 .. by definition with the 3DGS constants: -C1*y, C1*z, -C1*x."""
    S = 0.4886025119029199
    sh = torch.zeros(1, 4, 3)
    sh[:, 1] = 1.0  # coefficient of Y_{1,-1}
    d = torch.tensor([[0.0, 1.0, 0.0]])
    assert torch.allclose(eval_sh(sh, d, degree=1), torch.tensor([[-S, -S, -S]]), atol=1e-6)
    sh = torch.zeros(1, 4, 3)
    sh[:, 2] = 1.0
    assert torch.allclose(eval_sh(sh, torch.tensor([[0.0, 0.0, 1.0]]), degree=1), torch.full((1, 3), S), atol=1e-6)


def test_orthonormality_of_basis_monte_carlo():
    """<Y_i, Y_j> over the sphere is 0 for i != j (Monte Carlo estimate)."""
    torch.manual_seed(0)
    dirs = torch.randn(20000, 3)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    k = 4
    values = torch.zeros(dirs.shape[0], k)
    for i in range(k):
        sh = torch.zeros(dirs.shape[0], k, 1)
        sh[:, i, 0] = 1.0
        values[:, i] = eval_sh(sh, dirs, degree=1)[:, 0]
    gram = values.T @ values / dirs.shape[0] * 4 * math.pi
    off = gram - torch.diag(torch.diag(gram))
    assert off.abs().max() < 0.1
    assert torch.allclose(torch.diag(gram), torch.ones(k), atol=0.1)
