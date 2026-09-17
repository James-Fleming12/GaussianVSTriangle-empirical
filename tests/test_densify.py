"""Densification tests: clone/split semantics, caps, optimizer-state transfer."""

import torch

from gvst.primitives import build_model


def _toy(method, cfg, n=8, seed=0):
    torch.manual_seed(seed)
    pts = torch.rand(n, 3) * 0.5
    cols = torch.rand(n, 3)
    return build_model(method, pts, cols, cfg, scene_extent=10.0, seed=seed)


def test_3dgs_clone_and_split(tiny_cfg):
    cfg = tiny_cfg.with_method("3dgs")
    cfg.densify_from_iter = 0
    cfg.densify_until_iter = 100
    cfg.densification_interval = 1
    model = _toy("3dgs", cfg)
    model.setup_optimizer(cfg)
    n0 = model.num_primitives
    grads = torch.zeros(n0)
    grads[0] = 1.0  # one small primitive -> clone
    with torch.no_grad():
        model.p["scaling"][0] = torch.log(torch.tensor(0.001))
    model._grad_accum = grads
    model._grad_count = 1
    model.densify(0)
    assert model.num_primitives == n0 + 1


def test_3dgs_split_halves_scale(tiny_cfg):
    cfg = tiny_cfg.with_method("3dgs")
    cfg.densify_from_iter = 0
    cfg.densify_until_iter = 100
    cfg.densification_interval = 1
    model = _toy("3dgs", cfg)
    model.setup_optimizer(cfg)
    with torch.no_grad():
        model.p["scaling"][0] = torch.log(torch.tensor(1.0))  # large -> split
    grads = torch.zeros(model.num_primitives)
    grads[0] = 1.0
    model._grad_accum = grads
    model._grad_count = 1
    before = model.scale()[0].max().item()
    model.densify(0)
    assert model.num_primitives == 9
    # two children of the split primitive exist with scale / 1.6
    # (check that the maximum scale dropped)
    assert model.scale().max().item() < before


def test_optimizer_state_is_preserved(tiny_cfg):
    cfg = tiny_cfg.with_method("3dgs")
    model = _toy("3dgs", cfg)
    model.setup_optimizer(cfg)
    for p in model.p.values():
        p.grad = torch.ones_like(p)
    model.optimizer.step()
    model.optimizer.zero_grad(set_to_none=True)
    model.rebuild_params(keep_mask=torch.ones(model.num_primitives, dtype=torch.bool))
    for group in model.optimizer.param_groups:
        p = group["params"][0]
        assert p is model.p[group["name"]]
        assert model.optimizer.state[p]["exp_avg"].shape[0] == model.num_primitives


def test_triangle_midpoint_split_conserves_area(tiny_cfg):
    cfg = tiny_cfg.with_method("triangle")
    model = _toy("triangle", cfg, n=1)
    model.setup_optimizer(cfg)
    with torch.no_grad():
        model.p["vertices"] = torch.nn.Parameter(
            torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        )
    model._stats = {
        "importance": torch.tensor([1.0]),
        "image_size": torch.tensor([100.0]),  # > split_size -> midpoint subdivision
        "coverage": torch.tensor([3.0]),
    }
    parent_area = model.triangle_area().sum().item()
    append = model._build_append(torch.tensor([0]))
    assert append["vertices"].shape == (4, 3, 3)
    children = append["vertices"]
    cross = torch.linalg.cross(children[:, 1] - children[:, 0], children[:, 2] - children[:, 0], dim=1)
    child_area = 0.5 * cross.norm(dim=1).sum().item()
    assert abs(child_area - parent_area) < 1e-6


def test_triangle_clone_for_small(tiny_cfg):
    cfg = tiny_cfg.with_method("triangle")
    model = _toy("triangle", cfg, n=1)
    model.setup_optimizer(cfg)
    model._stats = {
        "importance": torch.tensor([1.0]),
        "image_size": torch.tensor([2.0]),  # small -> clone (2 children)
        "coverage": torch.tensor([3.0]),
    }
    append = model._build_append(torch.tensor([0]))
    assert append["vertices"].shape[0] == 2
    # children opacity combines back to the parent's opacity
    op = model.opacity
    o_child = torch.sigmoid(append["opacity"][0])
    combined = 1.0 - (1.0 - o_child) * (1.0 - o_child)
    assert abs(combined.item() - op.item()) < 1e-4


def test_triangle_respects_max_shapes(tiny_cfg):
    cfg = tiny_cfg.with_method("triangle")
    cfg.tri_max_shapes = 10
    cfg.tri_growth = 1.3
    cfg.tri_densify_from_iter = 0
    cfg.tri_densify_until_iter = 100
    cfg.tri_densification_interval = 1
    cfg.tri_importance_threshold = -1.0  # nothing is dead
    model = _toy("triangle", cfg, n=8)
    model.setup_optimizer(cfg)
    model._stats = {
        "importance": torch.full((8,), 1.0),
        "image_size": torch.full((8,), 2.0),
        "coverage": torch.full((8,), 3.0),
    }
    model.densify(0)
    assert model.num_primitives <= 10


def test_empty_stats_do_not_crash(tiny_cfg):
    for method in ("3dgs", "2dgs", "triangle"):
        cfg = tiny_cfg.with_method(method)
        model = _toy(method, cfg, n=4)
        model.setup_optimizer(cfg)
        model.densify(0)  # no stats collected yet
