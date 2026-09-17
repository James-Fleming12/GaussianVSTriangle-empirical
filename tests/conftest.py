"""Shared pytest fixtures."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# tiny tensors + many ops: the thread pool costs more than it saves on CPU
torch.set_num_threads(1)

from gvst.config import TrainConfig, scaled_train_config
from gvst.scene.cameras import Camera, field_of_view_camera


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def tiny_scene(device):
    from gvst.scene.synthetic import make_scene

    return make_scene("corner").to(device)


@pytest.fixture
def tiny_cameras(tiny_scene, device):
    return tiny_scene.cameras(n=4, width=16, height=16, fov_deg=45.0, radius=3.0)


@pytest.fixture
def tiny_cfg():
    return scaled_train_config(TrainConfig(iterations=20, sh_degree=1, seed=0))


@pytest.fixture
def axis_camera():
    """Camera at (0,0,3) looking at the origin with +y up."""
    eye = torch.tensor([0.0, 0.0, 3.0])
    target = torch.tensor([0.0, 0.0, 0.0])
    return field_of_view_camera(eye, target, 45.0, 16, 16)
