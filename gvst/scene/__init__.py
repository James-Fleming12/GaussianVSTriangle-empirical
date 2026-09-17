"""Scene package: cameras and synthetic ground truth."""

from .cameras import Camera, field_of_view_camera, look_at, orbit_cameras
from .synthetic import SCENES, Scene, make_scene

__all__ = ["Camera", "field_of_view_camera", "look_at", "orbit_cameras", "SCENES", "Scene", "make_scene"]
