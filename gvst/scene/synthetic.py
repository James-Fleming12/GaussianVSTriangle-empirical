"""Analytic synthetic scenes used as exact ground truth.

Every scene is a small collection of analytic surfaces (spheres, quads) with
procedural textures and a simple Lambertian + specular shading model.  Images
are produced by exact ray casting, so:

* ground truth is noise-free and resolution-independent,
* depth and normals are available in closed form (useful in tests),
* a *view-dependent* appearance (specular) can be enabled by choice.

The scenes are intentionally split by which geometry they stress:

``plane``   a single textured plane seen from a ring of shallow elevations
            (surface vs. volume primitives; oblique texture detail).
``sphere``  a curved, mostly convex object with a specular highlight
            (view-dependent appearance + smooth curvature).
``corner``  three mutually orthogonal textured planes forming a room corner
            (sharp edges, depth discontinuities, occlusion boundaries).
``sheets``  several thin, disconnected double-sided rectangles
            (thin structure; a classic failure case for volumetric splats).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor

from .cameras import Camera, orbit_cameras

# ---------------------------------------------------------------------------
# hit records and shapes
# ---------------------------------------------------------------------------


@dataclass
class Hit:
    t: Tensor  # [P] ray parameter (only meaningful where valid)
    valid: Tensor  # [P] bool
    albedo: Tensor  # [P,3]
    normal: Tensor  # [P,3]

    @classmethod
    def empty(cls, n: int, device, dtype) -> "Hit":
        z = torch.zeros(n, device=device, dtype=dtype)
        return cls(z, torch.zeros(n, dtype=torch.bool, device=device), z.new_zeros(n, 3), z.new_zeros(n, 3))


class Shape:
    def intersect(self, origins: Tensor, dirs: Tensor) -> Hit:  # pragma: no cover - interface
        raise NotImplementedError


def _uv_grid(n: int, device, dtype) -> Tensor:
    return torch.rand(n, 2, device=device, dtype=dtype)


def _downsample(out: dict, ss: int) -> dict:
    """Box-filter supersampled buffers back to the output resolution."""
    import torch.nn.functional as F

    img = F.avg_pool2d(out["image"].unsqueeze(0), ss)[0]
    mask_hi = out["mask"].float()
    cover = F.avg_pool2d(mask_hi.unsqueeze(0).unsqueeze(0), ss)[0, 0]
    depth = torch.where(
        cover > 0,
        F.avg_pool2d((out["depth"] * mask_hi).unsqueeze(0).unsqueeze(0), ss)[0, 0] / cover.clamp_min(1e-6),
        torch.zeros_like(cover),
    )
    normal = F.avg_pool2d((out["normal"] * mask_hi).unsqueeze(0), ss)[0] / cover.clamp_min(1e-6).unsqueeze(0)
    return {"image": img, "depth": depth, "normal": normal, "mask": cover > 0}


def checker(uv: Tensor, scale: float = 8.0, c1=(0.9, 0.9, 0.9), c2=(0.15, 0.15, 0.15)) -> Tensor:
    k = torch.floor(uv * scale).sum(dim=-1, keepdim=True)
    f = (k % 2).clamp(0, 1)
    c1 = torch.tensor(c1, device=uv.device, dtype=uv.dtype)
    c2 = torch.tensor(c2, device=uv.device, dtype=uv.dtype)
    return c1 * (1 - f) + c2 * f


def high_freq(uv: Tensor, freq: float = 24.0, amp: float = 0.25) -> Tensor:
    """Deterministic sinusoidal texture in [0, 1], band-limited below Nyquist
    at the benchmark resolution (``freq`` cycles per unit UV interval)."""
    s = 0.5 + 0.5 * torch.sin(2 * torch.pi * freq * uv[..., 0:1]) * torch.sin(2 * torch.pi * freq * uv[..., 1:2])
    s = s * (1 - amp) + amp * (0.5 + 0.5 * torch.sin(2 * torch.pi * (freq * 0.37) * uv[..., 0:1] + 1.3))
    return s.clamp(0, 1)


class Quad(Shape):
    """Parallelogram ``p0 + a*e1 + b*e2`` for ``a,b in [0,1]`` with a texture."""

    def __init__(self, p0, e1, e2, texture: Callable[[Tensor], Tensor], double_sided: bool = False):
        self.p0 = torch.as_tensor(p0, dtype=torch.float32)
        self.e1 = torch.as_tensor(e1, dtype=torch.float32)
        self.e2 = torch.as_tensor(e2, dtype=torch.float32)
        self.texture = texture
        self.double_sided = double_sided
        self._n = torch.linalg.cross(self.e1, self.e2)
        self._n = self._n / self._n.norm().clamp_min(1e-8)

    def to(self, device) -> "Quad":
        self.p0 = self.p0.to(device)
        self.e1 = self.e1.to(device)
        self.e2 = self.e2.to(device)
        self._n = self._n.to(device)
        return self

    def sample(self, n: int, generator: torch.Generator | None = None, device=None) -> dict:
        """Uniform surface samples ``(points, normals, albedo)`` on the quad."""
        device = self.p0.device if device is None else device
        uv = torch.rand(n, 2, generator=generator, device="cpu").to(device)
        points = self.p0 + uv[:, 0:1] * self.e1 + uv[:, 1:2] * self.e2
        normals = self._n.unsqueeze(0).expand(n, 3).clone()
        return {"points": points, "normals": normals, "albedo": self.texture(uv)}

    def intersect(self, origins: Tensor, dirs: Tensor) -> Hit:
        n = self._n
        denom = dirs @ n
        t = ((self.p0 - origins) @ n) / denom.clamp(min=-1e6, max=1e6)
        t = torch.where(denom.abs() < 1e-8, torch.full_like(t, torch.inf), t)
        q = origins + t[:, None] * dirs - self.p0
        # solve [e1 e2] [a b]^T = q in the plane
        a11 = (self.e1 @ self.e1).clamp_min(1e-8)
        a12 = self.e1 @ self.e2
        a22 = (self.e2 @ self.e2).clamp_min(1e-8)
        q1 = q @ self.e1
        q2 = q @ self.e2
        det = (a11 * a22 - a12 * a12).clamp_min(1e-8)
        a = (q1 * a22 - q2 * a12) / det
        b = (q2 * a11 - q1 * a12) / det
        valid = (t > 1e-4) & (a >= 0) & (a <= 1) & (b >= 0) & (b <= 1)
        if not self.double_sided:
            valid = valid & (denom < 0)
        uv = torch.stack([a, b], dim=-1).clamp(0, 1)
        albedo = self.texture(uv)
        normal = n.unsqueeze(0).expand_as(albedo).clone()
        normal = torch.where((denom > 0)[:, None], -normal, normal)
        return Hit(t, valid, albedo, normal)


class Sphere(Shape):
    def __init__(self, center, radius: float, texture: Callable[[Tensor], Tensor]):
        self.center = torch.as_tensor(center, dtype=torch.float32)
        self.radius = float(radius)
        self.texture = texture

    def to(self, device) -> "Sphere":
        self.center = self.center.to(device)
        return self

    def sample(self, n: int, generator: torch.Generator | None = None, device=None) -> dict:
        """Uniform surface samples ``(points, normals, albedo)`` on the sphere."""
        device = self.center.device if device is None else device
        d = torch.randn(n, 3, generator=generator, device="cpu").to(device)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        points = self.center + self.radius * d
        u = torch.atan2(points[..., 2] - self.center[2], points[..., 0] - self.center[0]) / (2 * torch.pi) + 0.5
        v = torch.acos((points[..., 1] - self.center[1]).div(self.radius).clamp(-1, 1)) / torch.pi
        uv = torch.stack([u, v], dim=-1)
        return {"points": points, "normals": d, "albedo": self.texture(uv)}

    def intersect(self, origins: Tensor, dirs: Tensor) -> Hit:
        oc = origins - self.center  # broadcasts: [3] or [P,3]
        b = 2.0 * (oc * dirs).sum(-1)
        c = (oc * oc).sum(-1) - self.radius**2
        disc = b * b - 4.0 * c
        valid_disc = disc > 0
        sq = torch.sqrt(disc.clamp_min(0.0))
        t0 = (-b - sq) / 2.0
        t1 = (-b + sq) / 2.0
        t = torch.where(t0 > 1e-4, t0, t1)
        valid = valid_disc & (t > 1e-4)
        p = origins + t[:, None] * dirs
        n = (p - self.center) / self.radius
        u = torch.atan2(p[..., 2] - self.center[2], p[..., 0] - self.center[0]) / (2 * torch.pi) + 0.5
        v = torch.acos((p[..., 1] - self.center[1]).div(self.radius).clamp(-1, 1)) / torch.pi
        uv = torch.stack([u, v], dim=-1)
        albedo = self.texture(uv)
        return Hit(t, valid, albedo, n)


@dataclass
class Scene:
    name: str
    description: str
    shapes: list
    background: tuple = (0.05, 0.05, 0.07)
    light_dir: tuple = (-0.5, 0.7, 0.4)
    ambient: float = 0.25
    specular: float = 0.0  # 0 disables the specular lobe
    shininess: float = 32.0
    target: tuple = (0.0, 0.0, 0.0)
    camera_radius: float = 3.0
    elevations_deg: tuple = (20.0, 25.0, 30.0)
    supersample: int = 2  # anti-aliased ground truth (box filter over SSAA grid)

    def to(self, device) -> "Scene":
        self.shapes = [s.to(device) for s in self.shapes]
        return self

    # ---- analytic ray casting ----
    def intersect(self, origins: Tensor, dirs: Tensor) -> Hit:
        hits = [s.intersect(origins, dirs) for s in self.shapes]
        ts = torch.stack([h.t for h in hits], dim=0)  # [S,P]
        valid = torch.stack([h.valid for h in hits], dim=0)
        ts = torch.where(valid, ts, torch.full_like(ts, torch.inf))
        best = ts.argmin(dim=0)
        p = origins + ts.gather(0, best[None])[0][:, None] * dirs
        albedo = torch.stack([h.albedo for h in hits], dim=0).gather(
            0, best[None, :, None].expand(1, -1, 3)
        )[0]
        normal = torch.stack([h.normal for h in hits], dim=0).gather(
            0, best[None, :, None].expand(1, -1, 3)
        )[0]
        t = ts.gather(0, best[None])[0]
        valid_out = valid.gather(0, best[None])[0]
        return Hit(t, valid_out, albedo, normal)

    # ---- image formation ----
    def shade(self, points: Tensor, normal: Tensor, albedo: Tensor, view_dir: Tensor) -> Tensor:
        light = torch.tensor(self.light_dir, device=points.device, dtype=points.dtype)
        light = light / light.norm()
        diffuse = (normal * light).sum(-1, keepdim=True).clamp_min(0.0)
        color = albedo * (self.ambient + diffuse)
        if self.specular > 0:
            refl = 2 * (normal * light).sum(-1, keepdim=True) * normal - light
            v = (-view_dir).clamp(-1.0, 1.0)
            spec = (refl * v).sum(-1, keepdim=True).clamp_min(0.0) ** self.shininess
            color = color + self.specular * spec
        return color.clamp(0.0, 1.0)

    def render(self, camera: Camera, supersample: Optional[int] = None) -> dict:
        """Render ground truth: image [3,H,W], depth [H,W], normal [3,H,W], mask [H,W].

        Rendering uses ``supersample x supersample`` rays per output pixel
        (box-filtered), so the ground truth is band-limited at the output
        resolution and does not alias the fine procedural textures.
        """
        ss = self.supersample if supersample is None else supersample
        if ss > 1:
            return _downsample(self._render_single(camera.scaled(ss)), ss)
        return self._render_single(camera)

    def _render_single(self, camera: Camera) -> dict:
        uv = camera.pixel_grid()
        origins, dirs = camera.ray(uv)
        hit = self.intersect(origins, dirs)
        color = self.shade(origins + hit.t[:, None] * dirs, hit.normal, hit.albedo, dirs)
        bg = torch.tensor(self.background, device=uv.device, dtype=uv.dtype)
        color = torch.where(hit.valid[:, None], color, bg.expand_as(color))
        image = color.reshape(camera.height, camera.width, 3).permute(2, 0, 1)
        pc = camera.world_to_camera(origins + torch.nan_to_num(hit.t, posinf=0.0)[:, None] * dirs)
        depth = (-pc[..., 2]).clamp_min(0)
        depth = torch.where(hit.valid, depth, torch.zeros_like(depth)).reshape(camera.height, camera.width)
        normal = hit.normal.reshape(camera.height, camera.width, 3).permute(2, 0, 1)
        return {
            "image": image,
            "depth": depth,
            "normal": normal,
            "mask": hit.valid.reshape(camera.height, camera.width),
        }

    # ---- protocol helpers ----
    @property
    def _device(self):
        return self.shapes[0].center.device if isinstance(self.shapes[0], Sphere) else self.shapes[0].p0.device

    def cameras(self, n: int, width: int, height: int, fov_deg: float, radius: Optional[float] = None) -> list:
        target = torch.tensor(self.target, device=self._device)
        return orbit_cameras(
            n=n,
            radius=self.camera_radius if radius is None else radius,
            target=target,
            fov_deg=fov_deg,
            width=width,
            height=height,
            elevations_deg=self.elevations_deg,
        )

    def init_points(self, cameras: list, n_points: int, seed: int = 0) -> tuple[Tensor, Tensor]:
        """SfM-like initialization: back-project visible training-view pixels.

        Returns ``(points [M,3], colors [M,3])``.  Only the *training* cameras
        passed in are used, so the held-out view never influences the model.
        """
        gen = torch.Generator(device="cpu").manual_seed(seed)
        pts_all, cols_all = [], []
        per_view = max(64, n_points)
        for cam in cameras:
            uv = torch.rand(per_view, 2, generator=gen).to(cam.device)
            uv[:, 0] *= cam.width
            uv[:, 1] *= cam.height
            origins, dirs = cam.ray(uv)
            hit = self.intersect(origins, dirs)
            if not hit.valid.any():
                continue
            p = origins + hit.t[:, None] * dirs
            col = self.shade(p, hit.normal, hit.albedo, dirs)
            pts_all.append(p[hit.valid])
            cols_all.append(col[hit.valid])
        points = torch.cat(pts_all, 0)
        colors = torch.cat(cols_all, 0)
        perm = torch.randperm(points.shape[0], generator=gen)[:n_points]
        return points[perm], colors[perm]

    def extent(self, cameras: list) -> float:
        c = torch.stack([cam.center for cam in cameras])
        center = c.mean(0)
        return float((c - center).norm(dim=-1).max()) * 1.1

    def sample_surface(self, n: int = 20_000, seed: int = 0) -> dict:
        """Analytic ground-truth surface samples for geometry metrics.

        Samples are spread evenly over the scene's shapes; ``points [n,3]``,
        ``normals [n,3]`` (unit, outward) and ``albedo [n,3]`` are returned.
        Because the shapes are analytic this is a noise-free reference surface
        rather than a depth-map back-projection.
        """
        gen = torch.Generator(device="cpu").manual_seed(seed)
        base, rem = divmod(n, max(1, len(self.shapes)))
        counts = [base + (1 if i < rem else 0) for i in range(len(self.shapes))]
        parts = [s.sample(c, generator=gen, device=self._device) for s, c in zip(self.shapes, counts) if c > 0]
        return {k: torch.cat([p[k] for p in parts], dim=0) for k in ("points", "normals", "albedo")}


# ---------------------------------------------------------------------------
# concrete scenes
# ---------------------------------------------------------------------------


def make_scene(name: str, seed: int = 0) -> Scene:
    if name == "plane":
        tex = lambda uv: checker(uv, scale=7.0, c1=(0.85, 0.55, 0.25), c2=(0.15, 0.2, 0.35)) * 0.6 + high_freq(uv) * 0.4
        q = Quad((-1.6, -1.0, -1.6), (0, 0, 3.2), (3.2, 0, 0), tex)
        return Scene(
            name="plane",
            description=(
                "A single textured ground plane.  Stresses grazing-angle texture "
                "reconstruction and shows whether a primitive can stay on a 2D "
                "surface instead of filling the space above it."
            ),
            shapes=[q],
            background=(0.4, 0.5, 0.7),
            target=(0.0, -1.0, 0.0),
            camera_radius=3.0,
            elevations_deg=(18.0, 30.0, 45.0),
        )
    if name == "sphere":
        tex = lambda uv: (
            0.55
            + 0.45 * torch.sin(2 * torch.pi * 6.0 * uv[..., 1:2])
            * torch.cos(2 * torch.pi * 4.0 * uv[..., 0:1])
        ).clamp(0, 1) * torch.tensor([0.9, 0.3, 0.2], device=uv.device, dtype=uv.dtype)
        s = Sphere((0.0, 0.0, 0.0), 1.0, tex)
        return Scene(
            name="sphere",
            description=(
                "A curved sphere with a smooth (low-frequency) albedo pattern and "
                "a specular lobe.  Tests curvature reconstruction and whether the "
                "primitive can represent view-dependent appearance."
            ),
            shapes=[s],
            background=(0.05, 0.05, 0.07),
            specular=0.6,
            shininess=48.0,
            camera_radius=3.2,
            elevations_deg=(15.0, 28.0, 40.0),
        )
    if name == "corner":
        floor = Quad((-1.2, -1.0, -1.2), (0, 0, 2.4), (2.4, 0, 0), lambda uv: checker(uv, 6.0, (0.8, 0.8, 0.8), (0.2, 0.2, 0.22)))
        wall_x = Quad((-1.2, -1.0, -1.2), (0, 2.0, 0), (0, 0, 2.4), lambda uv: checker(uv, 4.0, (0.75, 0.3, 0.2), (0.7, 0.65, 0.3)))
        wall_z = Quad((-1.2, -1.0, -1.2), (2.4, 0, 0), (0, 2.0, 0), lambda uv: (high_freq(uv) * 0.5 + 0.3).repeat(1, 3))
        return Scene(
            name="corner",
            description=(
                "A room corner made of three orthogonal textured planes.  Tests "
                "sharp occlusion/edge behaviour where surface primitives (2DGS, "
                "triangles) should beat volumetric blobs (3DGS)."
            ),
            shapes=[floor, wall_x, wall_z],
            background=(0.1, 0.12, 0.16),
            target=(0.0, 0.0, 0.0),
            camera_radius=3.0,
            elevations_deg=(15.0, 27.0, 40.0),
        )
    if name == "sheets":
        tex_a = lambda uv: checker(uv, 5.0, (0.9, 0.2, 0.2), (0.95, 0.9, 0.85))
        tex_b = lambda uv: high_freq(uv, freq=30.0).expand(-1, 3) * torch.tensor([0.3, 0.8, 0.4], device=uv.device, dtype=uv.dtype)
        sheet1 = Quad((-1.0, -0.7, 0.0), (2.0, 0, 0), (0, 1.4, 0), tex_a, double_sided=True)
        sheet2 = Quad((0.0, -0.7, -1.0), (0, 0, 2.0), (0, 1.4, 0), tex_b, double_sided=True)
        sheet3 = Quad((-0.4, 0.6, -0.4), (0.8, 0.0, 0.0), (0, -0.35, 0.35), lambda uv: checker(uv, 8.0, (0.3, 0.4, 0.9), (0.9, 0.9, 0.2)), double_sided=True)
        return Scene(
            name="sheets",
            description=(
                "Three thin, disconnected double-sided sheets.  Tests thin-structure "
                "reconstruction: volumetric primitives must either collapse onto the "
                "sheets (2DGS/triangles) or fill the space behind them (3DGS floaters)."
            ),
            shapes=[sheet1, sheet2, sheet3],
            background=(0.05, 0.05, 0.07),
            target=(0.0, 0.0, 0.0),
            camera_radius=3.2,
            elevations_deg=(15.0, 27.0, 40.0),
        )
    if name == "intersect":
        tex_xy = lambda uv: checker(uv, 6.0, (0.9, 0.25, 0.2), (0.15, 0.5, 0.9))
        tex_yz = lambda uv: checker(uv, 5.0, (0.95, 0.85, 0.2), (0.2, 0.7, 0.4))
        tex_xz = lambda uv: high_freq(uv, freq=18.0).expand(-1, 3)
        q_xy = Quad((-1.0, -1.0, 0.0), (2.0, 0, 0), (0, 2.0, 0), tex_xy, double_sided=True)
        q_yz = Quad((0.0, -1.0, -1.0), (0, 2.0, 0), (0, 0, 2.0), tex_yz, double_sided=True)
        q_xz = Quad((-1.0, 0.0, -1.0), (2.0, 0, 0), (0, 0, 2.0), tex_xz, double_sided=True)
        return Scene(
            name="intersect",
            description=(
                "Three mutually intersecting (not just adjacent) double-sided sheets "
                "passing through the origin.  Tests depth ordering under interpenetration: "
                "a single front-to-back sort by primitive centre is not a valid ordering "
                "where triangles cross, so transmittance and gradients are corrupted."
            ),
            shapes=[q_xy, q_yz, q_xz],
            background=(0.05, 0.05, 0.07),
            target=(0.0, 0.0, 0.0),
            camera_radius=3.2,
            elevations_deg=(15.0, 30.0, 45.0),
        )
    if name == "hf":
        tex = lambda uv: (
            checker(uv, 22.0, (0.95, 0.95, 0.95), (0.05, 0.05, 0.05)) * 0.6
            + high_freq(uv, freq=40.0) * 0.4
        )
        q = Quad((-1.6, -1.0, -1.6), (0, 0, 3.2), (3.2, 0, 0), tex)
        return Scene(
            name="hf",
            description=(
                "A high-frequency textured plane (checker + sinusoidal detail close to the "
                "training Nyquist).  Tests anti-aliasing and resolution scaling: primitives "
                "with hard windows can alias and must densify aggressively to resolve detail."
            ),
            shapes=[q],
            background=(0.4, 0.5, 0.7),
            target=(0.0, -1.0, 0.0),
            camera_radius=3.0,
            elevations_deg=(20.0, 32.0, 45.0),
        )
    if name == "solid":
        solid_tex = lambda uv: torch.tensor([0.55, 0.55, 0.58], device=uv.device, dtype=uv.dtype).expand(*uv.shape[:-1], 3)
        s = Sphere((0.0, 0.0, 0.0), 1.0, solid_tex)
        floor = Quad((-1.8, -1.0, -1.8), (0, 0, 3.6), (3.6, 0, 0), lambda uv: torch.full_like(uv[..., :1], 0.25).expand(-1, 3))
        return Scene(
            name="solid",
            description=(
                "Textureless geometry: a constant-albedo sphere over a constant-albedo "
                "floor.  Without texture the photometric loss constrains only shading and "
                "silhouettes, so it isolates how well a primitive recovers pure geometry "
                "instead of fitting colour."
            ),
            shapes=[s, floor],
            background=(0.3, 0.35, 0.45),
            target=(0.0, 0.0, 0.0),
            camera_radius=3.2,
            elevations_deg=(15.0, 28.0, 40.0),
        )
    raise KeyError(f"unknown scene {name!r}")


SCENES = ("plane", "sphere", "corner", "sheets", "intersect", "hf", "solid")
