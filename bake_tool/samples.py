from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .mesh import Mesh, normalized_rows


@dataclass
class SampleBundle:
    low: Mesh
    high: Mesh | None
    metadata: dict[str, object]


def _grid_indices(nu: int, nv: int) -> np.ndarray:
    triangles: list[tuple[int, int, int]] = []
    for j in range(nv - 1):
        for i in range(nu - 1):
            a = j * nu + i
            b = a + 1
            d = (j + 1) * nu + i
            c = d + 1
            triangles.append((a, b, c))
            triangles.append((a, c, d))
    return np.asarray(triangles, dtype=np.int32)


def _surface(patch: int, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    x0, x1 = (-0.92, -0.16) if patch == 0 else (0.16, 0.92)
    y0, y1 = -0.72, 0.72
    x = x0 + (x1 - x0) * s
    y = y0 + (y1 - y0) * t
    phase = 0.35 if patch == 0 else 1.10
    z = (
        0.075 * np.sin(np.pi * s) * np.sin(np.pi * t)
        + 0.020 * s * t
        + 0.008 * np.sin(2.0 * np.pi * s + phase) * np.sin(np.pi * t)
    )
    return np.stack((x, y, z), axis=-1)


def _base_normals(patch: int, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    x0, x1 = (-0.92, -0.16) if patch == 0 else (0.16, 0.92)
    y0, y1 = -0.72, 0.72
    dx = x1 - x0
    dy = y1 - y0
    phase = 0.35 if patch == 0 else 1.10
    dz_ds = (
        0.075 * np.pi * np.cos(np.pi * s) * np.sin(np.pi * t)
        + 0.020 * t
        + 0.016 * np.pi * np.cos(2.0 * np.pi * s + phase) * np.sin(np.pi * t)
    )
    dz_dt = (
        0.075 * np.pi * np.sin(np.pi * s) * np.cos(np.pi * t)
        + 0.020 * s
        + 0.008 * np.sin(2.0 * np.pi * s + phase) * np.pi * np.cos(np.pi * t)
    )
    normal = np.stack(
        (-dz_ds * dy, -dz_dt * dx, np.full_like(s, dx * dy)),
        axis=-1,
    )
    return normalized_rows(normal)


def _high_point(patch: int, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    base = _surface(patch, s, t)
    normal = _base_normals(patch, s, t)
    phase = 0.0 if patch == 0 else 0.8
    displacement = (
        0.026
        + 0.010 * np.sin(2.0 * np.pi * (3.0 * s + 0.15 * t) + phase)
        * np.sin(2.0 * np.pi * (2.0 * t - 0.10 * s))
        + 0.004 * np.cos(2.0 * np.pi * 7.0 * s) * np.sin(2.0 * np.pi * 5.0 * t)
    )
    return base + normal * displacement[..., None]


def _high_normals(patch: int, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    h = 2.0e-4
    s_plus = np.clip(s + h, 0.0, 1.0)
    s_minus = np.clip(s - h, 0.0, 1.0)
    t_plus = np.clip(t + h, 0.0, 1.0)
    t_minus = np.clip(t - h, 0.0, 1.0)
    dp_ds = (_high_point(patch, s_plus, t) - _high_point(patch, s_minus, t)) / np.maximum(
        (s_plus - s_minus)[..., None], 1.0e-12
    )
    dp_dt = (_high_point(patch, s, t_plus) - _high_point(patch, s, t_minus)) / np.maximum(
        (t_plus - t_minus)[..., None], 1.0e-12
    )
    normals = normalized_rows(np.cross(dp_ds, dp_dt))
    base = _base_normals(patch, s, t)
    flip = np.einsum("...i,...i->...", normals, base) < 0.0
    normals[flip] *= -1.0
    return normals


def _material(
    patch: int,
    s: np.ndarray,
    t: np.ndarray,
    high_detail: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if patch == 0:
        base_color = np.asarray([0.34, 0.46, 0.58])
        metallic_base = 0.16
    else:
        base_color = np.asarray([0.66, 0.42, 0.24])
        metallic_base = 0.78
    detail = 1.0 if high_detail else 0.35
    variation = np.stack(
        (
            0.075 * np.sin(2.0 * np.pi * (2.3 * s + 0.7 * t)),
            0.065 * np.cos(2.0 * np.pi * (1.6 * s - 2.1 * t)),
            0.055 * np.sin(2.0 * np.pi * (2.8 * s + 1.3 * t)),
        ),
        axis=-1,
    )
    albedo = np.clip(base_color + detail * variation, 0.02, 0.98)
    roughness = np.clip(
        0.38
        + detail * 0.18 * (0.5 + 0.5 * np.sin(2.0 * np.pi * (4.0 * s + 1.5 * t))),
        0.08,
        0.92,
    )
    metallic = np.clip(
        metallic_base
        + detail * 0.09 * np.sin(2.0 * np.pi * (1.2 * s - 2.6 * t)),
        0.0,
        1.0,
    )
    return albedo, roughness, metallic


def _make_patch_mesh(
    patch: int,
    nu: int,
    nv: int,
    uv_rect: tuple[float, float, float, float],
    high: bool,
    mirror_u: bool,
) -> Mesh:
    ss = np.linspace(0.0, 1.0, nu)
    tt = np.linspace(0.0, 1.0, nv)
    s, t = np.meshgrid(ss, tt, indexing="xy")
    s = s.reshape(-1)
    t = t.reshape(-1)

    if high:
        positions = _high_point(patch, s, t)
        normals = _high_normals(patch, s, t)
    else:
        positions = _surface(patch, s, t)
        normals = _base_normals(patch, s, t)

    u0, v0, u1, v1 = uv_rect
    u = u0 + (u1 - u0) * s
    if mirror_u:
        u = u0 + (u1 - u0) * (1.0 - s)
    v = v0 + (v1 - v0) * t
    uvs = np.stack((u, v), axis=-1)

    albedo, roughness, metallic = _material(patch, s, t, high)
    return Mesh(
        positions=positions.astype(np.float64),
        normals=normals.astype(np.float64),
        uvs=uvs.astype(np.float64),
        triangles=_grid_indices(nu, nv),
        albedo=albedo.astype(np.float64),
        roughness=roughness.astype(np.float64),
        metallic=metallic.astype(np.float64),
        name=f"patch_{patch}_{'high' if high else 'low'}",
    )


def _combine(meshes: list[Mesh], name: str) -> Mesh:
    offsets = np.cumsum([0] + [mesh.positions.shape[0] for mesh in meshes[:-1]])
    triangles = np.concatenate(
        [mesh.triangles + int(offset) for mesh, offset in zip(meshes, offsets)],
        axis=0,
    )
    return Mesh(
        positions=np.concatenate([mesh.positions for mesh in meshes], axis=0),
        normals=np.concatenate([mesh.normals for mesh in meshes], axis=0),
        uvs=np.concatenate([mesh.uvs for mesh in meshes], axis=0),
        triangles=triangles.astype(np.int32),
        albedo=np.concatenate([mesh.albedo for mesh in meshes], axis=0),
        roughness=np.concatenate([mesh.roughness for mesh in meshes], axis=0),
        metallic=np.concatenate([mesh.metallic for mesh in meshes], axis=0),
        name=name,
    )


def generate_sample(scenario: str = "default") -> SampleBundle:
    low_parts = [
        _make_patch_mesh(0, 9, 13, (0.055, 0.065, 0.455, 0.935), False, False),
        _make_patch_mesh(1, 9, 13, (0.545, 0.065, 0.945, 0.935), False, True),
    ]
    high_parts = [
        _make_patch_mesh(0, 29, 37, (0.055, 0.065, 0.455, 0.935), True, False),
        _make_patch_mesh(1, 29, 37, (0.545, 0.065, 0.945, 0.935), True, True),
    ]
    low = _combine(low_parts, "deterministic_two_island_low")
    high = _combine(high_parts, "deterministic_two_island_high")

    if scenario == "uv_out_of_range":
        low.uvs[0] = (1.08, low.uvs[0, 1])
    elif scenario == "missing_high":
        high = None
    elif scenario == "miss_rate":
        high.positions = high.positions + np.asarray([4.0, 0.0, 0.0])
    elif scenario != "default":
        raise ValueError(f"Unknown sample scenario: {scenario}")

    return SampleBundle(
        low=low,
        high=high,
        metadata={
            "sample_id": "procedural_two_island_v1",
            "scenario": scenario,
            "low_triangle_count": int(low.triangles.shape[0]),
            "high_triangle_count": int(high.triangles.shape[0]) if high is not None else 0,
            "mirrored_uv_island": True,
        },
    )

