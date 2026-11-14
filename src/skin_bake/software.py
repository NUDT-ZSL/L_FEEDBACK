"""Dependency-free software preview renderer used for deterministic checks."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import output_io
from .math3d import look_at, normalize, perspective
from .runtime import BakedAsset


@dataclass(frozen=True)
class SoftwareRenderResult:
    image: np.ndarray
    covered_pixels: int
    sample_index: int
    frame_time: float


def camera_matrices(asset: BakedAsset, aspect: float) -> tuple[np.ndarray, np.ndarray]:
    points = asset.scene.positions
    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    extent = float(np.max(np.ptp(points, axis=0)))
    eye = center + np.array([0.0, 0.12 * extent, 2.4 * extent], dtype=np.float64)
    target = center.copy()
    view = look_at(eye, target, np.array([0.0, 1.0, 0.0]))
    proj = perspective(np.deg2rad(42.0), aspect, 0.01, max(100.0, extent * 10.0))
    return view, proj


def render_software_frame(
    asset: BakedAsset,
    frame: int,
    width: int = 800,
    height: int = 600,
) -> SoftwareRenderResult:
    width = max(16, int(width))
    height = max(16, int(height))
    time = asset.frame_time(frame)
    skin, _displacement, positions = asset.displaced_frame(time)
    sample_index = asset.curve_sample_index(time)
    view, base_proj = camera_matrices(asset, width / height)

    homogeneous = np.concatenate(
        [positions, np.ones((positions.shape[0], 1), dtype=np.float64)], axis=1
    )
    projected = homogeneous @ view.T @ base_proj.T
    w = projected[:, 3]
    valid = np.abs(w) > 1e-12
    ndc = np.zeros((positions.shape[0], 3), dtype=np.float64)
    ndc[valid] = projected[valid, :3] / w[valid, None]
    screen = np.empty((positions.shape[0], 2), dtype=np.float64)
    screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * (width - 1)
    screen[:, 1] = (0.5 - ndc[:, 1] * 0.5) * (height - 1)
    depth = ndc[:, 2]

    light = normalize(np.array([0.25, 0.4, 0.9], dtype=np.float64))
    normals = skin.normals
    intensity = 0.2 + 0.8 * np.abs(normals @ light)
    base = np.array([0.12, 0.48, 0.78], dtype=np.float64)
    vertex_colors = np.clip(
        (0.25 + 0.75 * intensity)[:, None] * base[None, :] + 0.035,
        0.0,
        1.0,
    )

    yy = np.linspace(0.055, 0.015, height, dtype=np.float64)[:, None, None]
    xx = np.linspace(0.0, 0.025, width, dtype=np.float64)[None, :, None]
    image = np.broadcast_to(yy + xx, (height, width, 3)).copy()
    zbuffer = np.full((height, width), np.inf, dtype=np.float64)

    for tri in asset.scene.triangles:
        p0, p1, p2 = screen[tri[0]], screen[tri[1]], screen[tri[2]]
        z0, z1, z2 = depth[tri[0]], depth[tri[1]], depth[tri[2]]
        if not np.all(np.isfinite([z0, z1, z2])):
            continue
        min_x = max(0, int(np.floor(np.min([p0[0], p1[0], p2[0]]))))
        max_x = min(width - 1, int(np.ceil(np.max([p0[0], p1[0], p2[0]]))))
        min_y = max(0, int(np.floor(np.min([p0[1], p1[1], p2[1]]))))
        max_y = min(height - 1, int(np.ceil(np.max([p0[1], p1[1], p2[1]]))))
        if min_x > max_x or min_y > max_y:
            continue
        area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if abs(float(area)) < 1e-10:
            continue
        px = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
        py = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
        gx, gy = np.meshgrid(px, py)
        inv_area = 1.0 / area
        w0 = ((p1[0] - gx) * (p2[1] - gy) - (p1[1] - gy) * (p2[0] - gx)) * inv_area
        w1 = ((p2[0] - gx) * (p0[1] - gy) - (p2[1] - gy) * (p0[0] - gx)) * inv_area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-7) & (w1 >= -1e-7) & (w2 >= -1e-7)
        if not np.any(inside):
            continue
        zi = w0 * z0 + w1 * z1 + w2 * z2
        local_depth = zbuffer[min_y : max_y + 1, min_x : max_x + 1]
        mask = inside & (zi < local_depth)
        if not np.any(mask):
            continue
        color = (
            w0[..., None] * vertex_colors[tri[0]]
            + w1[..., None] * vertex_colors[tri[1]]
            + w2[..., None] * vertex_colors[tri[2]]
        )
        local_image = image[min_y : max_y + 1, min_x : max_x + 1]
        local_image[mask] = color[mask]
        local_depth[mask] = zi[mask]

    foreground = np.any(np.abs(image - np.array([0.035, 0.035, 0.035])) > 0.08, axis=2)
    return SoftwareRenderResult(
        image=np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8),
        covered_pixels=int(np.count_nonzero(foreground)),
        sample_index=sample_index,
        frame_time=time,
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def write_png(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG image must have shape (height, width, 3)")
    height, width, _ = image.shape
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=9))
        + _png_chunk(b"IEND", b"")
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
    except OSError as exc:
        raise output_io(f"cannot write preview '{path}': {exc}") from exc

