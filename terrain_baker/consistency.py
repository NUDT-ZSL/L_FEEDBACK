"""Geometric consistency between the normal map and every LOD mesh.

Mesh vertex normals are recomputed from the same heightfield at that
level's sample spacing, while the normal map is sampled at each vertex's
UV. Both live in tangent space (X=U, Y=V downward, Z=viewer), so their
angular difference is measured directly. Sampling uses bilinear filtering
with edge clamping, matching a standard GPU texture sampler.
"""

from __future__ import annotations

import numpy as np

from .lod import tangent_normals_from_height
from .normal import decode_rgb8


def bilinear_sample_clamped(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample an HxWxC image at UVs in [0, 1], edges clamped."""
    height, width = image.shape[0], image.shape[1]
    fx = np.clip(u, 0.0, 1.0) * np.float32(width - 1)
    fy = np.clip(v, 0.0, 1.0) * np.float32(height - 1)
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = (fx - x0.astype(np.float32))[..., None]
    ty = (fy - y0.astype(np.float32))[..., None]
    top = image[y0, x0].astype(np.float32) * (1.0 - tx) + image[y0, x1].astype(
        np.float32
    ) * tx
    bottom = image[y1, x0].astype(np.float32) * (1.0 - tx) + image[
        y1, x1
    ].astype(np.float32) * tx
    return top * (1.0 - ty) + bottom * ty


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between two arrays of unit vectors."""
    dot = np.sum(a * b, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def check_level(
    height: np.ndarray,
    spacing_world: float,
    normal_image: np.ndarray,
    sample_count: int,
    rng: np.random.Generator,
) -> dict:
    """Return max/mean angular error for one LOD using deterministic points."""
    coarse = height.shape[0]
    mesh_normals = tangent_normals_from_height(height, spacing_world)
    points = rng.integers(0, coarse, size=(sample_count, 2))
    rows = points[:, 0]
    cols = points[:, 1]
    u = cols.astype(np.float32) / np.float32(coarse - 1)
    v = rows.astype(np.float32) / np.float32(coarse - 1)
    sampled = bilinear_sample_clamped(decode_rgb8(normal_image), u, v)
    sampled = sampled / np.maximum(
        np.linalg.norm(sampled, axis=1, keepdims=True), np.float32(1e-12)
    )
    reference = mesh_normals[rows, cols]
    angles = angle_between(reference, sampled)
    return {
        "sample_count": int(sample_count),
        "max_angle_deg": float(np.max(angles)),
        "mean_angle_deg": float(np.mean(angles)),
        "p95_angle_deg": float(np.percentile(angles, 95.0)),
        "threshold_deg": 5.0,
        "passed": bool(np.max(angles) <= 5.0),
    }
