"""Deterministic procedural capsule mesh and skin weight generation."""

from __future__ import annotations

import numpy as np

# Bone layout: root bone at the origin, child bone offset along +Y.
BONE0_BASE = np.array([0.0, -0.6, 0.0])
BONE0_TIP = np.array([0.0, 0.0, 0.0])
BONE1_BASE = np.array([0.0, 0.0, 0.0])
BONE1_TIP = np.array([0.0, 0.6, 0.0])
CHILD_OFFSET = np.array([0.0, 0.6, 0.0])  # child bind-pose offset along Y


def make_capsule(
    radius: float = 0.4,
    half_height: float = 0.5,
    segments: int = 16,
    hemi_rings: int = 4,
    cyl_rings: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a capsule aligned with the Y axis.

    Returns (vertices, triangles): vertices (N,3) float64, triangles (M,3)
    int32 with outward winding.
    """
    profile: list[tuple[float, float]] = []  # (y, ring_radius)
    profile.append((-(half_height + radius), 0.0))  # bottom pole
    for i in range(1, hemi_rings + 1):
        phi = -np.pi / 2 + (np.pi / 2) * i / (hemi_rings + 1)
        profile.append((-half_height + radius * np.sin(phi),
                        radius * np.cos(phi)))
    for j in range(cyl_rings + 1):
        profile.append((-half_height + 2 * half_height * j / cyl_rings, radius))
    for i in range(1, hemi_rings + 1):
        phi = (np.pi / 2) * i / (hemi_rings + 1)
        profile.append((half_height + radius * np.sin(phi),
                        radius * np.cos(phi)))
    profile.append((half_height + radius, 0.0))  # top pole

    verts: list[tuple[float, float, float]] = []
    ring_start: list[int] = []
    ring_size: list[int] = []
    for y, rad in profile:
        ring_start.append(len(verts))
        if rad <= 0.0:
            ring_size.append(1)
            verts.append((0.0, y, 0.0))
        else:
            ring_size.append(segments)
            for s in range(segments):
                theta = 2.0 * np.pi * s / segments
                verts.append((rad * np.cos(theta), y, rad * np.sin(theta)))

    tris: list[tuple[int, int, int]] = []
    n_rings = len(profile)
    for r in range(n_rings - 1):
        a0, na = ring_start[r], ring_size[r]
        b0, nb = ring_start[r + 1], ring_size[r + 1]
        if na == 1 and nb == 1:
            continue
        if na == 1:
            # Bottom fan: (pole, j, j+1) gives downward/outward normal.
            for j in range(nb):
                tris.append((a0, b0 + j, b0 + (j + 1) % nb))
        elif nb == 1:
            # Top fan: (pole, j+1, j) gives upward/outward normal.
            for j in range(na):
                tris.append((b0, a0 + (j + 1) % na, a0 + j))
        else:
            for j in range(na):
                a = a0 + j
                b = a0 + (j + 1) % na
                c = b0 + (j + 1) % nb
                d = b0 + j
                tris.append((a, d, b))
                tris.append((b, d, c))

    vertices = np.asarray(verts, dtype=np.float64)
    triangles = np.asarray(tris, dtype=np.int32)
    return vertices, triangles


def _point_segment_distance(points: np.ndarray,
                            a: np.ndarray,
                            b: np.ndarray) -> np.ndarray:
    """Distance from each point to segment [a, b]."""
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    closest = a + t[:, None] * ab
    return np.linalg.norm(points - closest, axis=1)


def compute_skin_weights(vertices: np.ndarray,
                         eps: float = 1e-3) -> np.ndarray:
    """Inverse-distance-to-bone-axis weights, normalized to sum to 1.

    Returns (N, 2) float64: column 0 -> root bone, column 1 -> child bone.
    """
    d0 = _point_segment_distance(vertices, BONE0_BASE, BONE0_TIP)
    d1 = _point_segment_distance(vertices, BONE1_BASE, BONE1_TIP)
    w0 = 1.0 / (d0 + eps)
    w1 = 1.0 / (d1 + eps)
    total = w0 + w1
    weights = np.stack([w0 / total, w1 / total], axis=1)
    return weights
