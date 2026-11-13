"""Arch frame tessellation with arc-length-parameterized stripe attribute.

Each gate segment is a curved frame band: an elliptical half-arch
(inner/outer edge) extruded along local z. Vertices carry a normalized
arc-length coordinate `s` in [0, 1] measured along the arch mid-line, so
stripe phase/duty stay stable for any arch height/width. Inner and outer
edges share the same angular parameter, so stripes align radially and no
tearing appears at silhouette edges or thin triangles.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .scene import Segment

ARC_STEPS = 28  # angular samples along the half-arch


def _arc_length_params(steps: int, r_mid: float, h_mid: float) -> np.ndarray:
    """Normalized cumulative arc length s(t) along the arch mid-line."""
    ts = np.linspace(0.0, math.pi, steps + 1)
    xs = r_mid * np.cos(ts)
    ys = h_mid * np.sin(ts)
    seg = np.hypot(np.diff(xs), np.diff(ys))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    return cum / total if total > 0 else np.zeros_like(cum)


@lru_cache(maxsize=256)
def _tessellate_cached(arch_height, arch_width, thickness, depth, steps):
    ri = arch_width * 0.5
    ro = ri + thickness
    hi = arch_height
    ho = arch_height + thickness
    r_mid = (ri + ro) * 0.5
    h_mid = (hi + ho) * 0.5

    s_param = _arc_length_params(steps, r_mid, h_mid)
    ts = np.linspace(0.0, math.pi, steps + 1)
    cos_t, sin_t = np.cos(ts), np.sin(ts)
    zf, zb = depth * 0.5, -depth * 0.5

    # Per station: inner-front, outer-front, inner-back, outer-back.
    verts = np.empty(((steps + 1) * 4, 3), dtype=np.float64)
    verts[0::4] = np.stack([ri * cos_t, hi * sin_t, np.full_like(ts, zf)], axis=1)
    verts[1::4] = np.stack([ro * cos_t, ho * sin_t, np.full_like(ts, zf)], axis=1)
    verts[2::4] = np.stack([ri * cos_t, hi * sin_t, np.full_like(ts, zb)], axis=1)
    verts[3::4] = np.stack([ro * cos_t, ho * sin_t, np.full_like(ts, zb)], axis=1)
    s_attr = np.repeat(s_param, 4)

    tris = []
    IF, OF, IB, OB = 0, 1, 2, 3
    for i in range(steps):
        a, b = 4 * i, 4 * (i + 1)
        # front face (z = +depth/2)
        tris += [(a + IF, b + IF, b + OF), (a + IF, b + OF, a + OF)]
        # back face (reversed winding)
        tris += [(a + IB, b + OB, b + IB), (a + IB, a + OB, b + OB)]
        # inner rim
        tris += [(a + IF, a + IB, b + IB), (a + IF, b + IB, b + IF)]
        # outer rim
        tris += [(a + OF, b + OF, b + OB), (a + OF, b + OB, a + OB)]
    # end caps
    a, b = 0, 4 * steps
    tris += [(a + IF, a + OF, a + OB), (a + IF, a + OB, a + IB)]
    tris += [(b + IF, b + OB, b + OF), (b + IF, b + IB, b + OB)]

    return verts, s_attr, np.asarray(tris, dtype=np.int64)


def tessellate(segment: Segment):
    """Return (local verts (N,3), s (N,), triangles (M,3)) for a segment."""
    return _tessellate_cached(
        segment.arch_height,
        segment.arch_width,
        segment.thickness,
        segment.depth,
        ARC_STEPS,
    )


def world_transform(segment: Segment, verts_local: np.ndarray) -> np.ndarray:
    yaw = math.radians(segment.yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    return verts_local @ rot.T + np.asarray(segment.translation, dtype=np.float64)


def segment_aabb(segment: Segment):
    verts, _, _ = tessellate(segment)
    world = world_transform(segment, verts)
    return world.min(axis=0), world.max(axis=0)
