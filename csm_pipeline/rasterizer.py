"""Pure-NumPy software triangle rasterizer.

The rasterizer uses 32-bit float depth buffers with NDC depth in ``[0, 1]``,
top-left fill convention and a strict less-than depth test.  No GPU or driver
calls are involved.
"""

from __future__ import annotations

import numpy as np

from .errors import ERR_INVALID_DEPTH, PipelineError
from .geometry import Mesh


def rasterize(
    mesh: Mesh,
    view_proj: np.ndarray,
    width: int,
    height: int,
    with_attributes: bool = True,
) -> dict:
    if width < 1 or height < 1:
        raise ValueError("raster target must be at least 1x1")
    depth = np.ones((height, width), dtype=np.float32)
    hit = np.zeros((height, width), dtype=bool)
    world_pos = np.zeros((height, width, 3), dtype=np.float32)
    normal_buf = np.zeros((height, width, 3), dtype=np.float32)

    positions = np.asarray(mesh.positions, dtype=np.float64)
    normals = np.asarray(mesh.normals, dtype=np.float64)
    triangle_count = positions.shape[0]

    for tri in range(triangle_count):
        tri_pos = positions[tri]
        clip = _clip_coords(view_proj, tri_pos)
        if np.any(clip[:, 3] <= 0.0):
            continue
        ndc = clip[:, :3] / clip[:, 3:4]
        screen_x = (ndc[:, 0] + 1.0) * 0.5 * width
        screen_y = (1.0 - ndc[:, 1]) * 0.5 * height

        min_x = max(int(np.floor(min(screen_x))), 0)
        max_x = min(int(np.ceil(max(screen_x))), width - 1)
        min_y = max(int(np.floor(min(screen_y))), 0)
        max_y = min(int(np.ceil(max(screen_y))), height - 1)
        if min_x > max_x or min_y > max_y:
            continue

        area, edge01, edge12, edge20 = _edge_setup(
            screen_x, screen_y
        )
        if area <= 0.0:
            continue

        gx, gy = np.meshgrid(
            np.arange(min_x, max_x + 1, dtype=np.float64),
            np.arange(min_y, max_y + 1, dtype=np.float64),
        )
        px = gx + 0.5
        py = gy + 0.5
        w0 = edge12[0] * (px - screen_x[1]) + edge12[1] * (py - screen_y[1])
        w1 = edge20[0] * (px - screen_x[2]) + edge20[1] * (py - screen_y[2])
        w2 = edge01[0] * (px - screen_x[0]) + edge01[1] * (py - screen_y[0])
        inside = (w0 >= 0.0) & (w1 >= 0.0) & (w2 >= 0.0)
        if not bool(inside.any()):
            continue
        b0 = w0 / area
        b1 = w1 / area
        b2 = w2 / area
        interpolated_depth = (
            b0 * ndc[0, 2] + b1 * ndc[1, 2] + b2 * ndc[2, 2]
        ).astype(np.float32)

        region_depth = depth[min_y : max_y + 1, min_x : max_x + 1]
        draw = inside & (interpolated_depth < region_depth)
        if not bool(draw.any()):
            continue
        region_depth[draw] = interpolated_depth[draw]
        depth[min_y : max_y + 1, min_x : max_x + 1] = region_depth

        hit_region = hit[min_y : max_y + 1, min_x : max_x + 1]
        hit_region[draw] = True

        if with_attributes:
            wp = (
                b0[..., None] * tri_pos[0]
                + b1[..., None] * tri_pos[1]
                + b2[..., None] * tri_pos[2]
            ).astype(np.float32)
            face_normal = np.cross(
                tri_pos[1] - tri_pos[0], tri_pos[2] - tri_pos[0]
            )
            face_norm = np.linalg.norm(face_normal)
            if face_norm > 0.0:
                face_normal = face_normal / face_norm
            else:
                face_normal = normals[tri, 0]
            wp_region = world_pos[min_y : max_y + 1, min_x : max_x + 1]
            wp_region[draw] = wp[draw]
            n_region = normal_buf[min_y : max_y + 1, min_x : max_x + 1]
            n_region[draw] = face_normal.astype(np.float32)

    if not np.isfinite(depth).all():
        raise PipelineError(
            ERR_INVALID_DEPTH, "rasterizer produced non-finite depth values"
        )
    result = {
        "depth": depth,
        "hit": hit,
        "width": int(width),
        "height": int(height),
    }
    if with_attributes:
        result["world_pos"] = world_pos
        result["normal"] = normal_buf
    return result


def _clip_coords(view_proj: np.ndarray, tri_pos: np.ndarray) -> np.ndarray:
    homog = np.concatenate(
        [tri_pos, np.ones((3, 1), dtype=np.float64)], axis=1
    )
    return homog @ np.asarray(view_proj, dtype=np.float64).T


def _edge_setup(screen_x: np.ndarray, screen_y: np.ndarray) -> tuple:
    edge01 = (
        screen_y[0] - screen_y[1],
        screen_x[1] - screen_x[0],
    )
    edge12 = (
        screen_y[1] - screen_y[2],
        screen_x[2] - screen_x[1],
    )
    edge20 = (
        screen_y[2] - screen_y[0],
        screen_x[0] - screen_x[2],
    )
    area = (
        edge12[0] * (screen_x[0] - screen_x[1])
        + edge12[1] * (screen_y[0] - screen_y[1])
    )
    return area, edge01, edge12, edge20


def validate_depth_map(depth_map: np.ndarray, hit: np.ndarray) -> None:
    arr = np.asarray(depth_map)
    if arr.ndim != 2 or arr.dtype != np.float32:
        raise PipelineError(
            ERR_INVALID_DEPTH, "depth map must be a float32 2-D array"
        )
    if not np.isfinite(arr).all():
        raise PipelineError(ERR_INVALID_DEPTH, "depth map contains non-finite values")
    if bool((arr[hit] < 0.0).any()) or bool((arr[hit] > 1.0).any()):
        raise PipelineError(
            ERR_INVALID_DEPTH, "depth map values outside the [0, 1] NDC range"
        )

