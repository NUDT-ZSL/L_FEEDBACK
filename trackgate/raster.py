"""Deterministic CPU rasterizer: RGBA + float depth + segment-id buffers.

Visibility is decided purely by a per-pixel depth test (closest camera
space z wins); draw order never determines the visible pixel. Stripe mask
evaluation uses perspective-correct barycentric interpolation of the
arc-length attribute at pixel centers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .arch import tessellate, world_transform
from .camera import Camera
from .errors import BadImageSizeError
from .scene import Scene

BG_COLOR = (12, 12, 20, 255)
LIGHT_DIR = np.array([0.4, 0.8, 0.45], dtype=np.float64)
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)


@dataclass
class RenderResult:
    color: np.ndarray            # (H, W, 4) uint8
    depth: np.ndarray            # (H, W) float32, inf where untouched
    ids: np.ndarray              # (H, W) int32, -1 where untouched


def _clip_polygon_plane(pts, s, plane, keep_positive):
    """Sutherland-Hodgman clip against z >= c or z <= c in camera space."""
    c = plane
    z = pts[:, 2]
    d = z - c if keep_positive else c - z
    out_p, out_s = [], []
    n = len(pts)
    for i in range(n):
        j = (i + 1) % n
        di, dj = d[i], d[j]
        if di >= 0:
            out_p.append(pts[i])
            out_s.append(s[i])
        if (di >= 0) != (dj >= 0):
            t = di / (di - dj)
            out_p.append(pts[i] + t * (pts[j] - pts[i]))
            out_s.append(s[i] + t * (s[j] - s[i]))
    if not out_p:
        return None, None
    return np.asarray(out_p), np.asarray(out_s, dtype=np.float64)


def _clip_triangle(cam_pts, s_attr, near, far):
    """Clip one triangle to near/far planes; yield (pts, s) sub-triangles."""
    pts, s = _clip_polygon_plane(cam_pts, s_attr, near, keep_positive=True)
    if pts is None:
        return
    pts, s = _clip_polygon_plane(pts, s, far, keep_positive=False)
    if pts is None:
        return
    for i in range(1, len(pts) - 1):
        idx = (0, i, i + 1)
        yield pts[list(idx)], s[list(idx)]


def _raster_triangle(
    res,
    camera,
    cam_pts,
    s_attr,
    base_color,
    stripe_color,
    stripe_count,
    stripe_duty,
    brightness,
    seg_index,
    width,
    height,
    x_range,
):
    tan = math.tan(math.radians(camera.vfov_deg) * 0.5)
    z = cam_pts[:, 2]
    x_ndc = cam_pts[:, 0] / (z * tan * camera.aspect)
    y_ndc = cam_pts[:, 1] / (z * tan)
    sx = (x_ndc + 1.0) * 0.5 * width
    sy = (1.0 - y_ndc) * 0.5 * height

    area = (sx[1] - sx[0]) * (sy[2] - sy[0]) - (sx[2] - sx[0]) * (sy[1] - sy[0])
    if abs(area) < 1e-12:
        return

    rx0, rx1 = x_range
    bx0 = max(rx0, int(math.floor(sx.min())))
    bx1 = min(rx1, int(math.ceil(sx.max())) + 1)
    by0 = max(0, int(math.floor(sy.min())))
    by1 = min(height, int(math.ceil(sy.max())) + 1)
    if bx0 >= bx1 or by0 >= by1:
        return

    px = np.arange(bx0, bx1, dtype=np.float64) + 0.5
    py = np.arange(by0, by1, dtype=np.float64) + 0.5
    gx, gy = np.meshgrid(px, py)

    w0 = (sx[1] - gx) * (sy[2] - gy) - (sx[2] - gx) * (sy[1] - gy)
    w1 = (sx[2] - gx) * (sy[0] - gy) - (sx[0] - gx) * (sy[2] - gy)
    w2 = (sx[0] - gx) * (sy[1] - gy) - (sx[1] - gx) * (sy[0] - gy)
    l0, l1, l2 = w0 / area, w1 / area, w2 / area
    eps = -1e-9
    mask = (l0 >= eps) & (l1 >= eps) & (l2 >= eps)
    if not mask.any():
        return

    # Perspective-correct interpolation of depth and arc-length attribute.
    iz = l0 / z[0] + l1 / z[1] + l2 / z[2]
    z_pix = 1.0 / iz
    s_pix = (
        l0 * s_attr[0] / z[0] + l1 * s_attr[1] / z[1] + l2 * s_attr[2] / z[2]
    ) * z_pix

    depth_sub = res.depth[by0:by1, bx0:bx1]
    mask &= z_pix < depth_sub
    if not mask.any():
        return

    u = s_pix * stripe_count
    on = (u - np.floor(u)) < stripe_duty
    chosen = np.where(on[..., None], stripe_color, base_color)
    rgb = np.clip(chosen * brightness, 0, 255).astype(np.uint8)

    color_sub = res.color[by0:by1, bx0:bx1]
    ids_sub = res.ids[by0:by1, bx0:bx1]
    color_sub[mask, 0:3] = rgb[mask]
    color_sub[mask, 3] = 255
    depth_sub[mask] = z_pix[mask].astype(np.float32)
    ids_sub[mask] = seg_index


def render(
    scene,
    camera,
    width,
    height,
    x_range=None,
    into=None,
):
    """Render the scene. With x_range set, only that column band is written
    (tile-restricted redraw); results are bit-identical to a full render
    because all edge math is done in full-image pixel coordinates."""
    if width <= 0 or height <= 0:
        raise BadImageSizeError(
            "image size must be positive", width=width, height=height
        )
    if into is None:
        color = np.empty((height, width, 4), dtype=np.uint8)
        color[:, :] = np.array(BG_COLOR, dtype=np.uint8)
        res = RenderResult(
            color=color,
            depth=np.full((height, width), np.inf, dtype=np.float32),
            ids=np.full((height, width), -1, dtype=np.int32),
        )
    else:
        res = into
    if x_range is None:
        x_range = (0, width)

    for seg in scene.segments:
        verts_local, s_local, tris = tessellate(seg)
        world = world_transform(seg, verts_local)
        mins, maxs = world.min(axis=0), world.max(axis=0)
        if not camera.aabb_visible(mins, maxs):
            continue
        cam = camera.to_camera(world)
        base = np.asarray(scene.material_color(seg.material_id), dtype=np.float64)
        stripe = np.asarray(seg.stripe.color, dtype=np.float64)

        tri_pts = world[tris]  # (M, 3, 3)
        normals = np.cross(
            tri_pts[:, 1] - tri_pts[:, 0], tri_pts[:, 2] - tri_pts[:, 0]
        )
        norms = np.linalg.norm(normals, axis=1)
        norms[norms == 0] = 1.0
        normals = normals / norms[:, None]
        lambert = np.abs(normals @ LIGHT_DIR)
        brightness = 0.35 + 0.65 * lambert

        cam_tris = cam[tris]
        s_tris = s_local[tris]
        for ti in range(tris.shape[0]):
            for pts, s in _clip_triangle(
                cam_tris[ti], s_tris[ti], camera.near, camera.far
            ):
                _raster_triangle(
                    res,
                    camera,
                    pts,
                    s,
                    base,
                    stripe,
                    seg.stripe.count,
                    seg.stripe.duty,
                    float(brightness[ti]),
                    seg.index,
                    width,
                    height,
                    x_range,
                )
    return res
