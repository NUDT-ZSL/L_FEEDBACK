"""Flat-shaded orthographic preview rasterizer and overview sheet."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .clip import OrthoFrustum

VIEWPORT_WIDTH = 64
OVERVIEW_COLUMNS = 10

LIGHT_DIR = np.array([0.4, 0.7, -0.6]) / np.linalg.norm([0.4, 0.7, -0.6])
BASE_COLOR = np.array([200.0, 120.0, 90.0])
BACKGROUND = (20, 20, 28)
AMBIENT = 0.2


def viewport_height(frustum: OrthoFrustum) -> int:
    """Viewport height derived from the projection parameters."""
    return max(1, round(VIEWPORT_WIDTH * frustum.half_height
                        / frustum.half_width))


def rasterize(
    positions_ndc: np.ndarray,
    face_normals: np.ndarray,
    face_indices: np.ndarray,
    width: int = VIEWPORT_WIDTH,
    height: int = VIEWPORT_WIDTH,
) -> Image.Image:
    """Rasterize clipped NDC triangle soup with per-pixel depth testing.

    Flat shading with one fixed directional light; depth is the linear NDC
    z, smallest value wins.
    """
    color = np.full((height, width, 3), BACKGROUND, dtype=np.float64)
    depth = np.full((height, width), np.inf, dtype=np.float64)

    n_tris = len(positions_ndc) // 3
    for t in range(n_tris):
        tri = positions_ndc[3 * t:3 * t + 3]
        fn = face_normals[face_indices[t]] if len(face_normals) else None
        if fn is None or not np.any(fn):
            shade = AMBIENT
        else:
            shade = AMBIENT + (1.0 - AMBIENT) * max(0.0, float(fn @ LIGHT_DIR))
        rgb = np.clip(BASE_COLOR * shade, 0.0, 255.0)

        xs = (tri[:, 0] + 1.0) * 0.5 * width
        ys = (1.0 - (tri[:, 1] + 1.0) * 0.5) * height
        zs = tri[:, 2]
        min_x = max(0, int(np.floor(xs.min())))
        max_x = min(width - 1, int(np.ceil(xs.max())))
        min_y = max(0, int(np.floor(ys.min())))
        max_y = min(height - 1, int(np.ceil(ys.max())))
        if min_x > max_x or min_y > max_y:
            continue

        d = (xs[1] - xs[0]) * (ys[2] - ys[0]) \
            - (xs[2] - xs[0]) * (ys[1] - ys[0])
        if abs(d) < 1e-12:
            continue
        for py in range(min_y, max_y + 1):
            cy = py + 0.5
            for px in range(min_x, max_x + 1):
                cx = px + 0.5
                w1 = ((xs[2] - cx) * (ys[0] - cy)
                      - (xs[0] - cx) * (ys[2] - cy)) / d
                w2 = ((xs[0] - cx) * (ys[1] - cy)
                      - (xs[1] - cx) * (ys[0] - cy)) / d
                w0 = 1.0 - w1 - w2
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                z = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
                if z < depth[py, px]:
                    depth[py, px] = z
                    color[py, px] = rgb

    return Image.fromarray(color.astype(np.uint8), mode="RGB")


def build_overview(
    frame_images: list[Image.Image],
    columns: int = OVERVIEW_COLUMNS,
) -> Image.Image:
    """Tile frame thumbnails into a single overview sheet."""
    if not frame_images:
        return Image.new("RGB", (1, 1), BACKGROUND)
    cell_w, cell_h = frame_images[0].size
    rows = (len(frame_images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), BACKGROUND)
    for idx, img in enumerate(frame_images):
        sheet.paste(img, ((idx % columns) * cell_w, (idx // columns) * cell_h))
    return sheet
