"""Core rasterization: painter order + per-pixel depth correction,
back-face refraction sampling and premultiplied-alpha compositing.

The same `render_region` routine serves full-frame and incremental
rendering. All per-pixel math is elementwise, so restricting the write
region cannot change values at pixels inside the region, provided every
refraction sample lands inside the region (guaranteed by the dirty-rect
fixpoint in renderer.py).
"""

from __future__ import annotations

import numpy as np

from .scene import CrystalState

LAYER_Z_STEP = 2.0


def _points_in_polygon(xs: np.ndarray, ys: np.ndarray, poly: np.ndarray) -> np.ndarray:
    inside = np.zeros(xs.shape, dtype=bool)
    xv = poly[:, 0]
    yv = poly[:, 1]
    count = len(poly)
    j = count - 1
    for i in range(count):
        crosses = (yv[i] > ys) != (yv[j] > ys)
        x_at = (xv[j] - xv[i]) * (ys - yv[i]) / (yv[j] - yv[i]) + xv[i]
        inside ^= crosses & (xs < x_at)
        j = i
    return inside


def _coverage(state: CrystalState, height: int, width: int):
    poly = state.screen_vertices
    x0 = max(0, int(np.floor(poly[:, 0].min())))
    y0 = max(0, int(np.floor(poly[:, 1].min())))
    x1 = min(width, int(np.ceil(poly[:, 0].max())) + 1)
    y1 = min(height, int(np.ceil(poly[:, 1].max())) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    xs = xs.astype(np.float64) + 0.5
    ys = ys.astype(np.float64) + 0.5
    mask = _points_in_polygon(xs, ys, poly)
    if not mask.any():
        return None
    gx, gy = state.depth_gradient
    z = state.layer * LAYER_Z_STEP + gx * xs + gy * ys
    return (x0, y0, mask, z)


def render_region(
    states: list[CrystalState],
    env: np.ndarray,
    region: tuple[int, int, int, int],
    acc: np.ndarray,
) -> int:
    """Composite all visible crystals into acc, restricted to region.

    acc must already hold the env background inside region. Refraction
    samples are gathered from acc itself (content behind the crystal in
    per-pixel depth order). Returns the number of shaded fragments.
    """
    height, width = acc.shape[0], acc.shape[1]
    rx0, ry0, rx1, ry1 = region

    ordered = sorted(
        (s for s in states if s.visible), key=lambda s: s.crystal_id
    )
    coverages = []
    for state in ordered:
        cov = _coverage(state, height, width)
        coverages.append(cov)

    n = len(ordered)
    covered_full = []
    z_full = []
    for cov in coverages:
        mask = np.zeros((height, width), dtype=bool)
        z = np.full((height, width), -np.inf, dtype=np.float64)
        if cov is not None:
            x0, y0, sub_mask, sub_z = cov
            h, w = sub_mask.shape
            mask[y0 : y0 + h, x0 : x0 + w] = sub_mask
            z[y0 : y0 + h, x0 : x0 + w] = sub_z
        covered_full.append(mask)
        z_full.append(z)

    ranks = [np.zeros((height, width), dtype=np.int64) for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            farther = covered_full[j] & covered_full[i] & (
                (z_full[j] < z_full[i])
                | (
                    (z_full[j] == z_full[i])
                    & (ordered[j].crystal_id < ordered[i].crystal_id)
                )
            )
            ranks[i] += farther.astype(np.int64)

    region_mask = np.zeros((height, width), dtype=bool)
    region_mask[ry0:ry1, rx0:rx1] = True

    shaded = 0
    for rank in range(n):
        for i in range(n):
            mask = covered_full[i] & (ranks[i] == rank) & region_mask
            if not mask.any():
                continue
            shaded += _shade_fragment(ordered[i], mask, acc)
    return shaded


def _shade_fragment(state: CrystalState, mask: np.ndarray, acc: np.ndarray) -> int:
    height, width = acc.shape[0], acc.shape[1]
    ys, xs = np.nonzero(mask)

    verts = state.screen_vertices
    centroid = verts.mean(axis=0)
    radius = float(np.sqrt(((verts - centroid) ** 2).sum(axis=1)).max())
    radius = max(radius, 1e-6)

    normal_x = (xs.astype(np.float64) + 0.5 - centroid[0]) / radius
    normal_y = (ys.astype(np.float64) + 0.5 - centroid[1]) / radius
    magnitude = state.refraction_radius

    qx = np.rint(xs + 0.5 + normal_x * magnitude).astype(np.int64)
    qy = np.rint(ys + 0.5 + normal_y * magnitude).astype(np.int64)
    np.clip(qx, 0, width - 1, out=qx)
    np.clip(qy, 0, height - 1, out=qy)

    sample = acc[qy, qx, :]

    red, green, blue, alpha = state.color
    tint = np.array([red, green, blue], dtype=np.float64)
    out_rgb = alpha * tint[None, :] + (1.0 - alpha) * tint[None, :] * sample[:, :3]
    out_alpha = alpha + (1.0 - alpha) * sample[:, 3]

    acc[ys, xs, 0] = out_rgb[:, 0]
    acc[ys, xs, 1] = out_rgb[:, 1]
    acc[ys, xs, 2] = out_rgb[:, 2]
    acc[ys, xs, 3] = out_alpha
    return int(len(xs))


def to_framebuffer(acc: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(acc, 0.0, 1.0) * 255.0).astype(np.uint8)
