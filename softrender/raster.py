"""Fixed-point, perspective-correct, tiled multithreaded Z-buffer rasterizer.

* Screen coordinates are quantized to 1/256 pixel (exceeds the required
  1/16 pixel fixed-point precision).
* Coverage uses integer edge functions with the top-left fill rule, so
  adjacent triangles share edges without cracks or double writes.
* The depth buffer stores float32 and compares with "less than".
* The frame is split into 16x16 pixel tiles rendered by a thread pool.
  Each tile keeps its own local depth buffer which is merged into the
  global Z-buffer when the tile finishes; tiles are disjoint, so the
  result is bit-identical regardless of thread count or scheduling.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .errors import ParameterError
from .geometry import SUBPIXEL

TILE = 16


def _signed_area2(x, y):
    """Twice the signed area, consistent with the edge functions below:
    A2 = E(c1, c2, c0) where E(a, b, p) = (p - a) x (b - a)."""
    return ((x[:, 0] - x[:, 1]) * (y[:, 2] - y[:, 1])
            - (y[:, 0] - y[:, 1]) * (x[:, 2] - x[:, 1]))


def validate_threads(nthreads: int) -> int:
    cores = os.cpu_count() or 1
    if not (1 <= nthreads <= cores):
        raise ParameterError(
            f"thread count {nthreads} out of range: must be between 1 and {cores} "
            f"(logical cores on this machine)"
        )
    return nthreads


def _orient(stris: dict) -> np.ndarray:
    """Fix winding so A2 > 0; return indices of kept (non-degenerate) tris."""
    x = stris["sx"]
    y = stris["sy"]
    a2 = _signed_area2(x, y)
    keep = a2 != 0
    flip = a2 < 0
    for key in ("sx", "sy", "z", "w", "world", "nrm", "uv"):
        arr = stris[key]
        arr[flip] = arr[flip][:, ::-1] if arr.ndim == 2 else arr[flip][:, ::-1, :]
    return np.nonzero(keep)[0]


def _prepare(stris: dict) -> dict:
    """Precompute per-triangle integer edge data (operates on a copy)."""
    g = {k: v for k, v in stris.items()}
    keep = _orient(g)
    for key in list(g.keys()):
        g[key] = g[key][keep]
    x = g["sx"]
    y = g["sy"]
    g["a2"] = _signed_area2(x, y).astype(np.float64)
    # Top-left fill rule (valid for the A2 > 0 orientation used here).
    dx = np.roll(x, -1, axis=1) - x
    dy = np.roll(y, -1, axis=1) - y
    g["topleft"] = (dy > 0) | ((dy == 0) & (dx < 0))
    g["px0"] = np.floor_divide(x.min(axis=1), SUBPIXEL)
    g["px1"] = np.floor_divide(x.max(axis=1), SUBPIXEL)
    g["py0"] = np.floor_divide(y.min(axis=1), SUBPIXEL)
    g["py1"] = np.floor_divide(y.max(axis=1), SUBPIXEL)
    return g


def _draw_triangle_tile(g, t, tx, ty, ldepth, lcolor, shader, radial_from):
    x = g["sx"][t]
    y = g["sy"][t]
    px0 = max(g["px0"][t], tx)
    px1 = min(g["px1"][t], tx + TILE - 1)
    py0 = max(g["py0"][t], ty)
    py1 = min(g["py1"][t], ty + TILE - 1)
    if px0 > px1 or py0 > py1:
        return
    xs = (np.arange(px0, px1 + 1, dtype=np.int64) << 8) + (SUBPIXEL // 2)
    ys = (np.arange(py0, py1 + 1, dtype=np.int64) << 8) + (SUBPIXEL // 2)
    x0, x1, x2 = int(x[0]), int(x[1]), int(x[2])
    y0, y1, y2 = int(y[0]), int(y[1]), int(y[2])
    e0 = (xs[None, :] - x1) * (y2 - y1) - (ys[:, None] - y1) * (x2 - x1)
    e1 = (xs[None, :] - x2) * (y0 - y2) - (ys[:, None] - y2) * (x0 - x2)
    e2 = (xs[None, :] - x0) * (y1 - y0) - (ys[:, None] - y0) * (x1 - x0)
    tl = g["topleft"][t]
    mask = ((e0 > 0) | ((e0 == 0) & tl[0])) & \
           ((e1 > 0) | ((e1 == 0) & tl[1])) & \
           ((e2 > 0) | ((e2 == 0) & tl[2]))
    if not mask.any():
        return
    a2 = g["a2"][t]
    l0 = e0[mask] / a2
    l1 = e1[mask] / a2
    l2 = e2[mask] / a2
    zv = g["z"][t]
    wv = g["w"][t]
    iw = l0 / wv[0] + l1 / wv[1] + l2 / wv[2]
    b0 = l0 / wv[0] / iw
    b1 = l1 / wv[1] / iw
    b2 = l2 / wv[2] / iw
    if radial_from is not None:
        wp = g["world"][t]
        pos = b0[:, None] * wp[0] + b1[:, None] * wp[1] + b2[:, None] * wp[2]
        z = np.linalg.norm(pos - radial_from, axis=1)
    else:
        z = l0 * zv[0] + l1 * zv[1] + l2 * zv[2]
    z32 = z.astype(np.float32)
    yy, xx = np.nonzero(mask)
    ly = py0 + yy - ty
    lx = px0 + xx - tx
    upd = z32 < ldepth[ly, lx]
    if not upd.any():
        return
    ly = ly[upd]
    lx = lx[upd]
    ldepth[ly, lx] = z32[upd]
    if shader is None or lcolor is None:
        return
    # Recompute attribute interpolation only for depth-passing pixels.
    b0 = b0[upd]
    b1 = b1[upd]
    b2 = b2[upd]
    data = {"px": (px0 + xx[upd]).astype(np.int64), "py": (py0 + yy[upd]).astype(np.int64)}
    for key, arr in (("world", g["world"][t]), ("nrm", g["nrm"][t]), ("uv", g["uv"][t])):
        data[key] = b0[:, None] * arr[0] + b1[:, None] * arr[1] + b2[:, None] * arr[2]
    rgb = shader(data, t)
    lcolor[ly, lx] = rgb.astype(np.float32)


def _render_tile(g, tile_tris, tx, ty, width, height, shader, radial_from, background):
    ldepth = np.full((TILE, TILE), np.inf, dtype=np.float32)
    lcolor = None
    if shader is not None:
        lcolor = np.zeros((TILE, TILE, 3), dtype=np.float32)
        h = min(TILE, height - ty)
        w = min(TILE, width - tx)
        lcolor[:h, :w] = background[ty:ty + h, tx:tx + w]
    for t in tile_tris:
        _draw_triangle_tile(g, t, tx, ty, ldepth, lcolor, shader, radial_from)
    return tx, ty, ldepth, lcolor


def rasterize(stris: dict, width: int, height: int, nthreads: int = 1,
              shader=None, background=None, radial_depth_from=None):
    """Rasterize triangles. Returns (color HxWx3 float32 or None, depth HxW float32)."""
    validate_threads(nthreads)
    g = _prepare(stris)
    ntris = g["sx"].shape[0]
    depth = np.full((height, width), np.inf, dtype=np.float32)
    color = None
    if shader is not None:
        if background is None:
            background = np.zeros((height, width, 3), dtype=np.float32)
        color = background.astype(np.float32).copy()
    # Bin triangles into tiles.
    tiles_x = (width + TILE - 1) // TILE
    tiles_y = (height + TILE - 1) // TILE
    bins = [[[] for _ in range(tiles_x)] for _ in range(tiles_y)]
    for t in range(ntris):
        tx0 = max(g["px0"][t] // TILE, 0)
        tx1 = min(g["px1"][t] // TILE, tiles_x - 1)
        ty0 = max(g["py0"][t] // TILE, 0)
        ty1 = min(g["py1"][t] // TILE, tiles_y - 1)
        for tyi in range(ty0, ty1 + 1):
            for txi in range(tx0, tx1 + 1):
                bins[tyi][txi].append(t)
    radial_from = None
    if radial_depth_from is not None:
        radial_from = np.asarray(radial_depth_from, dtype=np.float64)
    tasks = []
    for tyi in range(tiles_y):
        for txi in range(tiles_x):
            if bins[tyi][txi] or shader is not None:
                tasks.append((txi * TILE, tyi * TILE, bins[tyi][txi]))

    def work(args):
        tx, ty, tris = args
        return _render_tile(g, tris, tx, ty, width, height, shader, radial_from, background)

    if nthreads == 1:
        results = [work(a) for a in tasks]
    else:
        with ThreadPoolExecutor(max_workers=nthreads) as pool:
            results = list(pool.map(work, tasks))
    # Merge per-tile local depth/color into the global buffers.
    for tx, ty, ldepth, lcolor in results:
        h = min(TILE, height - ty)
        w = min(TILE, width - tx)
        depth[ty:ty + h, tx:tx + w] = ldepth[:h, :w]
        if color is not None:
            color[ty:ty + h, tx:tx + w] = lcolor[:h, :w]
    return color, depth


def coverage_count(stris: dict, width: int, height: int) -> np.ndarray:
    """Debug/test helper: per-pixel coverage counts (no depth test)."""
    g = _prepare(stris)
    counts = np.zeros((height, width), dtype=np.int32)
    for t in range(g["sx"].shape[0]):
        px0 = max(int(g["px0"][t]), 0)
        px1 = min(int(g["px1"][t]), width - 1)
        py0 = max(int(g["py0"][t]), 0)
        py1 = min(int(g["py1"][t]), height - 1)
        if px0 > px1 or py0 > py1:
            continue
        xs = (np.arange(px0, px1 + 1, dtype=np.int64) << 8) + (SUBPIXEL // 2)
        ys = (np.arange(py0, py1 + 1, dtype=np.int64) << 8) + (SUBPIXEL // 2)
        x = g["sx"][t]
        y = g["sy"][t]
        x0, x1, x2 = int(x[0]), int(x[1]), int(x[2])
        y0, y1, y2 = int(y[0]), int(y[1]), int(y[2])
        e0 = (xs[None, :] - x1) * (y2 - y1) - (ys[:, None] - y1) * (x2 - x1)
        e1 = (xs[None, :] - x2) * (y0 - y2) - (ys[:, None] - y2) * (x0 - x2)
        e2 = (xs[None, :] - x0) * (y1 - y0) - (ys[:, None] - y0) * (x1 - x0)
        tl = g["topleft"][t]
        mask = ((e0 > 0) | ((e0 == 0) & tl[0])) & \
               ((e1 > 0) | ((e1 == 0) & tl[1])) & \
               ((e2 > 0) | ((e2 == 0) & tl[2]))
        counts[py0:py1 + 1, px0:px1 + 1] += mask.astype(np.int32)
    return counts
