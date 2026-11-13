"""Hemisphere cosine-weighted irradiance core.

Determinism contract:
  * float64 accumulation, fixed loop orders (hemisphere sample index, ray
    step index), no BLAS calls, only elementwise ops and np.sum over fixed
    axes (pairwise order is a pure function of shape/strides).
  * A block's output depends only on the heightfield and parameters, never
    on which other blocks run concurrently, so results are bit-identical
    for any worker-thread count.
"""
import numpy as np

HEMI_SAMPLES = 32          # fixed hemisphere direction count
RAY_STEPS = 24             # fixed ray-march step count
RAY_STEP_LEN = 1.5         # march step length in heightfield samples
RAY_MAX_DIST = RAY_STEPS * RAY_STEP_LEN
CLEARANCE = 0.25           # probe lift above surface, in world z units
DEFAULT_Z_SCALE = 16.0

SKY_COLOR = (0.35, 0.5, 0.9)
LIGHT_COLOR = (1.0, 0.95, 0.85)


def _fibonacci_hemisphere(n: int) -> np.ndarray:
    """Fixed z-up hemisphere direction set, shape (n, 3)."""
    idx = np.arange(n, dtype=np.float64)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    cos_theta = (idx + 0.5) / n
    sin_theta = np.sqrt(np.clip(1.0 - cos_theta * cos_theta, 0.0, 1.0))
    phi = (idx * golden) % (2.0 * np.pi)
    return np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=1
    )


def _sample_bilinear(hf: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear heightfield sample at float coords, clamped to bounds."""
    h, w = hf.shape
    xs = np.clip(xs, 0.0, w - 1.0)
    ys = np.clip(ys, 0.0, h - 1.0)
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = xs - x0
    fy = ys - y0
    v00 = hf[y0, x0]
    v10 = hf[y0, x1]
    v01 = hf[y1, x0]
    v11 = hf[y1, x1]
    top = v00 + (v10 - v00) * fx
    bot = v01 + (v11 - v01) * fx
    return top + (bot - top) * fy


def _march_visibility(hf, z_scale, ox, oy, oz, dx, dy, dz):
    """Binary visibility along rays from (ox,oy,oz) toward (dx,dy,dz).

    All direction arrays share one broadcastable shape; the step loop is a
    fixed ascending sequence, so the occlusion reduction order is fixed.
    """
    occluded = np.zeros(np.broadcast(ox, dx).shape, dtype=np.float64)
    for step in range(1, RAY_STEPS + 1):
        t = step * RAY_STEP_LEN
        sx = ox + dx * t
        sy = oy + dy * t
        sz = oz + dz * t
        terrain_z = _sample_bilinear(hf, sx, sy) * z_scale
        occluded = np.maximum(occluded, (terrain_z > sz).astype(np.float64))
    return 1.0 - occluded


def bake_block(
    hf: np.ndarray,
    z_scale: float,
    spacing: int,
    block_ox: int,
    block_oy: int,
    block_w: int,
    block_h: int,
    light_dir,
    light_color=LIGHT_COLOR,
    sky_color=SKY_COLOR,
) -> np.ndarray:
    """Bake one probe block; returns float32 array of shape (bh, bw, 4).

    RGB = sky irradiance (cosine-weighted, occlusion-aware) + directional
    light contribution; A = cosine-weighted sky visibility fraction.
    """
    hf64 = hf.astype(np.float64)
    h, w = hf.shape
    py, px = np.mgrid[0:block_h, 0:block_w]
    gx = np.minimum((block_ox + px) * spacing, w - 1).astype(np.float64)
    gy = np.minimum((block_oy + py) * spacing, h - 1).astype(np.float64)

    # Central-difference normals (clamped stencil at borders).
    ix = gx.astype(np.int64)
    iy = gy.astype(np.int64)
    hx_l = hf64[iy, np.maximum(ix - 1, 0)]
    hx_r = hf64[iy, np.minimum(ix + 1, w - 1)]
    hy_d = hf64[np.maximum(iy - 1, 0), ix]
    hy_u = hf64[np.minimum(iy + 1, h - 1), ix]
    span_x = (np.minimum(ix + 1, w - 1) - np.maximum(ix - 1, 0)).astype(np.float64)
    span_y = (np.minimum(iy + 1, h - 1) - np.maximum(iy - 1, 0)).astype(np.float64)
    dzdx = (hx_r - hx_l) * z_scale / span_x
    dzdy = (hy_u - hy_d) * z_scale / span_y
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(nx)
    inv_len = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx * inv_len, ny * inv_len, nz * inv_len

    oz = hf64[iy, ix] * z_scale + CLEARANCE
    ox = gx
    oy = gy

    # Tangent frame per probe (fixed formulas, no branching on values).
    ref_x = np.where(np.abs(nz) > 0.999, 1.0, 0.0)
    ref_z = np.where(np.abs(nz) > 0.999, 0.0, 1.0)
    # tangent = normalize(cross(ref, n))
    tx = 0.0 * nz - ref_z * ny
    ty = ref_z * nx - ref_x * nz
    tz = ref_x * ny - 0.0 * nx
    tlen = np.sqrt(tx * tx + ty * ty + tz * tz)
    tx, ty, tz = tx / tlen, ty / tlen, tz / tlen
    # bitangent = cross(n, tangent)
    bx = ny * tz - nz * ty
    by = nz * tx - nx * tz
    bz = nx * ty - ny * tx

    hemi = _fibonacci_hemisphere(HEMI_SAMPLES)          # (M, 3) local frame
    weights = hemi[:, 2] * (2.0 * np.pi / HEMI_SAMPLES)  # cosine * dOmega

    p = block_h * block_w
    fx = (tx.reshape(p, 1) * hemi[None, :, 0]
          + bx.reshape(p, 1) * hemi[None, :, 1]
          + nx.reshape(p, 1) * hemi[None, :, 2])
    fy = (ty.reshape(p, 1) * hemi[None, :, 0]
          + by.reshape(p, 1) * hemi[None, :, 1]
          + ny.reshape(p, 1) * hemi[None, :, 2])
    fz = (tz.reshape(p, 1) * hemi[None, :, 0]
          + bz.reshape(p, 1) * hemi[None, :, 1]
          + nz.reshape(p, 1) * hemi[None, :, 2])

    vis = _march_visibility(
        hf64, z_scale,
        ox.reshape(p, 1), oy.reshape(p, 1), oz.reshape(p, 1), fx, fy, fz,
    )
    # Fixed-axis reduction: sum over hemisphere sample axis (axis=1).
    sky_integral = np.sum(vis * weights[None, :], axis=1) / np.pi  # (p,)

    # Directional light: single shadow ray toward the light.
    lx, ly, lz = (float(v) for v in light_dir)
    llen = float(np.sqrt(lx * lx + ly * ly + lz * lz))
    if llen == 0.0:
        raise ValueError("light direction must be non-zero")
    lx, ly, lz = lx / llen, ly / llen, lz / llen
    ndotl = np.clip(nx.reshape(p) * lx + ny.reshape(p) * ly + nz.reshape(p) * lz,
                    0.0, None)
    lvis = _march_visibility(
        hf64, z_scale,
        ox.reshape(p), oy.reshape(p), oz.reshape(p),
        np.full(p, lx), np.full(p, ly), np.full(p, lz),
    )
    direct = ndotl * lvis  # (p,)

    sky_rgb = np.asarray(sky_color, dtype=np.float64)
    light_rgb = np.asarray(light_color, dtype=np.float64)
    rgb = sky_rgb[None, :] * sky_integral[:, None] + light_rgb[None, :] * direct[:, None]
    out = np.empty((p, 4), dtype=np.float64)
    out[:, 0:3] = rgb
    out[:, 3] = sky_integral
    return out.reshape(block_h, block_w, 4).astype(np.float32)


def probe_grid_dims(width: int, height: int, spacing: int):
    nx = (width + spacing - 1) // spacing
    ny = (height + spacing - 1) // spacing
    return nx, ny


def affected_blocks(edit_rect, spacing, block_size, probe_nx, probe_ny):
    """Probe blocks touched by an edit rect, dilated by the full support of
    a probe (ray-march reach + normal stencil), in pixel coordinates."""
    x, y, w, h = (int(v) for v in edit_rect)
    pad = int(np.ceil(RAY_MAX_DIST)) + 2
    x0 = x - pad
    y0 = y - pad
    x1 = x + w + pad
    y1 = y + h + pad
    # Probe i sits at pixel i*spacing; block b covers probes [b*B, (b+1)*B).
    bx0 = max(0, x0 // spacing // block_size)
    by0 = max(0, y0 // spacing // block_size)
    bx1 = min((probe_nx + block_size - 1) // block_size,
              (max(x1 - 1, 0) // spacing) // block_size + 1)
    by1 = min((probe_ny + block_size - 1) // block_size,
              (max(y1 - 1, 0) // spacing) // block_size + 1)
    return [(bx, by) for by in range(by0, by1) for bx in range(bx0, bx1)]
