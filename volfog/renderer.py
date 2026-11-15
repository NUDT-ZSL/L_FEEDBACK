"""Numba ray-marching kernels shared by the serial and parallel paths.

The two paths compile the exact same loop body; only the ``parallel`` flag of
the JIT decorator differs, which guarantees per-pixel equality.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange, set_num_threads

from .config import RenderConfig

EPS = 1e-8


@njit(cache=True, inline="always")
def sample_density(grid: np.ndarray, px: float, py: float, pz: float) -> float:
    """Trilinear lookup; the voxel cube spans world coordinates [-1, 1]^3."""
    n = grid.shape[0]
    if px < -1.0 or px > 1.0 or py < -1.0 or py > 1.0 or pz < -1.0 or pz > 1.0:
        return 0.0
    fx = (px + 1.0) * 0.5 * n - 0.5
    fy = (py + 1.0) * 0.5 * n - 0.5
    fz = (pz + 1.0) * 0.5 * n - 0.5
    ix = int(np.floor(fx))
    iy = int(np.floor(fy))
    iz = int(np.floor(fz))
    tx = fx - ix
    ty = fy - iy
    tz = fz - iz
    ix0 = max(0, min(ix, n - 1))
    iy0 = max(0, min(iy, n - 1))
    iz0 = max(0, min(iz, n - 1))
    ix1 = max(0, min(ix + 1, n - 1))
    iy1 = max(0, min(iy + 1, n - 1))
    iz1 = max(0, min(iz + 1, n - 1))

    c00 = grid[ix0, iy0, iz0] * (1.0 - tx) + grid[ix1, iy0, iz0] * tx
    c10 = grid[ix0, iy1, iz0] * (1.0 - tx) + grid[ix1, iy1, iz0] * tx
    c01 = grid[ix0, iy0, iz1] * (1.0 - tx) + grid[ix1, iy0, iz1] * tx
    c11 = grid[ix0, iy1, iz1] * (1.0 - tx) + grid[ix1, iy1, iz1] * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


@njit(cache=True, inline="always")
def light_transmittance(
    grid: np.ndarray,
    ox: float, oy: float, oz: float,
    lx: float, ly: float, lz: float,
    extinction: float, samples: int,
) -> float:
    """Beer-Lambert attenuation along the shadow segment to the point light."""
    dx = lx - ox
    dy = ly - oy
    dz = lz - oz
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < EPS:
        return 1.0
    dx /= dist
    dy /= dist
    dz /= dist
    dt = dist / samples
    tau = 0.0
    for k in range(samples):
        t = (k + 0.5) * dt
        tau += extinction * sample_density(grid, ox + dx * t, oy + dy * t, oz + dz * t) * dt
    return np.exp(-tau)


@njit(cache=True, inline="always")
def henyey_greenstein(cos_theta: float, g: float) -> float:
    denom = 1.0 + g * g - 2.0 * g * cos_theta
    return (1.0 - g * g) / (4.0 * np.pi * denom * np.sqrt(denom))


def _build_kernel(parallel: bool):
    @njit(cache=True, parallel=parallel, fastmath=False)
    def ray_march(
        density, blurred,
        color_out, trans_out, depth_out,
        cam_pos, cam_right, cam_up, cam_forward,
        focal, width, height,
        light_pos, light_color, light_intensity,
        step_size, max_samples,
        terminate_threshold, depth_threshold,
        extinction, albedo, phase_g,
        light_samples, mode, multiple_strength,
        max_distance,
    ):
        aspect = width / height
        for py in prange(height):
            for px in range(width):
                ndc_x = ((px + 0.5) / width) * 2.0 - 1.0
                ndc_y = 1.0 - ((py + 0.5) / height) * 2.0
                rdx = cam_forward[0] * focal + cam_right[0] * ndc_x * aspect + cam_up[0] * ndc_y
                rdy = cam_forward[1] * focal + cam_right[1] * ndc_x * aspect + cam_up[1] * ndc_y
                rdz = cam_forward[2] * focal + cam_right[2] * ndc_x * aspect + cam_up[2] * ndc_y
                rlen = np.sqrt(rdx * rdx + rdy * rdy + rdz * rdz)
                rdx /= rlen
                rdy /= rlen
                rdz /= rlen

                transmittance = 1.0
                depth = max_distance
                depth_hit = False
                cr = 0.0
                cg = 0.0
                cb = 0.0

                for i in range(max_samples):
                    t = (i + 0.5) * step_size
                    sx = cam_pos[0] + rdx * t
                    sy = cam_pos[1] + rdy * t
                    sz = cam_pos[2] + rdz * t
                    dens = sample_density(density, sx, sy, sz)
                    if dens <= 0.0:
                        if transmittance < terminate_threshold:
                            break
                        continue

                    tau = extinction * dens * step_size
                    beam_trans = 1.0 - np.exp(-tau)

                    ldx = light_pos[0] - sx
                    ldy = light_pos[1] - sy
                    ldz = light_pos[2] - sz
                    ldist = np.sqrt(ldx * ldx + ldy * ldy + ldz * ldz)
                    shadow = light_transmittance(
                        density, sx, sy, sz,
                        light_pos[0], light_pos[1], light_pos[2],
                        extinction, light_samples,
                    )
                    attenuation = light_intensity / (ldist * ldist + EPS)
                    cos_theta = (rdx * ldx + rdy * ldy + rdz * ldz) / (ldist + EPS)
                    phase = henyey_greenstein(cos_theta, phase_g)

                    incident = light_color * attenuation * shadow * phase
                    weight = transmittance * beam_trans * albedo
                    cr += weight * incident[0]
                    cg += weight * incident[1]
                    cb += weight * incident[2]

                    if mode == 1:
                        # Multiple-scattering approximation: the 3x3x3 averaged
                        # density stands in for radiance arriving after at least
                        # one extra scattering event; re-emission is isotropic.
                        ambient = sample_density(blurred, sx, sy, sz)
                        ms_weight = transmittance * beam_trans * albedo * multiple_strength
                        cr += ms_weight * ambient * light_color[0]
                        cg += ms_weight * ambient * light_color[1]
                        cb += ms_weight * ambient * light_color[2]

                    transmittance *= np.exp(-tau)
                    if (not depth_hit) and transmittance < depth_threshold:
                        depth = t
                        depth_hit = True
                    if transmittance < terminate_threshold:
                        break

                color_out[py, px, 0] = cr
                color_out[py, px, 1] = cg
                color_out[py, px, 2] = cb
                trans_out[py, px] = transmittance
                depth_out[py, px] = depth

    return ray_march


_serial_kernel = _build_kernel(parallel=False)
_parallel_kernel = _build_kernel(parallel=True)


def blur_density_field(density: np.ndarray) -> np.ndarray:
    """Uniform 3x3x3 average used as the multiple-scattering proxy field."""
    padded = np.pad(density, 1, mode="constant", constant_values=0.0)
    blurred = np.zeros_like(density)
    for ox in range(3):
        for oy in range(3):
            for oz in range(3):
                blurred += padded[ox:ox + density.shape[0],
                                  oy:oy + density.shape[1],
                                  oz:oz + density.shape[2]]
    return blurred / 27.0


def _camera_frame(config: RenderConfig):
    eye = np.asarray(config.camera_pos, dtype=np.float64)
    target = np.asarray(config.camera_target, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return eye, right, up, forward


def render_frame(config: RenderConfig, density: np.ndarray):
    """Render one frame. Returns (linear HDR color, transmittance, depth)."""
    config.validate()
    density = density.astype(np.float64)
    blurred = blur_density_field(density)

    eye, right, up, forward = _camera_frame(config)
    width, height = config.width, config.height
    color = np.zeros((height, width, 3), dtype=np.float64)
    trans = np.ones((height, width), dtype=np.float64)
    depth = np.full((height, width), config.max_distance, dtype=np.float64)

    args = (
        density, blurred,
        color, trans, depth,
        eye, right, up, forward,
        float(config.focal_length), width, height,
        np.asarray(config.light_pos, dtype=np.float64),
        np.asarray(config.light_color, dtype=np.float64),
        float(config.light_intensity),
        float(config.step_size), config.max_samples,
        float(config.terminate_threshold), float(config.depth_threshold),
        float(config.extinction), float(config.albedo), float(config.phase_g),
        config.light_samples, 1 if config.mode == "multiple" else 0,
        float(config.multiple_strength),
        float(config.max_distance),
    )
    if config.parallel:
        if config.threads > 0:
            set_num_threads(config.threads)
        _parallel_kernel(*args)
    else:
        _serial_kernel(*args)
    return color, trans, depth
