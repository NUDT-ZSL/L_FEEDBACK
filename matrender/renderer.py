"""Deterministic CPU path tracer for the material-sphere regression scenes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import os
from typing import Any

import numpy as np

from .reporting import build_material_regions, summarize_linear
from .scene import Scene


_EPS = 1.0e-9
_SHADOW_EPS = 1.0e-5
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


@dataclass(frozen=True)
class RenderOptions:
    samples: int
    threads: int
    tile_size: int
    max_memory_bytes: int
    ssao: bool | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class RenderPlan:
    width: int
    height: int
    samples: int
    requested_width: int
    requested_height: int
    requested_samples: int
    threads: int
    requested_threads: int
    logical_cores: int
    tile_size: int
    ssao_enabled: bool
    max_memory_bytes: int
    initial_estimated_bytes: int
    estimated_bytes: int
    degradation_events: tuple[dict[str, Any], ...]
    warnings: tuple[dict[str, Any], ...]

    @property
    def degraded(self) -> bool:
        return bool(self.degradation_events)


@dataclass(frozen=True)
class GBuffer:
    sphere_index: np.ndarray
    depth: np.ndarray
    normal: np.ndarray
    ray_direction: np.ndarray


@dataclass(frozen=True)
class RenderResult:
    linear_hdr: np.ndarray
    sphere_masks: tuple[np.ndarray, ...]
    sphere_index: np.ndarray
    ambient_occlusion: np.ndarray
    plan: RenderPlan
    global_stats: dict[str, object]
    region_stats: tuple[dict[str, object], ...]
    material_regions: tuple[dict[str, object], ...]


def estimate_memory_bytes(
    width: int,
    height: int,
    samples: int,
    threads: int,
    tile_size: int,
) -> int:
    """Conservative full-frame estimate used by resource preflight.

    The implementation renders only one tile at a time per worker, but the
    preflight deliberately models a full-frame float64 per-sample buffer. This
    makes the budget independent of scheduling details and conservative.
    """

    pixels = width * height
    bytes_per_sample = 3 * 8
    accumulation_bytes = 3 * 8
    gbuffer_bytes = 4 + 8 + 3 * 8 + 3 * 8
    display_buffer_bytes = 3 * 8
    worker_pixels = min(tile_size, width) * min(tile_size, height)
    worker_scratch_bytes = threads * worker_pixels * 128
    return (
        pixels
        * (bytes_per_sample * samples + accumulation_bytes + gbuffer_bytes)
        + pixels * display_buffer_bytes
        + worker_scratch_bytes
    )


def make_render_plan(scene: Scene, options: RenderOptions) -> RenderPlan:
    """Clamp thread use, preflight memory, and produce a deterministic plan."""

    if not 1 <= options.samples <= 64:
        raise ValueError("samples must be in [1, 64]")
    if options.threads < 1:
        raise ValueError("threads must be at least 1")
    if options.tile_size < 1:
        raise ValueError("tile_size must be at least 1")
    if options.max_memory_bytes < 1:
        raise ValueError("max_memory_bytes must be positive")

    width = scene.width if options.width is None else options.width
    height = scene.height if options.height is None else options.height
    if not 1 <= width <= 8192 or not 1 <= height <= 8192:
        raise ValueError("render resolution must be in [1, 8192]")

    logical_cores = os.cpu_count() or 1
    used_threads = min(options.threads, logical_cores)
    warnings: list[dict[str, Any]] = []
    if options.threads > logical_cores:
        warnings.append(
            {
                "code": "W_THREAD_LIMIT",
                "severity": "warning",
                "message": (
                    f"requested {options.threads} threads, limited to "
                    f"{logical_cores} logical cores"
                ),
            }
        )

    requested_width = width
    requested_height = height
    requested_samples = options.samples
    samples = options.samples
    events: list[dict[str, Any]] = []
    initial_estimate = estimate_memory_bytes(
        width, height, samples, used_threads, options.tile_size
    )
    estimate = initial_estimate

    while estimate > options.max_memory_bytes and samples > 1:
        new_samples = max(1, samples // 2)
        events.append(
            {
                "action": "reduce_samples",
                "from": samples,
                "to": new_samples,
                "estimated_bytes_before": estimate,
            }
        )
        samples = new_samples
        estimate = estimate_memory_bytes(
            width, height, samples, used_threads, options.tile_size
        )

    while estimate > options.max_memory_bytes and (width > 1 or height > 1):
        new_width = max(1, int(math.floor(width * 0.75)))
        new_height = max(1, int(math.floor(height * 0.75)))
        if new_width == width and new_height == height:
            break
        events.append(
            {
                "action": "reduce_resolution",
                "from": [width, height],
                "to": [new_width, new_height],
                "estimated_bytes_before": estimate,
            }
        )
        width, height = new_width, new_height
        estimate = estimate_memory_bytes(
            width, height, samples, used_threads, options.tile_size
        )

    if events:
        if estimate > options.max_memory_bytes:
            events.append(
                {
                    "action": "cannot_reduce_further",
                    "estimated_bytes": estimate,
                    "limit_bytes": options.max_memory_bytes,
                }
            )
            warnings.append(
                {
                    "code": "E_RESOURCE_LIMIT",
                    "severity": "warning",
                    "message": (
                        f"memory estimate {initial_estimate} B exceeds limit "
                        f"{options.max_memory_bytes} B; degraded to "
                        f"{width}x{height}, {samples} spp, but the irreducible "
                        f"estimate is {estimate} B"
                    ),
                }
            )
        else:
            warnings.append(
                {
                    "code": "E_RESOURCE_LIMIT",
                    "severity": "warning",
                    "message": (
                        f"memory estimate {initial_estimate} B exceeds limit "
                        f"{options.max_memory_bytes} B; degraded to "
                        f"{width}x{height}, {samples} spp "
                        f"(estimated {estimate} B)"
                    ),
                }
            )

    ssao_enabled = scene.ssao_enabled if options.ssao is None else options.ssao
    return RenderPlan(
        width=width,
        height=height,
        samples=samples,
        requested_width=requested_width,
        requested_height=requested_height,
        requested_samples=requested_samples,
        threads=used_threads,
        requested_threads=options.threads,
        logical_cores=logical_cores,
        tile_size=options.tile_size,
        ssao_enabled=ssao_enabled,
        max_memory_bytes=options.max_memory_bytes,
        initial_estimated_bytes=initial_estimate,
        estimated_bytes=estimate,
        degradation_events=tuple(events),
        warnings=tuple(warnings),
    )


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    length = math.sqrt(float(np.sum(vector * vector)))
    if length <= _EPS:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / length


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    lengths = np.sqrt(np.sum(vectors * vectors, axis=-1, keepdims=True))
    return vectors / np.maximum(lengths, _EPS)


@dataclass(frozen=True)
class _CameraBasis:
    position: np.ndarray
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    tan_half_fov: float
    aspect: float


def _camera_basis(scene: Scene, width: int, height: int) -> _CameraBasis:
    position = np.asarray(scene.camera.position, dtype=np.float64)
    target = np.asarray(scene.camera.target, dtype=np.float64)
    up_hint = np.asarray(scene.camera.up, dtype=np.float64)
    forward = _normalize(target - position)
    right = _normalize(np.cross(forward, up_hint))
    up = _normalize(np.cross(right, forward))
    return _CameraBasis(
        position=position,
        forward=forward,
        right=right,
        up=up,
        tan_half_fov=math.tan(math.radians(scene.camera.fov_y_deg) * 0.5),
        aspect=float(width) / float(height),
    )


def _ray_grid(
    basis: _CameraBasis,
    width: int,
    height: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    offset_x: np.ndarray | float,
    offset_y: np.ndarray | float,
) -> np.ndarray:
    xs = np.arange(x0, x1, dtype=np.float64)[None, :] + offset_x
    ys = np.arange(y0, y1, dtype=np.float64)[:, None] + offset_y
    ndc_x = ((xs / float(width)) * 2.0 - 1.0) * basis.aspect * basis.tan_half_fov
    ndc_y = (1.0 - (ys / float(height)) * 2.0) * basis.tan_half_fov
    if not np.isscalar(offset_x) or not np.isscalar(offset_y):
        ndc_x = np.broadcast_to(ndc_x, (y1 - y0, x1 - x0))
        ndc_y = np.broadcast_to(ndc_y, (y1 - y0, x1 - x0))
    directions = (
        basis.forward[None, None, :]
        + ndc_x[..., None] * basis.right[None, None, :]
        + ndc_y[..., None] * basis.up[None, None, :]
    )
    return _normalize_rows(directions)


def _hash01(
    x: np.ndarray | int, y: np.ndarray | int, salt: int
) -> np.ndarray | float:
    """Stable integer hash mapped to [0,1), with no PRNG or clock input."""

    x64 = np.asarray(x, dtype=np.uint64)
    y64 = np.asarray(y, dtype=np.uint64)
    h = x64 * np.uint64(0x9E3779B97F4A7C15)
    h = h ^ (y64 * np.uint64(0xC2B2AE3D27D4EB4F))
    salt_product = (salt * 0x165667B19E3779F9) & 0xFFFFFFFFFFFFFFFF
    h = h ^ np.uint64(salt_product)
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94D049BB133111EB)
    h ^= h >> np.uint64(31)
    result = (h & np.uint64(0x00FFFFFF)).astype(np.float64) / 16777216.0
    if result.ndim == 0:
        return float(result)
    return result


def _sample_offsets(
    samples: int,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    sample_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if samples == 1:
        shape = pixel_x.shape
        return np.full(shape, 0.5), np.full(shape, 0.5)

    grid_width = int(math.ceil(math.sqrt(samples)))
    grid_x = sample_index % grid_width
    grid_y = sample_index // grid_width
    jitter_x = _hash01(pixel_x, pixel_y, 1013 + sample_index * 2)
    jitter_y = _hash01(pixel_x, pixel_y, 2027 + sample_index * 2)
    jitter_x = np.asarray(jitter_x)
    jitter_y = np.asarray(jitter_y)
    offset_x = (grid_x + 0.05 + 0.90 * jitter_x) / grid_width
    offset_y = (grid_y + 0.05 + 0.90 * jitter_y) / grid_width
    return offset_x, offset_y


def _intersect_spheres(
    origin: np.ndarray,
    directions: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    near: float,
    far: float,
) -> tuple[np.ndarray, np.ndarray]:
    shape = directions.shape[:-1]
    best_t = np.full(shape, np.inf, dtype=np.float64)
    best_index = np.full(shape, -1, dtype=np.int32)

    for sphere_index in range(centers.shape[0]):
        oc = origin - centers[sphere_index]
        b = np.sum(directions * oc, axis=-1)
        c = np.sum(oc * oc, axis=-1) - radii[sphere_index] ** 2
        discriminant = b * b - c
        root = np.sqrt(np.maximum(discriminant, 0.0))
        t0 = -b - root
        t1 = -b + root
        t = np.where(t0 > near, t0, np.where(t1 > near, t1, np.inf))
        hit = (discriminant >= 0.0) & (t > near) & (t < far)
        closer = hit & (t < best_t)
        best_t = np.where(closer, t, best_t)
        best_index = np.where(closer, sphere_index, best_index)

    return best_t, best_index


def _compute_gbuffer(
    scene: Scene,
    basis: _CameraBasis,
    width: int,
    height: int,
    centers: np.ndarray,
    radii: np.ndarray,
) -> GBuffer:
    directions = _ray_grid(
        basis, width, height, 0, width, 0, height, 0.5, 0.5
    )
    depth, sphere_index = _intersect_spheres(
        basis.position,
        directions,
        centers,
        radii,
        scene.camera.near,
        scene.camera.far,
    )
    hit = sphere_index >= 0
    points = basis.position[None, None, :] + directions * depth[..., None]
    normals = np.zeros_like(directions)
    for sphere_index_value in range(centers.shape[0]):
        mask = sphere_index == sphere_index_value
        if np.any(mask):
            normals[mask] = (
                points[mask] - centers[sphere_index_value]
            ) / radii[sphere_index_value]
    normals = _normalize_rows(normals)
    depth = np.where(hit, depth, np.inf)
    return GBuffer(
        sphere_index=sphere_index.astype(np.int32, copy=False),
        depth=depth.astype(np.float64, copy=False),
        normal=normals.astype(np.float64, copy=False),
        ray_direction=directions.astype(np.float64, copy=False),
    )


def _compute_ssao(
    scene: Scene,
    gbuffer: GBuffer,
    basis: _CameraBasis,
    enabled: bool,
) -> np.ndarray:
    height, width = gbuffer.depth.shape
    ao = np.ones((height, width), dtype=np.float64)
    if not enabled or scene.ssao_intensity == 0.0:
        return ao

    hit = gbuffer.sphere_index >= 0
    if not np.any(hit):
        return ao

    points = (
        basis.position[None, None, :]
        + gbuffer.ray_direction * gbuffer.depth[..., None]
    )
    points = np.where(hit[..., None], points, 0.0)
    normals = gbuffer.normal
    pixel_x = np.arange(width, dtype=np.uint64)[None, :]
    pixel_y = np.arange(height, dtype=np.uint64)[:, None]
    angle = _hash01(pixel_x, pixel_y, 4099) * (2.0 * math.pi)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    helper = np.zeros_like(normals)
    use_x = np.abs(normals[..., 0]) < 0.9
    helper[use_x, 0] = 1.0
    helper[~use_x, 1] = 1.0
    base_tangent = _normalize_rows(np.cross(helper, normals))
    base_bitangent = np.cross(normals, base_tangent)
    tangent = (
        cos_angle[..., None] * base_tangent
        + sin_angle[..., None] * base_bitangent
    )
    bitangent = (
        -sin_angle[..., None] * base_tangent
        + cos_angle[..., None] * base_bitangent
    )

    sample_count = scene.ssao_samples
    kernel: list[tuple[float, float, float]] = []
    for sample_index in range(sample_count):
        radial = math.sqrt((sample_index + 0.5) / sample_count)
        theta = sample_index * _GOLDEN_ANGLE
        kernel.append(
            (
                math.cos(theta) * radial,
                math.sin(theta) * radial,
                math.sqrt(max(0.0, 1.0 - radial * radial)),
            )
        )

    occlusion_sum = np.zeros((height, width), dtype=np.float64)
    y_coords = np.arange(height, dtype=np.int64)[:, None]
    x_coords = np.arange(width, dtype=np.int64)[None, :]
    radius = scene.ssao_radius
    bias = max(radius * 0.02, 1.0e-4)

    for sample_index, (kernel_x, kernel_y, kernel_z) in enumerate(kernel):
        scale = 0.35 + 0.65 * np.asarray(
            _hash01(pixel_x, pixel_y, 7001 + sample_index)
        )
        sample_points = (
            points
            + tangent * (kernel_x * scale * radius)[..., None]
            + bitangent * (kernel_y * scale * radius)[..., None]
            + normals * (kernel_z * scale * radius)[..., None]
        )
        relative = sample_points - basis.position[None, None, :]
        view_x = np.sum(relative * basis.right, axis=-1)
        view_y = np.sum(relative * basis.up, axis=-1)
        view_z = np.sum(relative * basis.forward, axis=-1)
        safe_view_z = np.where(view_z > 1.0e-6, view_z, 1.0)
        sample_x = np.rint(
            (
                view_x
                / (safe_view_z * basis.tan_half_fov * basis.aspect)
                + 1.0
            )
            * 0.5
            * width
        ).astype(np.int64)
        sample_y = np.rint(
            (
                1.0
                - view_y / (safe_view_z * basis.tan_half_fov)
            )
            * 0.5
            * height
        ).astype(np.int64)
        valid = (
            (sample_x >= 0)
            & (sample_x < width)
            & (sample_y >= 0)
            & (sample_y < height)
            & (view_z > 0.0)
        )
        clipped_x = np.clip(sample_x, 0, width - 1)
        clipped_y = np.clip(sample_y, 0, height - 1)
        neighbor_depth = gbuffer.depth[clipped_y, clipped_x]
        neighbor_index = gbuffer.sphere_index[clipped_y, clipped_x]
        neighbor_ray = gbuffer.ray_direction[clipped_y, clipped_x]
        valid &= (neighbor_index >= 0) & np.isfinite(neighbor_depth)
        safe_neighbor_depth = np.where(valid, neighbor_depth, 0.0)

        neighbor_point = (
            basis.position[None, None, :]
            + neighbor_ray * safe_neighbor_depth[..., None]
        )
        neighbor_relative = neighbor_point - basis.position[None, None, :]
        neighbor_view_z = np.sum(
            neighbor_relative * basis.forward, axis=-1
        )
        base_relative = points - basis.position[None, None, :]
        base_view_z = np.sum(base_relative * basis.forward, axis=-1)
        depth_delta = view_z - neighbor_view_z
        in_range = np.abs(neighbor_view_z - base_view_z) < radius * 2.0
        contribution = np.clip((depth_delta - bias) / radius, 0.0, 1.0)
        contribution = np.where(in_range, contribution, 0.0)
        occlusion_sum += np.where(valid & hit, contribution, 0.0)

    normalized_occlusion = (
        occlusion_sum / float(sample_count) * scene.ssao_intensity
    )
    ao[hit] = np.clip(1.0 - normalized_occlusion[hit], 0.0, 1.0)
    return ao


def _perturb_normal(
    points: np.ndarray,
    geometric_normal: np.ndarray,
    amplitude: np.ndarray,
) -> np.ndarray:
    if np.all(amplitude == 0.0):
        return geometric_normal

    helper = np.zeros_like(geometric_normal)
    use_x = np.abs(geometric_normal[:, 0]) < 0.9
    helper[use_x, 0] = 1.0
    helper[~use_x, 1] = 1.0
    tangent = _normalize_rows(np.cross(helper, geometric_normal))
    bitangent = np.cross(geometric_normal, tangent)

    u = np.sum(points * tangent, axis=-1)
    v = np.sum(points * bitangent, axis=-1)
    phase_u = 10.0 * u + 0.7
    phase_v = 13.0 * v - 0.4
    dh_du = (
        amplitude * 10.0 * np.cos(phase_u) * np.cos(phase_v)
    )
    dh_dv = (
        -amplitude * 13.0 * np.sin(phase_u) * np.sin(phase_v)
    )
    perturbed = (
        geometric_normal
        - dh_du[:, None] * tangent
        - dh_dv[:, None] * bitangent
    )
    return _normalize_rows(perturbed)


def _is_shadowed(
    points: np.ndarray,
    geometric_normal: np.ndarray,
    light_direction: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    origins = points + geometric_normal * _SHADOW_EPS
    directions = np.broadcast_to(light_direction, origins.shape)
    _, sphere_index = _intersect_spheres(
        origins, directions, centers, radii, _SHADOW_EPS, 1.0e9
    )
    return sphere_index >= 0


def _shade_sample(
    scene: Scene,
    basis: _CameraBasis,
    directions: np.ndarray,
    depth: np.ndarray,
    sphere_index: np.ndarray,
    ao: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    base_colors: np.ndarray,
    metallic_values: np.ndarray,
    roughness_values: np.ndarray,
    normal_amplitudes: np.ndarray,
) -> np.ndarray:
    background = np.asarray(scene.background_color, dtype=np.float64)
    output = np.broadcast_to(background, directions.shape).copy()
    light_toward = -_normalize(np.asarray(scene.light.direction, dtype=np.float64))
    light_radiance = (
        np.asarray(scene.light.color, dtype=np.float64) * scene.light.intensity
    )
    ambient_irradiance = (
        np.asarray(scene.ambient_color, dtype=np.float64) * scene.ambient
    )

    for current_index in range(centers.shape[0]):
        mask = sphere_index == current_index
        if not np.any(mask):
            continue

        ray_direction = directions[mask]
        hit_point = basis.position[None, :] + ray_direction * depth[mask, None]
        geometric_normal = _normalize_rows(
            (hit_point - centers[current_index]) / radii[current_index]
        )
        amplitude = normal_amplitudes[current_index]
        shading_normal = _perturb_normal(
            hit_point,
            geometric_normal,
            np.full(hit_point.shape[0], amplitude, dtype=np.float64),
        )

        view = -ray_direction
        albedo = np.broadcast_to(
            base_colors[current_index], (hit_point.shape[0], 3)
        )
        metallic = metallic_values[current_index]
        roughness = roughness_values[current_index]
        f0 = 0.04 * (1.0 - metallic) + albedo * metallic

        shadowed = _is_shadowed(
            hit_point, geometric_normal, light_toward, centers, radii
        )
        n_dot_l = np.clip(np.sum(shading_normal * light_toward, axis=-1), 0.0, 1.0)
        n_dot_v = np.clip(np.sum(shading_normal * view, axis=-1), 1.0e-6, 1.0)
        half_vector = _normalize_rows(view + light_toward)
        n_dot_h = np.clip(np.sum(shading_normal * half_vector, axis=-1), 0.0, 1.0)
        v_dot_h = np.clip(np.sum(view * half_vector, axis=-1), 0.0, 1.0)

        alpha = max(roughness * roughness, 1.0e-4)
        alpha_sq = alpha * alpha
        denominator = n_dot_h * n_dot_h * (alpha_sq - 1.0) + 1.0
        distribution = alpha_sq / (math.pi * denominator * denominator + _EPS)
        k = (roughness + 1.0) * (roughness + 1.0) / 8.0
        geometry_v = n_dot_v / (n_dot_v * (1.0 - k) + k)
        geometry_l = n_dot_l / (n_dot_l * (1.0 - k) + k + _EPS)
        geometry = geometry_v * geometry_l
        fresnel = f0 + (1.0 - f0) * np.power(1.0 - v_dot_h, 5.0)[:, None]
        specular = (
            distribution[:, None]
            * geometry[:, None]
            * fresnel
            / (4.0 * n_dot_v[:, None] * n_dot_l[:, None] + _EPS)
        )
        diffuse = (
            (1.0 - fresnel)
            * (1.0 - metallic)
            * albedo
            / math.pi
        )
        direct = (
            light_radiance[None, :]
            * (n_dot_l * ~shadowed)[:, None]
            * (diffuse + specular)
        )

        ambient_brdf = albedo * (1.0 - metallic) + f0
        ambient = (
            ambient_irradiance[None, :]
            * ambient_brdf
            * ao[mask, None]
        )
        shaded = ambient + direct
        output[mask] = np.maximum(shaded, 0.0)

    return output


def _render_tile(
    scene: Scene,
    basis: _CameraBasis,
    gbuffer_ao: np.ndarray,
    render_width: int,
    render_height: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    samples: int,
    centers: np.ndarray,
    radii: np.ndarray,
    base_colors: np.ndarray,
    metallic_values: np.ndarray,
    roughness_values: np.ndarray,
    normal_amplitudes: np.ndarray,
) -> np.ndarray:
    height = y1 - y0
    width = x1 - x0
    pixel_x = np.arange(x0, x1, dtype=np.uint64)[None, :]
    pixel_y = np.arange(y0, y1, dtype=np.uint64)[:, None]
    pixel_x = np.broadcast_to(pixel_x, (height, width))
    pixel_y = np.broadcast_to(pixel_y, (height, width))
    ao = gbuffer_ao[y0:y1, x0:x1]
    accumulation = np.zeros((height, width, 3), dtype=np.float64)

    for sample_index in range(samples):
        offset_x, offset_y = _sample_offsets(
            samples, pixel_x, pixel_y, sample_index
        )
        directions = _ray_grid(
            basis,
            render_width,
            render_height,
            x0,
            x1,
            y0,
            y1,
            offset_x,
            offset_y,
        )
        depth, sphere_index = _intersect_spheres(
            basis.position,
            directions,
            centers,
            radii,
            scene.camera.near,
            scene.camera.far,
        )
        accumulation += _shade_sample(
            scene,
            basis,
            directions,
            depth,
            sphere_index,
            ao,
            centers,
            radii,
            base_colors,
            metallic_values,
            roughness_values,
            normal_amplitudes,
        )

    accumulation /= float(samples)
    return accumulation.astype(np.float32)


def render(scene: Scene, plan: RenderPlan) -> RenderResult:
    """Render a scene according to an already validated resource plan."""

    width = plan.width
    height = plan.height
    basis = _camera_basis(scene, width, height)
    centers = np.asarray([sphere.center for sphere in scene.spheres], dtype=np.float64)
    radii = np.asarray([sphere.radius for sphere in scene.spheres], dtype=np.float64)
    base_colors = np.asarray(
        [sphere.base_color for sphere in scene.spheres], dtype=np.float64
    )
    metallic_values = np.asarray(
        [sphere.metallic for sphere in scene.spheres], dtype=np.float64
    )
    roughness_values = np.asarray(
        [sphere.roughness for sphere in scene.spheres], dtype=np.float64
    )
    normal_amplitudes = np.asarray(
        [sphere.normal_perturbation for sphere in scene.spheres], dtype=np.float64
    )

    gbuffer = _compute_gbuffer(scene, basis, width, height, centers, radii)
    ao = _compute_ssao(scene, gbuffer, basis, plan.ssao_enabled)

    read_only_arrays = (
        centers,
        radii,
        base_colors,
        metallic_values,
        roughness_values,
        normal_amplitudes,
        gbuffer.sphere_index,
        gbuffer.depth,
        gbuffer.normal,
        gbuffer.ray_direction,
        ao,
    )
    for array in read_only_arrays:
        array.setflags(write=False)

    tile_coordinates = [
        (y0, min(y0 + plan.tile_size, height), x0, min(x0 + plan.tile_size, width))
        for y0 in range(0, height, plan.tile_size)
        for x0 in range(0, width, plan.tile_size)
    ]

    def render_task(
        coordinates: tuple[int, int, int, int],
    ) -> tuple[tuple[int, int, int, int], np.ndarray]:
        y0, y1, x0, x1 = coordinates
        tile = _render_tile(
            scene,
            basis,
            ao,
            width,
            height,
            x0,
            x1,
            y0,
            y1,
            plan.samples,
            centers,
            radii,
            base_colors,
            metallic_values,
            roughness_values,
            normal_amplitudes,
        )
        return coordinates, tile

    linear_hdr = np.empty((height, width, 3), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=plan.threads) as executor:
        # executor.map preserves the fixed row-major tile order. Each worker
        # allocates its own tile and never writes shared scene or output data.
        for (y0, y1, x0, x1), tile in executor.map(
            render_task, tile_coordinates
        ):
            linear_hdr[y0:y1, x0:x1] = tile

    masks = tuple(
        gbuffer.sphere_index == sphere.index for sphere in scene.spheres
    )
    global_stats = summarize_linear(linear_hdr)
    region_stats = tuple(
        summarize_linear(linear_hdr, mask) for mask in masks
    )
    material_regions = tuple(
        build_material_regions(linear_hdr, masks, scene.spheres)
    )
    return RenderResult(
        linear_hdr=linear_hdr,
        sphere_masks=masks,
        sphere_index=gbuffer.sphere_index,
        ambient_occlusion=ao,
        plan=plan,
        global_stats=global_stats,
        region_stats=region_stats,
        material_regions=material_regions,
    )

