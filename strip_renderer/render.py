from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .sampling import sample_texture
from .scene import Scene
from .texture import TextureSet


EPSILON = 1.0e-12
NORMAL_ZERO_LENGTH_EPSILON = 1.0e-2


@dataclass(frozen=True)
class RenderBuffers:
    hit_mask: np.ndarray
    world_position: np.ndarray
    world_normal: np.ndarray
    shading_normal: np.ndarray
    uv: np.ndarray


@dataclass(frozen=True)
class RenderResult:
    base_color: np.ndarray
    direct_light: np.ndarray
    ambient_light: np.ndarray
    composite: np.ndarray
    buffers: RenderBuffers
    normal_fallback_count: int
    normal_fallback_breakdown: dict[str, int]
    textures: TextureSet

    @property
    def hit_pixels(self) -> int:
        return int(np.count_nonzero(self.buffers.hit_mask))


@dataclass(frozen=True)
class _CameraBasis:
    forward: np.ndarray
    right: np.ndarray
    up: np.ndarray
    tan_half_fov: float
    aspect: float


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= EPSILON:
        return np.zeros_like(vector)
    return vector / length


def _camera_basis(scene: Scene) -> _CameraBasis:
    position = np.asarray(scene.camera.position, dtype=np.float64)
    target = np.asarray(scene.camera.target, dtype=np.float64)
    up_hint = np.asarray(scene.camera.up, dtype=np.float64)
    forward = _unit(target - position)
    right = _unit(np.cross(forward, up_hint))
    up = _unit(np.cross(right, forward))
    return _CameraBasis(
        forward=forward,
        right=right,
        up=up,
        tan_half_fov=math.tan(
            math.radians(scene.camera.vertical_fov_degrees) * 0.5
        ),
        aspect=scene.output.width / scene.output.height,
    )


def _project_sphere_screen_bounds(
    scene: Scene,
    basis: _CameraBasis,
) -> tuple[int, int, int, int] | None:
    """Return a conservative screen-space AABB for the sphere.

    The analytic intersection is still evaluated per pixel. This bounds the
    screen-space rasterization pass and skips empty tiles.
    """
    width = scene.output.width
    height = scene.output.height
    center = np.asarray(scene.sphere.center, dtype=np.float64)
    radius = scene.sphere.radius
    position = np.asarray(scene.camera.position, dtype=np.float64)
    corners = []
    for x in (-radius, radius):
        for y in (-radius, radius):
            for z in (-radius, radius):
                corners.append(center + np.asarray((x, y, z), dtype=np.float64))

    eye_vectors = [corner - position for corner in corners]
    depths = np.asarray(
        [float(np.dot(vector, basis.forward)) for vector in eye_vectors]
    )
    if np.any(depths <= scene.camera.near + EPSILON):
        return None

    ndc_x = np.asarray(
        [
            float(np.dot(vector, basis.right))
            / (depth * basis.aspect * basis.tan_half_fov)
            for vector, depth in zip(eye_vectors, depths)
        ]
    )
    ndc_y = np.asarray(
        [
            float(np.dot(vector, basis.up))
            / (depth * basis.tan_half_fov)
            for vector, depth in zip(eye_vectors, depths)
        ]
    )
    x0 = int(math.floor((float(np.min(ndc_x)) + 1.0) * 0.5 * width - 0.5)) - 2
    x1 = int(math.ceil((float(np.max(ndc_x)) + 1.0) * 0.5 * width - 0.5)) + 2
    y0 = int(math.floor((1.0 - float(np.max(ndc_y))) * 0.5 * height - 0.5)) - 2
    y1 = int(math.ceil((1.0 - float(np.min(ndc_y))) * 0.5 * height - 0.5)) + 2
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x0 >= x1 or y0 >= y1:
        return x0, x1, y0, y1
    return x0, x1, y0, y1


def _block_ranges(
    length: int,
    block_size: int,
    reverse: bool,
) -> list[tuple[int, int]]:
    ranges = [
        (start, min(start + block_size, length))
        for start in range(0, length, block_size)
    ]
    if reverse:
        ranges.reverse()
    return ranges


def _tile_rays(
    scene: Scene,
    basis: _CameraBasis,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> np.ndarray:
    width = scene.output.width
    height = scene.output.height
    pixel_x = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5
    pixel_y = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5
    ndc_x = pixel_x * (2.0 / width) - 1.0
    ndc_y = 1.0 - pixel_y * (2.0 / height)
    directions = (
        basis.forward[None, None, :]
        + basis.right[None, None, :]
        * (ndc_x[..., None] * basis.aspect * basis.tan_half_fov)
        + basis.up[None, None, :] * (ndc_y[..., None] * basis.tan_half_fov)
    )
    lengths = np.linalg.norm(directions, axis=-1, keepdims=True)
    directions = directions / lengths
    return directions


def _intersect_sphere(
    scene: Scene,
    origins: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(scene.sphere.center, dtype=np.float64)
    radius = scene.sphere.radius
    offset = origins - center[None, None, :]
    a = np.sum(directions * directions, axis=-1)
    b = 2.0 * np.sum(offset * directions, axis=-1)
    c = np.sum(offset * offset, axis=-1) - radius * radius
    discriminant = b * b - 4.0 * a * c
    valid_discriminant = discriminant >= 0.0
    root = np.sqrt(np.maximum(discriminant, 0.0))
    t0 = (-b - root) / (2.0 * a)
    t1 = (-b + root) / (2.0 * a)
    valid0 = (
        valid_discriminant
        & (t0 > scene.camera.near)
        & (t0 < scene.camera.far)
    )
    valid1 = (
        valid_discriminant
        & (t1 > scene.camera.near)
        & (t1 < scene.camera.far)
    )
    distance = np.where(valid0, t0, np.where(valid1, t1, np.inf))
    hit = np.isfinite(distance)
    return hit, distance


def _sphere_uv(normals: np.ndarray) -> np.ndarray:
    nx = normals[..., 0]
    ny = np.clip(normals[..., 1], -1.0, 1.0)
    nz = normals[..., 2]
    u = 0.5 + np.arctan2(nx, nz) / (2.0 * math.pi)
    v = np.arccos(ny) / math.pi
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    return np.stack((u, v), axis=-1)


def _tangent_frame(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    world_up = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
    tangent = np.cross(world_up[None, None, :], normals)
    lengths = np.linalg.norm(tangent, axis=-1)
    degenerate = lengths <= EPSILON
    tangent[degenerate] = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    tangent = tangent - normals * np.sum(tangent * normals, axis=-1, keepdims=True)
    tangent = tangent / np.maximum(
        np.linalg.norm(tangent, axis=-1, keepdims=True),
        EPSILON,
    )
    bitangent = np.cross(tangent, normals)
    bitangent = bitangent / np.maximum(
        np.linalg.norm(bitangent, axis=-1, keepdims=True),
        EPSILON,
    )
    return tangent, bitangent


def _perturb_normals(
    scene: Scene,
    textures: TextureSet,
    geometric_normals: np.ndarray,
    uv: np.ndarray,
    hit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    fallback = np.zeros(hit.shape, dtype=bool)
    breakdown = {
        "missing_texture": 0,
        "zero_length": 0,
        "invalid_result": 0,
    }
    if textures.normal is None:
        fallback[:] = hit
        breakdown["missing_texture"] = int(np.count_nonzero(hit))
        return geometric_normals.copy(), fallback, breakdown

    encoded = sample_texture(
        textures.normal,
        uv[..., 0],
        uv[..., 1],
        scene.output.texture_filter,
    )
    tangent_normal = np.clip(encoded * 2.0 - 1.0, -1.0, 1.0)
    local_length = np.linalg.norm(tangent_normal, axis=-1)
    zero_length = hit & (local_length <= NORMAL_ZERO_LENGTH_EPSILON)
    fallback |= zero_length
    breakdown["zero_length"] = int(np.count_nonzero(zero_length))

    safe_length = np.maximum(local_length, EPSILON)
    tangent_normal = tangent_normal / safe_length[..., None]
    tangent_normal[..., 0:2] *= scene.material.normal_strength
    tangent, bitangent = _tangent_frame(geometric_normals)
    perturbed = (
        tangent * tangent_normal[..., 0:1]
        + bitangent * tangent_normal[..., 1:2]
        + geometric_normals * tangent_normal[..., 2:3]
    )
    perturbed_length = np.linalg.norm(perturbed, axis=-1)
    invalid = hit & ~zero_length & (
        (perturbed_length <= EPSILON)
        | ~np.isfinite(perturbed_length)
        | ~np.all(np.isfinite(perturbed), axis=-1)
    )
    fallback |= invalid
    breakdown["invalid_result"] = int(np.count_nonzero(invalid))
    perturbed = perturbed / np.maximum(
        perturbed_length[..., None],
        EPSILON,
    )
    perturbed = np.where(fallback[..., None], geometric_normals, perturbed)
    return perturbed, fallback, breakdown


def _shade_tile(
    scene: Scene,
    textures: TextureSet,
    world_position: np.ndarray,
    world_normal: np.ndarray,
    uv: np.ndarray,
    hit: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int],
]:
    base_sample = sample_texture(
        textures.base_color,
        uv[..., 0],
        uv[..., 1],
        scene.output.texture_filter,
    )
    tint = np.asarray(scene.material.base_color_tint, dtype=np.float64)
    albedo = np.clip(base_sample * tint[None, None, :], 0.0, 1.0)

    roughness_sample = sample_texture(
        textures.roughness,
        uv[..., 0],
        uv[..., 1],
        scene.output.texture_filter,
    )
    roughness = np.clip(roughness_sample, 0.04, 1.0)

    shaded_normal, fallback, fallback_breakdown = _perturb_normals(
        scene,
        textures,
        world_normal,
        uv,
        hit,
    )
    light_direction = np.asarray(scene.light.direction, dtype=np.float64)
    light_color = np.asarray(scene.light.color, dtype=np.float64)
    camera_position = np.asarray(scene.camera.position, dtype=np.float64)
    view_direction = camera_position[None, None, :] - world_position
    view_direction = view_direction / np.maximum(
        np.linalg.norm(view_direction, axis=-1, keepdims=True),
        EPSILON,
    )

    n_dot_l = np.maximum(np.sum(shaded_normal * light_direction, axis=-1), 0.0)
    n_dot_v = np.maximum(np.sum(shaded_normal * view_direction, axis=-1), 0.0)
    half_vector = light_direction[None, None, :] + view_direction
    half_length = np.linalg.norm(half_vector, axis=-1, keepdims=True)
    half_vector = half_vector / np.maximum(half_length, EPSILON)
    n_dot_h = np.maximum(np.sum(shaded_normal * half_vector, axis=-1), 0.0)

    shininess = np.clip(2.0 / np.maximum(roughness**4, 1.0e-6) - 2.0, 2.0, 2048.0)
    specular_weight = (
        scene.material.specular_strength
        * (1.0 - roughness) ** 2
        * np.where(n_dot_v > 0.0, 1.0, 0.0)
    )
    specular = specular_weight * np.power(n_dot_h, shininess) * n_dot_l
    diffuse = n_dot_l / math.pi
    direct_lighting = scene.light.intensity * (diffuse + specular)
    direct = np.clip(
        albedo * light_color[None, None, :] * direct_lighting[..., None],
        0.0,
        1.0,
    )

    sky = np.asarray(scene.environment.sky_color, dtype=np.float64)
    ground = np.asarray(scene.environment.ground_color, dtype=np.float64)
    normal_y = shaded_normal[..., 1]
    ambient_radiance = scene.environment.intensity * (
        0.5 * (sky + ground)[None, None, :]
        + (1.0 / 3.0)
        * (sky - ground)[None, None, :]
        * normal_y[..., None]
    )
    ambient = np.clip(albedo * ambient_radiance, 0.0, 1.0)

    return albedo, direct, ambient, shaded_normal, fallback, fallback_breakdown


def render_scene(scene: Scene, textures: TextureSet) -> RenderResult:
    width = scene.output.width
    height = scene.output.height
    background = np.asarray(scene.output.background_color, dtype=np.float64)

    base = np.broadcast_to(background, (height, width, 3)).copy()
    direct = base.copy()
    ambient = base.copy()
    hit_mask = np.zeros((height, width), dtype=bool)
    world_position = np.zeros((height, width, 3), dtype=np.float64)
    world_normal = np.zeros((height, width, 3), dtype=np.float64)
    shading_normal = np.zeros((height, width, 3), dtype=np.float64)
    uv = np.zeros((height, width, 2), dtype=np.float64)
    fallback_mask = np.zeros((height, width), dtype=bool)
    fallback_breakdown = {
        "missing_texture": 0,
        "zero_length": 0,
        "invalid_result": 0,
    }

    basis = _camera_basis(scene)
    screen_bounds = _project_sphere_screen_bounds(scene, basis)
    reverse = scene.output.block_order == "reverse"
    if scene.output.block_axis == "row":
        row_ranges = _block_ranges(height, scene.output.block_size, reverse)
        column_ranges = [(0, width)]
    else:
        row_ranges = [(0, height)]
        column_ranges = _block_ranges(width, scene.output.block_size, reverse)

    camera_position = np.asarray(scene.camera.position, dtype=np.float64)
    for y0, y1 in row_ranges:
        for x0, x1 in column_ranges:
            tile_x0, tile_x1, tile_y0, tile_y1 = x0, x1, y0, y1
            if screen_bounds is not None:
                sx0, sx1, sy0, sy1 = screen_bounds
                tile_x0 = max(tile_x0, sx0)
                tile_x1 = min(tile_x1, sx1)
                tile_y0 = max(tile_y0, sy0)
                tile_y1 = min(tile_y1, sy1)
            if tile_x0 >= tile_x1 or tile_y0 >= tile_y1:
                continue

            directions = _tile_rays(
                scene,
                basis,
                tile_x0,
                tile_x1,
                tile_y0,
                tile_y1,
            )
            origins = np.broadcast_to(
                camera_position,
                directions.shape,
            )
            hit, distance = _intersect_sphere(scene, origins, directions)
            if not np.any(hit):
                continue

            safe_distance = np.where(hit, distance, 0.0)
            positions = origins + directions * safe_distance[..., None]
            normals = positions - np.asarray(
                scene.sphere.center,
                dtype=np.float64,
            )[None, None, :]
            normals = normals / np.maximum(
                np.linalg.norm(normals, axis=-1, keepdims=True),
                EPSILON,
            )
            tile_uv = _sphere_uv(normals)
            (
                shaded_base,
                shade_direct,
                shade_ambient,
                tile_shading_normal,
                fallback,
                breakdown,
            ) = _shade_tile(
                scene,
                textures,
                positions,
                normals,
                tile_uv,
                hit,
            )

            tile_slice = np.s_[tile_y0:tile_y1, tile_x0:tile_x1]
            hit_mask[tile_slice] = hit
            world_position[tile_slice] = np.where(
                hit[..., None],
                positions,
                0.0,
            )
            world_normal[tile_slice] = np.where(
                hit[..., None],
                normals,
                0.0,
            )
            shading_normal[tile_slice] = np.where(
                hit[..., None],
                tile_shading_normal,
                0.0,
            )
            uv[tile_slice] = np.where(hit[..., None], tile_uv, 0.0)
            base[tile_slice] = np.where(hit[..., None], shaded_base, background)
            direct[tile_slice] = np.where(hit[..., None], shade_direct, background)
            ambient[tile_slice] = np.where(hit[..., None], shade_ambient, background)
            fallback_mask[tile_slice] = fallback
            for key, value in breakdown.items():
                fallback_breakdown[key] += value

    base = np.clip(base, 0.0, 1.0)
    direct = np.clip(direct, 0.0, 1.0)
    ambient = np.clip(ambient, 0.0, 1.0)
    weights = scene.output.weights
    composite = (
        base * weights.base_color
        + direct * weights.direct
        + ambient * weights.ambient
    )

    return RenderResult(
        base_color=base,
        direct_light=direct,
        ambient_light=ambient,
        composite=composite,
        buffers=RenderBuffers(
            hit_mask=hit_mask,
            world_position=world_position,
            world_normal=world_normal,
            shading_normal=shading_normal,
            uv=uv,
        ),
        normal_fallback_count=int(np.count_nonzero(fallback_mask)),
        normal_fallback_breakdown=fallback_breakdown,
        textures=textures,
    )
