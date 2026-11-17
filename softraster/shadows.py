"""Per-light depth maps and deterministic PCF visibility."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math3d import OrthographicCamera, PerspectiveCamera, normalize
from .raster import rasterize_depth_map
from .scene import Light, Scene, ShadowSettings


@dataclass(frozen=True)
class ShadowMap:
    kind: str
    maps: np.ndarray
    cameras: tuple[OrthographicCamera | PerspectiveCamera, ...]
    settings: ShadowSettings
    eye: np.ndarray


_CUBE_FACES: tuple[tuple[np.ndarray, np.ndarray], ...] = (
    (np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0])),
    (np.asarray([-1.0, 0.0, 0.0]), np.asarray([0.0, 1.0, 0.0])),
    (np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0])),
    (np.asarray([0.0, -1.0, 0.0]), np.asarray([0.0, 0.0, 1.0])),
    (np.asarray([0.0, 0.0, 1.0]), np.asarray([0.0, 1.0, 0.0])),
    (np.asarray([0.0, 0.0, -1.0]), np.asarray([0.0, 1.0, 0.0])),
)


def build_shadow_maps(scene: Scene) -> tuple[ShadowMap | None, ...]:
    return tuple(_build_light_shadow(scene, light) for light in scene.lights)


def _build_light_shadow(scene: Scene, light: Light) -> ShadowMap | None:
    settings = light.shadow
    if not settings.enabled:
        return None
    if light.kind == "directional":
        return _build_directional_shadow(scene, light, settings)
    return _build_point_shadow(scene, light, settings)


def _build_directional_shadow(
    scene: Scene, light: Light, settings: ShadowSettings
) -> ShadowMap:
    assert light.direction is not None
    direction = normalize(light.direction)
    up_hint = np.asarray([0.0, 1.0, 0.0])
    if abs(float(np.dot(direction, up_hint))) > 0.98:
        up_hint = np.asarray([1.0, 0.0, 0.0])
    eye = scene.center - direction * (scene.radius * 2.0 + 1.0)
    camera = OrthographicCamera(
        eye=eye,
        forward=direction,
        up_hint=up_hint,
        half_extent=scene.radius * 1.02,
        near=1.0e-4,
        far=scene.radius * 4.0 + 2.0,
        width=settings.map_size,
        height=settings.map_size,
    )
    depth = rasterize_depth_map(camera, scene.meshes)
    return ShadowMap(
        kind="directional",
        maps=depth[None, :, :],
        cameras=(camera,),
        settings=settings,
        eye=eye,
    )


def _build_point_shadow(
    scene: Scene, light: Light, settings: ShadowSettings
) -> ShadowMap:
    assert light.position is not None
    distance_to_center = float(np.linalg.norm(scene.center - light.position))
    far = 2.0 * (distance_to_center + scene.radius) + 1.0
    near = max(1.0e-4, scene.radius * 1.0e-5)
    cameras: list[PerspectiveCamera] = []
    maps: list[np.ndarray] = []
    for forward, up_hint in _CUBE_FACES:
        camera = PerspectiveCamera(
            eye=light.position,
            target=light.position + forward,
            up_hint=up_hint,
            fov_y_radians=np.pi * 0.5,
            near=near,
            far=far,
            width=settings.map_size,
            height=settings.map_size,
        )
        cameras.append(camera)
        maps.append(rasterize_depth_map(camera, scene.meshes))
    return ShadowMap(
        kind="point",
        maps=np.stack(maps, axis=0),
        cameras=tuple(cameras),
        settings=settings,
        eye=light.position,
    )


def compute_visibility(
    light: Light,
    shadow_map: ShadowMap | None,
    points: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """Return one visibility value per receiver point for one light."""
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if shadow_map is None or points.shape[0] == 0:
        return np.ones(points.shape[0], dtype=np.float64)

    if light.kind == "directional":
        assert light.direction is not None
        toward_light = -normalize(light.direction)
        toward_light_array = np.broadcast_to(
            toward_light, points.shape
        ).copy()
    else:
        assert light.position is not None
        delta = light.position - points
        toward_light_array = delta / np.maximum(
            np.linalg.norm(delta, axis=1, keepdims=True), 1.0e-12
        )

    normal_sign = np.where(
        np.sum(normals * toward_light_array, axis=1) >= 0.0, 1.0, -1.0
    )
    ndotl = np.clip(
        np.sum(normals * toward_light_array, axis=1), 0.0, 1.0
    )
    offset_points = (
        points
        + normals
        * (shadow_map.settings.normal_bias * normal_sign)[:, None]
    )

    if shadow_map.kind == "directional":
        camera = shadow_map.cameras[0]
        assert isinstance(camera, OrthographicCamera)
        view = camera.to_view(offset_points)
        uv_x, uv_y, _ = camera.project(view)
        receiver_depth = camera.depth_metric(offset_points)
        face_index = np.zeros(offset_points.shape[0], dtype=np.int64)
        stacked_maps = shadow_map.maps
        texel_world_size = (
            2.0 * camera.half_extent / float(camera.width)
        )
    else:
        assert light.position is not None
        delta = offset_points - light.position
        absolute = np.abs(delta)
        face_index = np.argmax(absolute, axis=1).astype(np.int64)
        forwards = np.stack([camera.forward for camera in shadow_map.cameras])
        rights = np.stack([camera.right for camera in shadow_map.cameras])
        ups = np.stack([camera.up for camera in shadow_map.cameras])
        forward = forwards[face_index]
        right = rights[face_index]
        up = ups[face_index]
        receiver_depth = np.sum(delta * forward, axis=1)
        local_x = np.sum(delta * right, axis=1)
        local_y = np.sum(delta * up, axis=1)
        safe_depth = np.maximum(receiver_depth, 1.0e-12)
        ndc_x = local_x / safe_depth
        ndc_y = local_y / safe_depth
        map_size = shadow_map.settings.map_size
        uv_x = (ndc_x * 0.5 + 0.5) * map_size
        uv_y = (1.0 - (ndc_y * 0.5 + 0.5)) * map_size
        stacked_maps = shadow_map.maps
        texel_world_size = (
            2.0 * np.maximum(receiver_depth, 1.0e-4) / map_size
        )

    offsets = _pcf_offsets(
        shadow_map.settings.radius, shadow_map.settings.samples
    )
    lit_count = np.zeros(points.shape[0], dtype=np.float64)
    slope = np.minimum(
        np.sqrt(np.maximum(1.0 - ndotl * ndotl, 0.0))
        / np.maximum(ndotl, 0.15),
        2.0,
    )
    bias = (
        shadow_map.settings.bias
        + np.asarray(texel_world_size)
        * shadow_map.settings.radius
        * slope
    )
    receiver_threshold = receiver_depth - bias
    for offset_x, offset_y in offsets:
        sampled_depth = _sample_bilinear(
            stacked_maps,
            face_index,
            uv_x + float(offset_x),
            uv_y + float(offset_y),
        )
        lit_count += sampled_depth >= receiver_threshold
    return lit_count / float(len(offsets))


def _pcf_offsets(radius: float, samples: int) -> list[tuple[float, float]]:
    if radius <= 0.0 or samples <= 1:
        return [(0.0, 0.0)]
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    offsets: list[tuple[float, float]] = []
    for index in range(samples):
        angle = index * golden_angle
        ring_radius = radius * np.sqrt((index + 0.5) / samples)
        offsets.append(
            (
                float(np.cos(angle) * ring_radius),
                float(np.sin(angle) * ring_radius),
            )
        )
    return offsets


def _sample_bilinear(
    maps: np.ndarray,
    face_index: np.ndarray,
    uv_x: np.ndarray,
    uv_y: np.ndarray,
) -> np.ndarray:
    _, height, width = maps.shape
    local_x = uv_x - 0.5
    local_y = uv_y - 0.5
    x0_unclamped = np.floor(local_x).astype(np.int64)
    y0_unclamped = np.floor(local_y).astype(np.int64)
    x1_unclamped = x0_unclamped + 1
    y1_unclamped = y0_unclamped + 1
    weight_x = local_x - x0_unclamped
    weight_y = local_y - y0_unclamped

    valid = (
        (x0_unclamped >= 0)
        & (y0_unclamped >= 0)
        & (x1_unclamped < width)
        & (y1_unclamped < height)
    )
    x0 = np.clip(x0_unclamped, 0, width - 1)
    x1 = np.clip(x1_unclamped, 0, width - 1)
    y0 = np.clip(y0_unclamped, 0, height - 1)
    y1 = np.clip(y1_unclamped, 0, height - 1)
    face = np.clip(face_index, 0, maps.shape[0] - 1)
    d00 = maps[face, y0, x0]
    d10 = maps[face, y0, x1]
    d01 = maps[face, y1, x0]
    d11 = maps[face, y1, x1]
    finite = (
        np.isfinite(d00) & np.isfinite(d10) & np.isfinite(d01) & np.isfinite(d11)
    )
    sampled = (
        np.where(finite, d00, 0.0) * (1.0 - weight_x) * (1.0 - weight_y)
        + np.where(finite, d10, 0.0) * weight_x * (1.0 - weight_y)
        + np.where(finite, d01, 0.0) * (1.0 - weight_x) * weight_y
        + np.where(finite, d11, 0.0) * weight_x * weight_y
    )
    sampled = np.where(valid & finite, sampled, np.inf)
    return sampled
