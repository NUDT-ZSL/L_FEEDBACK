from __future__ import annotations

import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from .geometry import ViewCamera, prepare_screen_triangles
from .mathutil import normalize_single
from .raster import rasterize_depth_only
from .scene import Light, Scene


@dataclass
class SpotShadowMap:
    depth: np.ndarray
    camera: ViewCamera


@dataclass
class PointShadowMap:
    faces: List[np.ndarray]
    cameras: List[ViewCamera]


ShadowMap = Tuple[str, object]


def _make_light_camera(
    position: np.ndarray,
    forward: np.ndarray,
    up_hint: np.ndarray,
    resolution: int,
    fov_y_deg: float,
) -> ViewCamera:
    forward = normalize_single(forward)
    right = normalize_single(np.cross(forward, up_hint))
    up = normalize_single(np.cross(right, forward))
    return ViewCamera(
        position=np.asarray(position, dtype=np.float64),
        right=right,
        up=up,
        forward=forward,
        tan_half_fov=float(np.tan(np.deg2rad(fov_y_deg) * 0.5)),
        aspect=1.0,
        width=resolution,
        height=resolution,
        near=0.01,
        far=10000.0,
    )


def _shadow_geometry(
    scene: Scene, camera: ViewCamera, light: Light
) -> List[Dict[str, object]]:
    triangles = prepare_screen_triangles(scene, camera)
    for triangle in triangles:
        triangle["light_position"] = light.position
    return triangles


def _build_spot_map(scene: Scene, light: Light, resolution: int) -> SpotShadowMap:
    outer_degrees = float(np.rad2deg(np.arccos(np.clip(light.outer_cos, -1.0, 1.0))))
    fov = min(max(outer_degrees * 2.0 + 1.0, 2.0), 170.0)
    direction = light.direction if light.direction is not None else np.array([0.0, -1.0, 0.0])
    up_hint = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(direction, up_hint))) > 0.99:
        up_hint = np.array([0.0, 0.0, 1.0])
    camera = _make_light_camera(light.position, direction, up_hint, resolution, fov)
    triangles = _shadow_geometry(scene, camera, light)
    _, radial = rasterize_depth_only(
        triangles, resolution, resolution, camera.far
    )
    return SpotShadowMap(radial.astype(np.float32), camera)


def _build_point_map(scene: Scene, light: Light, resolution: int) -> PointShadowMap:
    face_definitions = [
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([-1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
        (np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])),
        (np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0])),
    ]
    cameras: List[ViewCamera] = []
    maps: List[np.ndarray] = []
    for forward, up_hint in face_definitions:
        camera = _make_light_camera(
            light.position, forward, up_hint, resolution, 90.0
        )
        cameras.append(camera)
        triangles = _shadow_geometry(scene, camera, light)
        _, radial = rasterize_depth_only(
            triangles, resolution, resolution, camera.far
        )
        maps.append(radial.astype(np.float32))
    return PointShadowMap(maps, cameras)


def build_shadow_maps(
    scene: Scene, preset_shadow_resolution: int
) -> Tuple[Dict[int, Optional[ShadowMap]], Dict[str, float]]:
    maps: Dict[int, Optional[ShadowMap]] = {}
    timings: Dict[str, float] = {}
    for light in scene.lights:
        if not light.cast_shadow:
            maps[light.id] = None
            continue
        resolution = light.shadow_resolution or preset_shadow_resolution
        started = time.perf_counter()
        outer_degrees = float(
            np.rad2deg(np.arccos(np.clip(light.outer_cos, -1.0, 1.0)))
        )
        if light.kind == "spot" and outer_degrees <= 84.0:
            maps[light.id] = ("spot", _build_spot_map(scene, light, resolution))
        else:
            maps[light.id] = ("point", _build_point_map(scene, light, resolution))
        timings["light_%d" % light.id] = (time.perf_counter() - started) * 1000.0
    return maps, timings


@lru_cache(maxsize=32)
def pcf_offsets(sample_count: int, radius_pixels: float) -> np.ndarray:
    side = int(round(math.sqrt(sample_count)))
    if side * side != sample_count or side < 1 or side > 16:
        raise ValueError("PCF sample count must be a square number between 1 and 256")
    if side == 1:
        coordinates = np.array([0.0], dtype=np.float64)
    else:
        coordinates = np.linspace(-1.0, 1.0, side, dtype=np.float64) * radius_pixels
    u, v = np.meshgrid(coordinates, coordinates)
    return np.stack([u.reshape(-1), v.reshape(-1)], axis=-1)


def _sample_map(
    depth_map: np.ndarray,
    pixel: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    height, width = depth_map.shape
    x = np.floor(pixel[:, 0:1] + offsets[None, :, 0]).astype(np.int64)
    y = np.floor(pixel[:, 1:2] + offsets[None, :, 1]).astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x_safe = np.clip(x, 0, width - 1)
    y_safe = np.clip(y, 0, height - 1)
    result = depth_map[y_safe, x_safe].astype(np.float64)
    result[~valid] = np.inf
    return result


def _spot_visibility(
    light: Light,
    shadow: SpotShadowMap,
    positions: np.ndarray,
    normals: np.ndarray,
    light_directions: np.ndarray,
    distances: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    footprint = (
        2.0
        * distances
        * shadow.camera.tan_half_fov
        / float(shadow.camera.width)
    )
    ndotl = np.clip(np.sum(normals * light_directions, axis=-1), 0.0, 1.0)
    slope = np.sqrt(np.maximum(1.0 - ndotl * ndotl, 0.0)) / np.maximum(ndotl, 0.2)
    normal_offset = footprint * (0.75 + (1.0 - ndotl))
    sample_positions = positions + normals * normal_offset[:, None]
    delta = sample_positions - light.position
    sample_distances = np.sqrt(np.sum(delta * delta, axis=-1))
    pixel = shadow.camera.view_direction_to_face_uv(delta)
    samples = _sample_map(shadow.depth, pixel, offsets)
    bias = (
        light.shadow_bias * sample_distances
        + footprint * (0.5 + slope)
    )
    bias = np.maximum(bias, 1.0e-4)[..., None]
    return np.mean(samples + bias >= sample_distances[..., None], axis=-1)


def _point_visibility(
    light: Light,
    shadow: PointShadowMap,
    positions: np.ndarray,
    normals: np.ndarray,
    light_directions: np.ndarray,
    distances: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    delta = positions - light.position
    absolute = np.abs(delta)
    dominant_axis = np.argmax(absolute, axis=-1)
    face_indices = np.empty(len(positions), dtype=np.int64)
    x_axis = dominant_axis == 0
    y_axis = dominant_axis == 1
    z_axis = dominant_axis == 2
    face_indices[x_axis] = np.where(delta[x_axis, 0] >= 0.0, 0, 1)
    face_indices[y_axis] = np.where(delta[y_axis, 1] >= 0.0, 2, 3)
    face_indices[z_axis] = np.where(delta[z_axis, 2] >= 0.0, 4, 5)
    visibility = np.zeros(len(positions), dtype=np.float64)
    for face_index in range(6):
        mask = face_indices == face_index
        if not np.any(mask):
            continue
        face_positions = positions[mask]
        face_distances = distances[mask]
        face_normals = normals[mask]
        face_light_directions = light_directions[mask]
        footprint = (
            2.0
            * face_distances
            * shadow.cameras[face_index].tan_half_fov
            / float(shadow.cameras[face_index].width)
        )
        ndotl = np.clip(
            np.sum(face_normals * face_light_directions, axis=-1), 0.0, 1.0
        )
        slope = np.sqrt(np.maximum(1.0 - ndotl * ndotl, 0.0)) / np.maximum(ndotl, 0.2)
        normal_offset = footprint * (0.75 + (1.0 - ndotl))
        sample_positions = face_positions + face_normals * normal_offset[:, None]
        face_delta = sample_positions - light.position
        sample_distances = np.sqrt(np.sum(face_delta * face_delta, axis=-1))
        pixel = shadow.cameras[face_index].view_direction_to_face_uv(face_delta)
        samples = _sample_map(shadow.faces[face_index], pixel, offsets)
        bias = (
            light.shadow_bias * sample_distances
            + footprint * (0.5 + slope)
        )
        bias = np.maximum(bias, 1.0e-4)[..., None]
        visibility[mask] = np.mean(
            samples + bias >= sample_distances[..., None], axis=-1
        )
    return visibility


def light_visibility(
    light: Light,
    shadow_map: Optional[ShadowMap],
    positions: np.ndarray,
    normals: np.ndarray,
    light_directions: np.ndarray,
    distances: np.ndarray,
    pcf_sample_count: int,
    pcf_radius: float = 1.0,
) -> np.ndarray:
    if shadow_map is None or len(positions) == 0:
        return np.ones(len(positions), dtype=np.float64)
    offsets = pcf_offsets(pcf_sample_count, pcf_radius)
    kind, shadow = shadow_map
    if kind == "spot":
        return _spot_visibility(
            light,
            shadow,
            positions,
            normals,
            light_directions,
            distances,
            offsets,
        )
    return _point_visibility(
        light,
        shadow,
        positions,
        normals,
        light_directions,
        distances,
        offsets,
    )
