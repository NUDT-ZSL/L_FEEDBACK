from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .linear import (
    look_at,
    normalize,
    orthographic,
    perspective,
    transform_points,
)
from .raster import TriangleBatch, camera_matrices, clip_near_far
from .scene import Light, Scene, ShadowSettings


@dataclass
class ShadowMap:
    name: str
    light_id: str
    kind: str
    depth: np.ndarray
    view_projection: np.ndarray
    near: float
    far: float
    split_near: Optional[float] = None
    split_far: Optional[float] = None
    face: Optional[str] = None

    @property
    def resolution(self) -> int:
        return int(self.depth.shape[0])

    @property
    def filename(self) -> str:
        return self.name + ".png"

    def report_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "file": self.filename,
            "resolution": self.resolution,
            "near": self.near,
            "far": self.far,
        }
        if self.split_near is not None:
            result["split_near"] = self.split_near
            result["split_far"] = self.split_far
        if self.face is not None:
            result["face"] = self.face
        return result


@dataclass
class DirectionalShadow:
    light_id: str
    cascades: List[ShadowMap]
    splits: List[float]

    def report_dict(self) -> Dict[str, Any]:
        return {
            "cascade_count": len(self.cascades),
            "cascade_split_distances": list(self.splits),
            "cascades": [cascade.report_dict() for cascade in self.cascades],
        }


@dataclass
class PointShadow:
    light_id: str
    position: np.ndarray
    faces: Dict[str, ShadowMap]
    near: float
    far: float

    def report_dict(self) -> Dict[str, Any]:
        return {
            "near": self.near,
            "far": self.far,
            "faces": [self.faces[key].report_dict() for key in _POINT_FACES],
        }


@dataclass
class SpotShadow:
    light_id: str
    shadow_map: ShadowMap
    fov_y_deg: float

    def report_dict(self) -> Dict[str, Any]:
        result = self.shadow_map.report_dict()
        result["cone_fov_y_deg"] = self.fov_y_deg
        return result


@dataclass
class ShadowCollection:
    settings: ShadowSettings
    directional: Dict[str, DirectionalShadow] = field(default_factory=dict)
    point: Dict[str, PointShadow] = field(default_factory=dict)
    spot: Dict[str, SpotShadow] = field(default_factory=dict)
    maps: List[ShadowMap] = field(default_factory=list)
    rendered_triangles: int = 0

    def report_dict(self) -> Dict[str, Any]:
        return {
            "resolution": self.settings.resolution,
            "depth_bias": self.settings.depth_bias,
            "slope_scale": self.settings.slope_scale,
            "compare_tolerance": self.settings.compare_tolerance,
            "pcf_radius": self.settings.pcf_radius,
            "rendered_triangles": self.rendered_triangles,
            "directional": {
                key: value.report_dict() for key, value in self.directional.items()
            },
            "point": {key: value.report_dict() for key, value in self.point.items()},
            "spot": {key: value.report_dict() for key, value in self.spot.items()},
            "map_count": len(self.maps),
        }

    def visibility(
        self,
        light: Light,
        positions: np.ndarray,
        camera_view: np.ndarray,
    ) -> np.ndarray:
        if light.light_type == "directional":
            return _query_directional(
                self.directional[light.light_id],
                positions,
                camera_view,
                self.settings.compare_tolerance,
                self.settings.pcf_radius,
            )
        if light.light_type == "point":
            return _query_point(
                self.point[light.light_id],
                positions,
                self.settings.compare_tolerance,
                self.settings.pcf_radius,
            )
        if light.light_type == "spot":
            return _query_single_map(
                self.spot[light.light_id].shadow_map,
                positions,
                self.settings.compare_tolerance,
                self.settings.pcf_radius,
            )
        raise ValueError("unsupported light type: %s" % light.light_type)


_POINT_FACES = ["+x", "-x", "+y", "-y", "+z", "-z"]

_POINT_FACE_DIRECTIONS = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}

_POINT_FACE_UPS = {
    "+x": np.array([0.0, 1.0, 0.0]),
    "-x": np.array([0.0, 1.0, 0.0]),
    "+y": np.array([0.0, 0.0, -1.0]),
    "-y": np.array([0.0, 0.0, 1.0]),
    "+z": np.array([0.0, 1.0, 0.0]),
    "-z": np.array([0.0, 1.0, 0.0]),
}

_POINT_FACE_LABELS = {
    "+x": "posX",
    "-x": "negX",
    "+y": "posY",
    "-y": "negY",
    "+z": "posZ",
    "-z": "negZ",
}


def _render_depth_map(
    batch: TriangleBatch,
    cast_mask: np.ndarray,
    view_projection: np.ndarray,
    resolution: int,
    depth_bias: float,
    slope_scale: float,
    cull_eye: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    depth_buffer = np.ones((resolution, resolution), dtype=np.float32)
    rendered_triangles = 0
    if batch.triangle_count == 0:
        return depth_buffer, 0

    selected_indices = np.nonzero(cast_mask)[0]
    for triangle_index in selected_indices:
        world = batch.positions[triangle_index]
        if cull_eye is not None:
            face_normal = np.cross(world[1] - world[0], world[2] - world[0])
            if np.linalg.norm(face_normal) < 1.0e-12:
                continue
            if float(np.dot(face_normal, cull_eye - world[0])) <= 0.0:
                continue
        homogeneous = np.concatenate(
            [world, np.ones((3, 1), dtype=np.float64)], axis=1
        )
        clip = homogeneous @ view_projection.T
        inside_depth = np.all(
            (clip[:, 3] > 1.0e-12)
            & (clip[:, 2] + clip[:, 3] >= -1.0e-12)
            & (clip[:, 3] - clip[:, 2] >= -1.0e-12)
        )
        if inside_depth:
            clipped_clip = clip
        else:
            dummy_attributes = np.zeros((3, 1), dtype=np.float64)
            clipped_clip, _ = clip_near_far(clip, dummy_attributes)
        if clipped_clip.shape[0] < 3:
            continue
        produced = False
        for fan_index in range(1, clipped_clip.shape[0] - 1):
            fan_clip = clipped_clip[[0, fan_index, fan_index + 1]]
            inverse_w = 1.0 / fan_clip[:, 3]
            ndc = fan_clip[:, :3] * inverse_w[:, None]
            screen_x = (ndc[:, 0] * 0.5 + 0.5) * resolution
            screen_y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * resolution
            screen_depth = ndc[:, 2] * 0.5 + 0.5

            minimum_x = float(np.min(screen_x))
            maximum_x = float(np.max(screen_x))
            minimum_y = float(np.min(screen_y))
            maximum_y = float(np.max(screen_y))
            if (
                maximum_x < 0.0
                or maximum_y < 0.0
                or minimum_x > resolution
                or minimum_y > resolution
            ):
                continue
            denominator = (
                (screen_y[1] - screen_y[2]) * (screen_x[0] - screen_x[2])
                + (screen_x[2] - screen_x[1]) * (screen_y[0] - screen_y[2])
            )
            if abs(denominator) < 1.0e-12:
                continue
            if denominator < 0.0:
                screen_x = screen_x[[0, 2, 1]]
                screen_y = screen_y[[0, 2, 1]]
                screen_depth = screen_depth[[0, 2, 1]]
                denominator = -denominator

            gradient_x = (
                (screen_depth[1] - screen_depth[0]) * (screen_y[2] - screen_y[0])
                - (screen_depth[2] - screen_depth[0]) * (screen_y[1] - screen_y[0])
            ) / denominator
            gradient_y = (
                (screen_x[1] - screen_x[0]) * (screen_depth[2] - screen_depth[0])
                - (screen_x[2] - screen_x[0]) * (screen_depth[1] - screen_depth[0])
            ) / denominator
            slope = max(abs(float(gradient_x)), abs(float(gradient_y)))
            bias = depth_bias + slope_scale * slope

            xmin = max(0, int(math.floor(float(np.min(screen_x))) - 1))
            xmax = min(resolution - 1, int(math.ceil(float(np.max(screen_x))) + 1))
            ymin = max(0, int(math.floor(float(np.min(screen_y))) - 1))
            ymax = min(resolution - 1, int(math.ceil(float(np.max(screen_y))) + 1))
            if xmin > xmax or ymin > ymax:
                continue
            pixel_x = np.arange(xmin, xmax + 1, dtype=np.float64)[None, :] + 0.5
            pixel_y = np.arange(ymin, ymax + 1, dtype=np.float64)[:, None] + 0.5
            b0 = (
                (screen_y[1] - screen_y[2]) * (pixel_x - screen_x[2])
                + (screen_x[2] - screen_x[1]) * (pixel_y - screen_y[2])
            ) / denominator
            b1 = (
                (screen_y[2] - screen_y[0]) * (pixel_x - screen_x[2])
                + (screen_x[0] - screen_x[2]) * (pixel_y - screen_y[2])
            ) / denominator
            b2 = 1.0 - b0 - b1
            candidate = (b0 >= -1.0e-9) & (b1 >= -1.0e-9) & (b2 >= -1.0e-9)
            if not np.any(candidate):
                continue
            produced = True
            pixel_depth = (
                b0 * screen_depth[0] + b1 * screen_depth[1] + b2 * screen_depth[2]
            )
            biased_depth = np.minimum(pixel_depth + bias, 1.0)
            region = depth_buffer[ymin:ymax + 1, xmin:xmax + 1]
            update = candidate & (biased_depth < region)
            region[update] = biased_depth[update].astype(np.float32)
        if produced:
            rendered_triangles += 1
    return depth_buffer, rendered_triangles


def _camera_frustum_corners(scene: Scene, near: float, far: float) -> np.ndarray:
    camera = scene.camera
    forward = normalize(camera.target - camera.position)
    right = normalize(np.cross(forward, camera.up))
    upward = np.cross(right, forward)
    tangent_y = math.tan(math.radians(camera.fov_y_deg) * 0.5)
    tangent_x = tangent_y * (scene.width / float(scene.height))
    corners: List[np.ndarray] = []
    for distance in (near, far):
        center = camera.position + forward * distance
        half_x = right * (distance * tangent_x)
        half_y = upward * (distance * tangent_y)
        corners.extend(
            [
                center - half_x - half_y,
                center + half_x - half_y,
                center + half_x + half_y,
                center - half_x + half_y,
            ]
        )
    return np.asarray(corners)


def _cascade_splits(scene: Scene, settings: ShadowSettings) -> List[float]:
    near = scene.camera.near
    far = settings.shadow_distance
    splits = [near]
    for index in range(1, settings.cascade_count + 1):
        fraction = index / float(settings.cascade_count)
        logarithmic = near * ((far / near) ** fraction)
        uniform = near + (far - near) * fraction
        distance = (
            settings.split_lambda * logarithmic
            + (1.0 - settings.split_lambda) * uniform
        )
        splits.append(float(distance))
    splits[-1] = far
    return splits


def _build_directional_cascade(
    light: Light,
    batch: TriangleBatch,
    cast_mask: np.ndarray,
    scene: Scene,
    settings: ShadowSettings,
    cascade_index: int,
    split_near: float,
    split_far: float,
) -> ShadowMap:
    corners = _camera_frustum_corners(scene, split_near, split_far)
    center = np.mean(corners, axis=0)
    radius = float(np.max(np.linalg.norm(corners - center, axis=1)))
    direction = normalize(light.direction)
    eye = center - direction * (radius * 2.0 + 1.0)
    view = look_at(eye, center, scene.camera.up)
    view_corners = transform_points(view, corners)
    minimum = np.min(view_corners, axis=0)
    maximum = np.max(view_corners, axis=0)
    resolution = settings.resolution
    extent = max(float(maximum[0] - minimum[0]), float(maximum[1] - minimum[1])) * 0.5
    extent = max(extent * 1.02, 1.0e-4)
    center_x = float((minimum[0] + maximum[0]) * 0.5)
    center_y = float((minimum[1] + maximum[1]) * 0.5)
    texel_size = (extent * 2.0) / resolution
    center_x = round(center_x / texel_size) * texel_size
    center_y = round(center_y / texel_size) * texel_size
    extent = max(
        maximum[0] - center_x,
        center_x - minimum[0],
        maximum[1] - center_y,
        center_y - minimum[1],
    )
    extent = math.ceil(extent / texel_size) * texel_size
    z_padding = max(0.01, radius * 0.01)
    near = max(0.01, float(-maximum[2]) - z_padding)
    far = max(near + 0.01, float(-minimum[2]) + z_padding)
    projection = orthographic(
        center_x - extent,
        center_x + extent,
        center_y - extent,
        center_y + extent,
        near,
        far,
    )
    view_projection = projection @ view
    depth, rendered = _render_depth_map(
        batch,
        cast_mask,
        view_projection,
        resolution,
        settings.depth_bias,
        settings.slope_scale,
        cull_eye=eye,
    )
    cascade = ShadowMap(
        name="directional_%s_cascade_%d" % (light.light_id, cascade_index),
        light_id=light.light_id,
        kind="directional_cascade",
        depth=depth,
        view_projection=view_projection,
        near=near,
        far=far,
        split_near=split_near,
        split_far=split_far,
    )
    cascade.rendered_triangles = rendered  # type: ignore[attr-defined]
    return cascade


def _build_point_shadow(
    light: Light,
    batch: TriangleBatch,
    cast_mask: np.ndarray,
    settings: ShadowSettings,
) -> PointShadow:
    near = max(1.0e-3, min(0.05, light.range * 0.001))
    far = float(light.range)
    projection = perspective(90.0, 1.0, near, far)
    faces: Dict[str, ShadowMap] = {}
    total_rendered = 0
    for face_name in _POINT_FACES:
        direction = _POINT_FACE_DIRECTIONS[face_name]
        view = look_at(
            light.position,
            light.position + direction,
            _POINT_FACE_UPS[face_name],
        )
        view_projection = projection @ view
        depth, rendered = _render_depth_map(
            batch,
            cast_mask,
            view_projection,
            settings.resolution,
            settings.depth_bias,
            settings.slope_scale,
            cull_eye=light.position,
        )
        total_rendered += rendered
        faces[face_name] = ShadowMap(
            name="point_%s_%s" % (light.light_id, _POINT_FACE_LABELS[face_name]),
            light_id=light.light_id,
            kind="point_face",
            depth=depth,
            view_projection=view_projection,
            near=near,
            far=far,
            face=face_name,
        )
        faces[face_name].rendered_triangles = rendered  # type: ignore[attr-defined]
    shadow = PointShadow(
        light_id=light.light_id,
        position=light.position.copy(),
        faces=faces,
        near=near,
        far=far,
    )
    shadow.rendered_triangles = total_rendered  # type: ignore[attr-defined]
    return shadow


def _build_spot_shadow(
    light: Light,
    batch: TriangleBatch,
    cast_mask: np.ndarray,
    settings: ShadowSettings,
) -> SpotShadow:
    near = max(1.0e-3, min(0.05, light.range * 0.001))
    far = float(light.range)
    fov = min(170.0, light.outer_angle_deg * 2.0 + 2.0)
    # The light direction is intentionally not normalized in scene.Light.report_dict;
    # scene parsing stores a normalized copy in the dataclass.
    up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(normalize(light.direction), up))) > 0.98:
        up = np.array([0.0, 0.0, 1.0])
    view = look_at(light.position, light.position + normalize(light.direction), up)
    projection = perspective(fov, 1.0, near, far)
    view_projection = projection @ view
    depth, rendered = _render_depth_map(
        batch,
        cast_mask,
        view_projection,
        settings.resolution,
        settings.depth_bias,
        settings.slope_scale,
        cull_eye=light.position,
    )
    shadow_map = ShadowMap(
        name="spot_%s" % light.light_id,
        light_id=light.light_id,
        kind="spot",
        depth=depth,
        view_projection=view_projection,
        near=near,
        far=far,
    )
    shadow_map.rendered_triangles = rendered  # type: ignore[attr-defined]
    return SpotShadow(light_id=light.light_id, shadow_map=shadow_map, fov_y_deg=fov)


def build_shadow_maps(scene: Scene, batch: TriangleBatch) -> ShadowCollection:
    settings = scene.shadows
    cast_mask = batch.cast_shadow
    collection = ShadowCollection(settings=settings)
    splits = _cascade_splits(scene, settings)
    total_rendered = 0
    for light in scene.lights:
        if light.light_type == "directional":
            cascades: List[ShadowMap] = []
            for index in range(settings.cascade_count):
                cascade = _build_directional_cascade(
                    light,
                    batch,
                    cast_mask,
                    scene,
                    settings,
                    index,
                    splits[index],
                    splits[index + 1],
                )
                cascades.append(cascade)
                collection.maps.append(cascade)
                total_rendered += int(getattr(cascade, "rendered_triangles", 0))
            collection.directional[light.light_id] = DirectionalShadow(
                light_id=light.light_id,
                cascades=cascades,
                splits=splits[1:],
            )
        elif light.light_type == "point":
            point_shadow = _build_point_shadow(light, batch, cast_mask, settings)
            collection.point[light.light_id] = point_shadow
            for face_name in _POINT_FACES:
                collection.maps.append(point_shadow.faces[face_name])
            total_rendered += int(getattr(point_shadow, "rendered_triangles", 0))
        elif light.light_type == "spot":
            spot_shadow = _build_spot_shadow(light, batch, cast_mask, settings)
            collection.spot[light.light_id] = spot_shadow
            collection.maps.append(spot_shadow.shadow_map)
            total_rendered += int(
                getattr(spot_shadow.shadow_map, "rendered_triangles", 0)
            )
    collection.rendered_triangles = total_rendered
    return collection


def _query_single_map(
    shadow_map: ShadowMap,
    positions: np.ndarray,
    tolerance: float,
    pcf_radius: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.size == 0:
        return np.ones((0,), dtype=np.float32)
    homogeneous = np.concatenate(
        [positions, np.ones((positions.shape[0], 1), dtype=np.float64)], axis=1
    )
    clip = homogeneous @ shadow_map.view_projection.T
    w = clip[:, 3]
    visible = np.ones((positions.shape[0],), dtype=np.float64)
    valid = w > 1.0e-9
    if not np.any(valid):
        return visible.astype(np.float32)
    ndc = clip[valid, :3] / w[valid, None]
    u = ndc[:, 0] * 0.5 + 0.5
    v = 1.0 - (ndc[:, 1] * 0.5 + 0.5)
    receiver_depth = ndc[:, 2] * 0.5 + 0.5
    inside = (
        (u >= 0.0)
        & (u <= 1.0)
        & (v >= 0.0)
        & (v <= 1.0)
        & (receiver_depth >= 0.0)
        & (receiver_depth <= 1.0)
    )
    if not np.any(inside):
        return visible.astype(np.float32)

    u = u[inside]
    v = v[inside]
    receiver_depth = receiver_depth[inside]
    resolution = shadow_map.resolution
    ix = np.clip(np.floor(u * resolution).astype(np.int64), 0, resolution - 1)
    iy = np.clip(np.floor(v * resolution).astype(np.int64), 0, resolution - 1)
    depth_map = shadow_map.depth
    sample_x_plus = np.clip(ix + 1, 0, resolution - 1)
    sample_x_minus = np.clip(ix - 1, 0, resolution - 1)
    sample_y_plus = np.clip(iy + 1, 0, resolution - 1)
    sample_y_minus = np.clip(iy - 1, 0, resolution - 1)
    gradient_x = np.clip(
        0.5
        * (
            depth_map[iy, sample_x_plus].astype(np.float64)
            - depth_map[iy, sample_x_minus].astype(np.float64)
        ),
        -0.25,
        0.25,
    )
    gradient_y = np.clip(
        0.5
        * (
            depth_map[sample_y_plus, ix].astype(np.float64)
            - depth_map[sample_y_minus, ix].astype(np.float64)
        ),
        -0.25,
        0.25,
    )
    visibility_sum = np.zeros_like(receiver_depth)
    samples = 0
    for offset_y in range(-pcf_radius, pcf_radius + 1):
        sample_y = np.clip(iy + offset_y, 0, resolution - 1)
        for offset_x in range(-pcf_radius, pcf_radius + 1):
            sample_x = np.clip(ix + offset_x, 0, resolution - 1)
            stored_depth = depth_map[sample_y, sample_x].astype(np.float64)
            slope_adjusted_depth = (
                receiver_depth + gradient_x * offset_x + gradient_y * offset_y
            )
            visibility_sum += slope_adjusted_depth <= stored_depth + tolerance
            samples += 1
    sampled_visibility = visibility_sum / float(max(samples, 1))
    temporary = np.ones((positions.shape[0],), dtype=np.float64)
    valid_indices = np.nonzero(valid)[0]
    inside_indices = valid_indices[inside]
    temporary[inside_indices] = sampled_visibility
    return temporary.astype(np.float32)


def _query_directional(
    shadow: DirectionalShadow,
    positions: np.ndarray,
    camera_view: np.ndarray,
    tolerance: float,
    pcf_radius: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    if positions.size == 0:
        return np.ones((0,), dtype=np.float32)
    homogeneous = np.concatenate(
        [positions, np.ones((positions.shape[0], 1), dtype=np.float64)], axis=1
    )
    view_positions = homogeneous @ camera_view.T
    view_depth = -view_positions[:, 2]
    visibility = np.ones((positions.shape[0],), dtype=np.float64)
    remaining = np.ones((positions.shape[0],), dtype=bool)
    for cascade in shadow.cascades:
        selected = remaining & (view_depth <= float(cascade.split_far) + 1.0e-9)
        if not np.any(selected):
            continue
        selected_indices = np.nonzero(selected)[0]
        visibility[selected_indices] = _query_single_map(
            cascade,
            positions[selected_indices],
            tolerance,
            pcf_radius,
        )
        remaining[selected_indices] = False
    return visibility.astype(np.float32)


def _query_point(
    shadow: PointShadow,
    positions: np.ndarray,
    tolerance: float,
    pcf_radius: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float64)
    visibility = np.ones((positions.shape[0],), dtype=np.float64)
    if positions.size == 0:
        return visibility.astype(np.float32)
    directions = positions - shadow.position[None, :]
    face_index = np.argmax(np.abs(directions), axis=1)
    face_names = np.asarray(_POINT_FACES, dtype=object)
    grouped_faces = face_names[face_index]
    for face_name in _POINT_FACES:
        selected = grouped_faces == face_name
        if not np.any(selected):
            continue
        indices = np.nonzero(selected)[0]
        visibility[indices] = _query_single_map(
            shadow.faces[face_name],
            positions[indices],
            tolerance,
            pcf_radius,
        )
    return visibility.astype(np.float32)
