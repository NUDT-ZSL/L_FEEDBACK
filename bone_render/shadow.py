from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math3d import F64, normalize
from .raster import rasterize_ortho_depth
from .scene import Scene


@dataclass(frozen=True)
class Cascade:
    index: int
    camera_near: float
    camera_far: float
    right: np.ndarray
    up: np.ndarray
    forward: np.ndarray
    center: np.ndarray
    half_extent: float
    depth_near: float
    depth_far: float
    resolution: int
    depth_map: np.ndarray

    @property
    def valid_texels(self) -> int:
        return int(np.count_nonzero(np.isfinite(self.depth_map)))

    @property
    def valid_ratio(self) -> float:
        return self.valid_texels / float(self.resolution * self.resolution)

    def project(self, world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        relative = world - self.center
        x = relative @ self.right
        y = relative @ self.up
        depth_world = relative @ self.forward
        screen_x = (x / self.half_extent + 1.0) * 0.5 * self.resolution
        screen_y = (1.0 - y / self.half_extent) * 0.5 * self.resolution
        normalized_depth = (depth_world - self.depth_near) / (
            self.depth_far - self.depth_near
        )
        return screen_x, screen_y, normalized_depth


@dataclass(frozen=True)
class ShadowContext:
    cascades: tuple[Cascade, ...]
    splits: np.ndarray
    resolution: int
    resolution_depth_bias: float
    normal_offset: float

    @property
    def valid_ratios(self) -> list[float]:
        return [cascade.valid_ratio for cascade in self.cascades]

    def visibility(
        self,
        scene: Scene,
        world: np.ndarray,
        normals: np.ndarray,
        view_depths: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Return directional-light visibility and a count of invalid samples."""

        if not scene.directional.cast_shadow or world.shape[0] == 0:
            return np.ones(world.shape[0], dtype=F64), 0
        cascade_indices = np.searchsorted(
            self.splits, view_depths, side="left"
        )
        cascade_indices = np.clip(cascade_indices, 0, len(self.cascades) - 1)
        visibility = np.ones(world.shape[0], dtype=F64)
        invalid_count = 0
        light_direction = scene.directional.direction
        for cascade in self.cascades:
            selected = np.flatnonzero(cascade_indices == cascade.index)
            if selected.size == 0:
                continue
            selected_world = world[selected]
            selected_normals = normals[selected]
            normal_offset = (
                self.normal_offset
                * scene.shadow.normal_offset_scale
            )
            offset_world = selected_world + selected_normals * normal_offset
            relative = offset_world - cascade.center
            x = relative @ cascade.right
            y = relative @ cascade.up
            depth_world = relative @ cascade.forward
            receiver_depth = (depth_world - cascade.depth_near) / (
                cascade.depth_far - cascade.depth_near
            )
            x_ndc = x / cascade.half_extent
            y_ndc = y / cascade.half_extent
            inside = (
                (x_ndc >= -1.0)
                & (x_ndc <= 1.0)
                & (y_ndc >= -1.0)
                & (y_ndc <= 1.0)
                & (receiver_depth >= 0.0)
                & (receiver_depth <= 1.0)
            )
            if not np.any(inside):
                continue
            ndl = np.maximum(0.0, selected_normals @ light_direction)
            depth_bias = (
                scene.shadow.bias
                + self.resolution_depth_bias
                + scene.shadow.slope_bias * (1.0 - ndl)
            )
            texture_x = (x_ndc + 1.0) * 0.5 * cascade.resolution - 0.5
            texture_y = (1.0 - y_ndc) * 0.5 * cascade.resolution - 0.5
            base_x = np.floor(texture_x).astype(np.int64)
            base_y = np.floor(texture_y).astype(np.int64)
            visible_samples = np.zeros(selected.size, dtype=np.int32)
            for offset_y in (-1, 0, 1):
                sample_y = np.clip(
                    base_y + offset_y, 0, cascade.resolution - 1
                )
                for offset_x in (-1, 0, 1):
                    sample_x = np.clip(
                        base_x + offset_x, 0, cascade.resolution - 1
                    )
                    stored_depth = cascade.depth_map[sample_y, sample_x]
                    visible = (
                        ~inside
                        | ~np.isfinite(stored_depth)
                        | (
                            stored_depth + depth_bias >= receiver_depth
                        )
                    )
                    visible_samples += visible.astype(np.int32)
            cascade_visibility = visible_samples.astype(F64) / 9.0
            invalid_count += int(
                np.count_nonzero(
                    inside
                    & (
                        ~np.isfinite(receiver_depth)
                        | ~np.isfinite(texture_x)
                        | ~np.isfinite(texture_y)
                    )
                )
            )
            visibility[selected] = cascade_visibility
        return visibility, invalid_count


def _light_basis(direction_to_light: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = normalize(-direction_to_light)
    up_hint = np.asarray([0.0, 1.0, 0.0], dtype=F64)
    if abs(float(np.dot(forward, up_hint))) > 0.95:
        up_hint = np.asarray([0.0, 0.0, 1.0], dtype=F64)
    right = normalize(np.cross(forward, up_hint))
    up = normalize(np.cross(right, forward))
    return right, up, forward


def _camera_slice_corners(
    scene: Scene, near_distance: float, far_distance: float
) -> np.ndarray:
    camera = scene.camera
    aspect = scene.width / scene.height
    tan_half_fov = np.tan(camera.fov_y_radians * 0.5)
    corners: list[np.ndarray] = []
    for distance in (near_distance, far_distance):
        half_height = tan_half_fov * distance
        half_width = half_height * aspect
        center = camera.position + camera.forward * distance
        for vertical in (-1.0, 1.0):
            for horizontal in (-1.0, 1.0):
                corners.append(
                    center
                    + camera.right * (horizontal * half_width)
                    + camera.true_up * (vertical * half_height)
                )
    return np.stack(corners, axis=0)


def build_shadow_context(
    scene: Scene,
    world_vertices: np.ndarray,
    issues: dict[str, int],
) -> ShadowContext:
    right, up, forward = _light_basis(scene.directional.direction)
    resolution = scene.shadow.resolution
    cascades: list[Cascade] = []
    previous_split = scene.camera.near
    for cascade_index, split in enumerate(scene.shadow.splits):
        near_distance = previous_split
        far_distance = float(split)
        previous_split = far_distance
        corners = _camera_slice_corners(scene, near_distance, far_distance)
        center_distance = (near_distance + far_distance) * 0.5
        reference = scene.camera.position + scene.camera.forward * center_distance
        relative = corners - reference
        x = relative @ right
        y = relative @ up
        z = relative @ forward
        half_extent = max(
            float(np.max(np.abs(x))),
            float(np.max(np.abs(y))),
            1e-3,
        )
        half_extent *= scene.shadow.extent_padding
        depth_near = float(np.min(z)) - scene.shadow.depth_padding
        depth_far = float(np.max(z)) + scene.shadow.depth_padding
        if depth_far <= depth_near:
            depth_far = depth_near + 1.0
        cascade = Cascade(
            index=cascade_index,
            camera_near=float(near_distance),
            camera_far=float(far_distance),
            right=right,
            up=up,
            forward=forward,
            center=reference,
            half_extent=half_extent,
            depth_near=depth_near,
            depth_far=depth_far,
            resolution=resolution,
            depth_map=np.full((resolution, resolution), np.inf, dtype=F64),
        )
        screen_x, screen_y, depth = cascade.project(world_vertices)
        depth_map = rasterize_ortho_depth(
            screen_x,
            screen_y,
            depth,
            scene.triangles,
            resolution,
            issues,
        )
        cascade = Cascade(
            index=cascade.index,
            camera_near=cascade.camera_near,
            camera_far=cascade.camera_far,
            right=cascade.right,
            up=cascade.up,
            forward=cascade.forward,
            center=cascade.center,
            half_extent=cascade.half_extent,
            depth_near=cascade.depth_near,
            depth_far=cascade.depth_far,
            resolution=cascade.resolution,
            depth_map=depth_map,
        )
        cascades.append(cascade)
    return ShadowContext(
        cascades=tuple(cascades),
        splits=np.asarray(scene.shadow.splits, dtype=F64),
        resolution=resolution,
        resolution_depth_bias=scene.shadow.resolution_depth_bias,
        normal_offset=scene.shadow.normal_offset,
    )
