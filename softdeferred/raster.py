from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .gbuffer import GBuffer
from .linear import (
    bounding_sphere,
    frustum_planes,
    normalize,
    sphere_outside_frustum,
    transform_points,
    transform_vectors,
)
from .scene import MeshNode, Scene


EPS = 1.0e-10


@dataclass
class TriangleBatch:
    positions: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    bitangents: np.ndarray
    uvs: np.ndarray
    material_indices: np.ndarray
    cast_shadow: np.ndarray

    @property
    def triangle_count(self) -> int:
        return int(self.positions.shape[0])

    def subset(self, mask: np.ndarray) -> "TriangleBatch":
        indices = np.nonzero(mask)[0]
        return TriangleBatch(
            positions=self.positions[indices],
            normals=self.normals[indices],
            tangents=self.tangents[indices],
            bitangents=self.bitangents[indices],
            uvs=self.uvs[indices],
            material_indices=self.material_indices[indices],
            cast_shadow=self.cast_shadow[indices],
        )

    def byte_size(self) -> int:
        return int(
            self.positions.nbytes
            + self.normals.nbytes
            + self.tangents.nbytes
            + self.bitangents.nbytes
            + self.uvs.nbytes
            + self.material_indices.nbytes
            + self.cast_shadow.nbytes
        )


@dataclass
class MeshCullInfo:
    name: str
    center: Sequence[float]
    radius: float
    triangle_count: int
    frustum_culled: bool
    triangle_start: int
    triangle_end: int

    def report_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "bounding_sphere": {
                "center": [float(value) for value in self.center],
                "radius": float(self.radius),
            },
            "triangle_count": self.triangle_count,
            "frustum_culled": self.frustum_culled,
        }


@dataclass
class GeometryData:
    scene: Scene
    batch: TriangleBatch
    visible_batch: TriangleBatch
    mesh_infos: List[MeshCullInfo]

    def report_dict(self) -> Dict[str, Any]:
        total = self.batch.triangle_count
        culled = sum(
            info.triangle_count for info in self.mesh_infos if info.frustum_culled
        )
        return {
            "meshes_total": len(self.mesh_infos),
            "meshes_culled": [
                info.name for info in self.mesh_infos if info.frustum_culled
            ],
            "triangles_before_culling": total,
            "triangles_after_frustum_culling": total - culled,
            "mesh_frustum_culled_triangles": culled,
            "mesh_details": [info.report_dict() for info in self.mesh_infos],
        }


def prepare_geometry(scene: Scene) -> GeometryData:
    all_positions: List[np.ndarray] = []
    all_normals: List[np.ndarray] = []
    all_tangents: List[np.ndarray] = []
    all_bitangents: List[np.ndarray] = []
    all_uvs: List[np.ndarray] = []
    all_materials: List[np.ndarray] = []
    all_casts: List[np.ndarray] = []
    visible_flags: List[np.ndarray] = []
    mesh_infos: List[MeshCullInfo] = []

    camera_view = _camera_view(scene)
    camera_projection = _camera_projection(scene)
    planes = frustum_planes(camera_projection @ camera_view)

    for mesh in scene.meshes:
        world_vertices = transform_points(mesh.transform, mesh.positions)
        center, radius = bounding_sphere(world_vertices)
        culled = sphere_outside_frustum(center, radius, planes)
        indices = mesh.indices
        triangle_count = int(indices.shape[0])
        triangle_start = sum(int(array.shape[0]) for array in all_positions)
        triangle_end = triangle_start + triangle_count

        linear = mesh.transform[:3, :3]
        try:
            normal_matrix = np.linalg.inv(linear).T
        except np.linalg.LinAlgError:
            raise ValueError("mesh '%s' has a singular transform" % mesh.name)
        world_normals = normalize(transform_vectors(normal_matrix, mesh.normals))
        world_tangents = normalize(transform_vectors(linear, mesh.tangents))
        world_bitangents = normalize(transform_vectors(linear, mesh.bitangents))
        # Orthogonalize transformed tangent frames before interpolation.
        world_tangents = normalize(
            world_tangents
            - world_normals * np.sum(world_normals * world_tangents, axis=1, keepdims=True)
        )
        world_bitangents = normalize(
            world_bitangents
            - world_normals * np.sum(world_normals * world_bitangents, axis=1, keepdims=True)
            - world_tangents * np.sum(world_tangents * world_bitangents, axis=1, keepdims=True)
        )

        positions = world_vertices[indices]
        normals = world_normals[indices]
        tangents = world_tangents[indices]
        bitangents = world_bitangents[indices]
        uvs = mesh.uvs[indices]
        material_indices = mesh.material_indices
        casts = np.full(triangle_count, mesh.cast_shadow, dtype=bool)

        all_positions.append(positions)
        all_normals.append(normals)
        all_tangents.append(tangents)
        all_bitangents.append(bitangents)
        all_uvs.append(uvs)
        all_materials.append(material_indices)
        all_casts.append(casts)
        visible_flags.append(np.full(triangle_count, not culled, dtype=bool))
        mesh_infos.append(
            MeshCullInfo(
                name=mesh.name,
                center=center,
                radius=radius,
                triangle_count=triangle_count,
                frustum_culled=culled,
                triangle_start=triangle_start,
                triangle_end=triangle_end,
            )
        )

    batch = TriangleBatch(
        positions=np.concatenate(all_positions, axis=0),
        normals=np.concatenate(all_normals, axis=0),
        tangents=np.concatenate(all_tangents, axis=0),
        bitangents=np.concatenate(all_bitangents, axis=0),
        uvs=np.concatenate(all_uvs, axis=0),
        material_indices=np.concatenate(all_materials, axis=0),
        cast_shadow=np.concatenate(all_casts, axis=0),
    )
    visible_mask = np.concatenate(visible_flags, axis=0)
    return GeometryData(
        scene=scene,
        batch=batch,
        visible_batch=batch.subset(visible_mask),
        mesh_infos=mesh_infos,
    )


def _camera_view(scene: Scene) -> np.ndarray:
    from .linear import look_at

    return look_at(scene.camera.position, scene.camera.target, scene.camera.up)


def _camera_projection(scene: Scene) -> np.ndarray:
    from .linear import perspective

    aspect = scene.width / float(scene.height)
    return perspective(
        scene.camera.fov_y_deg, aspect, scene.camera.near, scene.camera.far
    )


def camera_matrices(scene: Scene) -> Tuple[np.ndarray, np.ndarray]:
    return _camera_view(scene), _camera_projection(scene)


def _clip_polygon(
    clip_positions: np.ndarray,
    attributes: np.ndarray,
    signed_distance,
) -> Tuple[np.ndarray, np.ndarray]:
    if clip_positions.shape[0] == 0:
        return clip_positions, attributes
    keep = signed_distance(clip_positions) >= -1.0e-12
    if np.all(keep):
        return clip_positions, attributes
    if not np.any(keep):
        return np.empty((0, 4), dtype=np.float64), np.empty((0, attributes.shape[1]))
    output_clip: List[np.ndarray] = []
    output_attr: List[np.ndarray] = []
    count = clip_positions.shape[0]
    for index in range(count):
        previous = (index - 1) % count
        current = index
        previous_keep = bool(keep[previous])
        current_keep = bool(keep[current])
        if previous_keep != current_keep:
            previous_distance = float(signed_distance(clip_positions[previous:previous + 1])[0])
            current_distance = float(signed_distance(clip_positions[current:current + 1])[0])
            denominator = previous_distance - current_distance
            if abs(denominator) < 1.0e-15:
                parameter = 0.0
            else:
                parameter = previous_distance / denominator
            parameter = min(max(parameter, 0.0), 1.0)
            output_clip.append(
                clip_positions[previous]
                + parameter * (clip_positions[current] - clip_positions[previous])
            )
            output_attr.append(
                attributes[previous]
                + parameter * (attributes[current] - attributes[previous])
            )
        if current_keep:
            output_clip.append(clip_positions[current])
            output_attr.append(attributes[current])
    return np.asarray(output_clip), np.asarray(output_attr)


def clip_near_far(clip_positions: np.ndarray, attributes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    clip_positions, attributes = _clip_polygon(
        clip_positions, attributes, lambda value: value[:, 2] + value[:, 3]
    )
    if clip_positions.shape[0] == 0:
        return clip_positions, attributes
    return _clip_polygon(
        clip_positions, attributes, lambda value: value[:, 3] - value[:, 2]
    )


def _interpolate_clip_attributes(
    attributes: np.ndarray,
    barycentric: np.ndarray,
    inverse_w: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    q = barycentric * inverse_w[None, :]
    denominator = np.sum(q, axis=1)
    perspective = q / np.maximum(denominator[:, None], 1.0e-15)
    return perspective @ attributes, denominator


def _normalize_interpolated_frames(
    normals: np.ndarray,
    tangents: np.ndarray,
    bitangents: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_normals = normalize(normals)
    tangents = tangents - safe_normals * np.sum(
        safe_normals * tangents, axis=1, keepdims=True
    )
    tangents = normalize(tangents)
    bitangents = bitangents - safe_normals * np.sum(
        safe_normals * bitangents, axis=1, keepdims=True
    )
    bitangents = bitangents - tangents * np.sum(
        tangents * bitangents, axis=1, keepdims=True
    )
    return safe_normals, tangents, normalize(bitangents)


def rasterize_gbuffer(
    batch: TriangleBatch,
    view_projection: np.ndarray,
    camera_position: np.ndarray,
    width: int,
    height: int,
) -> Tuple[GBuffer, Dict[str, int]]:
    depth = np.ones((height, width), dtype=np.float32)
    coverage = np.zeros((height, width), dtype=np.uint8)
    position_buffer = np.zeros((height, width, 3), dtype=np.float32)
    normal_buffer = np.zeros((height, width, 3), dtype=np.float32)
    tangent_buffer = np.zeros((height, width, 3), dtype=np.float32)
    bitangent_buffer = np.zeros((height, width, 3), dtype=np.float32)
    uv_buffer = np.zeros((height, width, 2), dtype=np.float32)
    material_buffer = np.full((height, width), -1, dtype=np.int32)

    stats = {
        "triangles_submitted": batch.triangle_count,
        "backface_culled_triangles": 0,
        "near_far_culled_triangles": 0,
        "offscreen_culled_triangles": 0,
        "rasterized_triangles": 0,
        "fragments_written": 0,
    }

    for triangle_index in range(batch.triangle_count):
        world = batch.positions[triangle_index]
        edge1 = world[1] - world[0]
        edge2 = world[2] - world[0]
        face_normal = np.cross(edge1, edge2)
        face_length = float(np.linalg.norm(face_normal))
        if face_length < EPS:
            stats["backface_culled_triangles"] += 1
            continue
        face_normal = face_normal / face_length
        if float(np.dot(face_normal, camera_position - world[0])) <= 0.0:
            stats["backface_culled_triangles"] += 1
            continue

        homogeneous = np.concatenate(
            [world, np.ones((3, 1), dtype=np.float64)], axis=1
        )
        clip = homogeneous @ view_projection.T
        attributes = np.concatenate(
            [
                world,
                batch.normals[triangle_index],
                batch.tangents[triangle_index],
                batch.bitangents[triangle_index],
                batch.uvs[triangle_index],
            ],
            axis=1,
        )
        clipped_clip, clipped_attributes = clip_near_far(clip, attributes)
        if clipped_clip.shape[0] < 3:
            stats["near_far_culled_triangles"] += 1
            continue

        produced_candidate = False
        for fan_index in range(1, clipped_clip.shape[0] - 1):
            fan_clip = clipped_clip[[0, fan_index, fan_index + 1]]
            fan_attr = clipped_attributes[[0, fan_index, fan_index + 1]]
            inverse_w = 1.0 / fan_clip[:, 3]
            ndc = fan_clip[:, :3] * inverse_w[:, None]
            screen_x = (ndc[:, 0] * 0.5 + 0.5) * width
            screen_y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height
            screen_depth = ndc[:, 2] * 0.5 + 0.5
            screen = np.stack([screen_x, screen_y], axis=1)

            minimum_x = float(np.min(screen_x))
            maximum_x = float(np.max(screen_x))
            minimum_y = float(np.min(screen_y))
            maximum_y = float(np.max(screen_y))
            if maximum_x < 0.0 or maximum_y < 0.0 or minimum_x > width or minimum_y > height:
                continue

            denominator = (
                (screen_y[1] - screen_y[2]) * (screen_x[0] - screen_x[2])
                + (screen_x[2] - screen_x[1]) * (screen_y[0] - screen_y[2])
            )
            if abs(denominator) < 1.0e-12:
                continue
            if denominator < 0.0:
                fan_clip = fan_clip[[0, 2, 1]]
                fan_attr = fan_attr[[0, 2, 1]]
                inverse_w = inverse_w[[0, 2, 1]]
                screen_x = screen_x[[0, 2, 1]]
                screen_y = screen_y[[0, 2, 1]]
                screen_depth = screen_depth[[0, 2, 1]]
                denominator = -denominator

            xmin = max(0, int(math.floor(float(np.min(screen_x))) - 1))
            xmax = min(width - 1, int(math.ceil(float(np.max(screen_x))) + 1))
            ymin = max(0, int(math.floor(float(np.min(screen_y))) - 1))
            ymax = min(height - 1, int(math.ceil(float(np.max(screen_y))) + 1))
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
            edge_epsilon = -1.0e-9
            candidate = (b0 >= edge_epsilon) & (b1 >= edge_epsilon) & (b2 >= edge_epsilon)
            if not np.any(candidate):
                continue
            produced_candidate = True
            barycentric = np.stack([b0, b1, b2], axis=-1)
            pixel_depth = (
                b0 * screen_depth[0] + b1 * screen_depth[1] + b2 * screen_depth[2]
            )
            current_depth = depth[ymin:ymax + 1, xmin:xmax + 1]
            write_mask = candidate & (pixel_depth < current_depth)
            if not np.any(write_mask):
                continue

            region_ys, region_xs = np.nonzero(write_mask)
            selected_bary = barycentric[region_ys, region_xs]
            interpolated, denominator_w = _interpolate_clip_attributes(
                fan_attr, selected_bary, inverse_w
            )
            valid = denominator_w > 1.0e-15
            if not np.any(valid):
                continue
            region_ys = region_ys[valid]
            region_xs = region_xs[valid]
            ys = region_ys + ymin
            xs = region_xs + xmin
            interpolated = interpolated[valid]
            selected_depth = pixel_depth[region_ys, region_xs]

            world_position = interpolated[:, 0:3]
            normals = interpolated[:, 3:6]
            tangents = interpolated[:, 6:9]
            bitangents = interpolated[:, 9:12]
            uvs = interpolated[:, 12:14]
            normals, tangents, bitangents = _normalize_interpolated_frames(
                normals, tangents, bitangents
            )

            depth[ys, xs] = selected_depth.astype(np.float32)
            coverage[ys, xs] = 1
            position_buffer[ys, xs] = world_position.astype(np.float32)
            normal_buffer[ys, xs] = normals.astype(np.float32)
            tangent_buffer[ys, xs] = tangents.astype(np.float32)
            bitangent_buffer[ys, xs] = bitangents.astype(np.float32)
            uv_buffer[ys, xs] = uvs.astype(np.float32)
            material_buffer[ys, xs] = int(batch.material_indices[triangle_index])
            stats["fragments_written"] += int(ys.size)

        if produced_candidate:
            stats["rasterized_triangles"] += 1
        else:
            stats["offscreen_culled_triangles"] += 1

    gbuffer = GBuffer(
        position=position_buffer,
        normal=normal_buffer,
        tangent=tangent_buffer,
        bitangent=bitangent_buffer,
        uv=uv_buffer,
        material_index=material_buffer,
        depth=depth,
        coverage=coverage,
    )
    gbuffer.validate()
    return gbuffer, stats
