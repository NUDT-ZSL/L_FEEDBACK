"""Deterministic CPU raster renderer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .hiz import HierarchicalZ
from .math3d import (
    look_at,
    normalize,
    normalize_rows,
    orthographic,
    perspective,
    quantize_srgb,
    screen_triangle_depth,
    transform_directions,
    transform_points,
)
from .raster import (
    DepthTriangle,
    ProjectedTriangle,
    bbox_intersects,
    draw_depth_tested_line,
    rasterize_depth_triangles,
    screen_bbox,
    triangle_coverage,
)
from .scene import Camera, Light, Scene, load_scene
from .textures import sample_texture


POINT_FACE_DIRECTIONS = (
    np.array([1.0, 0.0, 0.0]),
    np.array([-1.0, 0.0, 0.0]),
    np.array([0.0, 1.0, 0.0]),
    np.array([0.0, -1.0, 0.0]),
    np.array([0.0, 0.0, 1.0]),
    np.array([0.0, 0.0, -1.0]),
)


@dataclass
class ShadowFace:
    matrix: np.ndarray
    depth: np.ndarray
    resolution: int
    bias_scale: float


@dataclass
class RenderedShadow:
    light: Light
    kind: str
    faces: tuple[ShadowFace, ...]
    face_directions: tuple[np.ndarray, ...] | None
    submitted_triangles: int = 0
    rasterized_triangles: int = 0


@dataclass
class Counter:
    total_triangles: int = 0
    backface_culled_triangles: int = 0
    degenerate_triangles: int = 0
    offscreen_triangles: int = 0
    near_clipped_original_triangles: int = 0
    near_fully_outside_triangles: int = 0
    near_clipped_output_triangles: int = 0
    projected_triangles: int = 0
    hiz_culled_triangles: int = 0
    depth_prepass_triangles: int = 0
    depth_prepass_rasterized_triangles: int = 0
    opaque_main_triangles: int = 0
    transparent_triangles: int = 0
    shaded_fragments: int = 0
    opaque_fragments: int = 0
    transparent_fragments: int = 0
    opaque_draw_calls: int = 0
    transparent_draw_calls: int = 0
    wireframe_pixels: int = 0
    normal_overlay_pixels: int = 0
    shaded_mask: np.ndarray = field(repr=False, default_factory=lambda: np.empty((0, 0), dtype=bool))


@dataclass
class RenderResult:
    image: np.ndarray
    stats: dict[str, Any]


def _generate_tangents(scene: Scene) -> np.ndarray:
    mesh = scene.mesh
    vertices = mesh.vertices
    normals = mesh.normals
    uvs = mesh.uvs
    tangent_accum = np.zeros_like(vertices)
    bitangent_accum = np.zeros_like(vertices)
    for primitive in mesh.primitives:
        triangles = primitive.indices
        p0 = vertices[triangles[:, 0]]
        p1 = vertices[triangles[:, 1]]
        p2 = vertices[triangles[:, 2]]
        uv0 = uvs[triangles[:, 0]]
        uv1 = uvs[triangles[:, 1]]
        uv2 = uvs[triangles[:, 2]]
        edge1 = p1 - p0
        edge2 = p2 - p0
        duv1 = uv1 - uv0
        duv2 = uv2 - uv0
        determinant = duv1[:, 0] * duv2[:, 1] - duv2[:, 0] * duv1[:, 1]
        valid = np.abs(determinant) > 1e-12
        if not np.any(valid):
            continue
        inverse = 1.0 / determinant[valid]
        tangent = (edge1[valid] * duv2[valid, 1, None] - edge2[valid] * duv1[valid, 1, None]) * inverse[:, None]
        bitangent = (edge2[valid] * duv1[valid, 0, None] - edge1[valid] * duv2[valid, 0, None]) * inverse[:, None]
        for corner in range(3):
            np.add.at(tangent_accum, triangles[valid, corner], tangent)
            np.add.at(bitangent_accum, triangles[valid, corner], bitangent)

    tangents = tangent_accum - normals * np.sum(normals * tangent_accum, axis=1, keepdims=True)
    lengths = np.linalg.norm(tangents, axis=1)
    bad = lengths <= 1e-10
    if np.any(bad):
        reference = np.where(
            (np.abs(normals[bad, 0]) < 0.9)[:, None],
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 1.0, 0.0]]),
        )
        tangents[bad] = np.cross(normals[bad], reference)
    tangents = normalize_rows(tangents, fallback=np.array([1.0, 0.0, 0.0]))
    handedness = np.sign(np.sum(np.cross(normals, tangents) * bitangent_accum, axis=1))
    handedness[handedness == 0.0] = 1.0
    return np.concatenate([tangents, handedness[:, None]], axis=1)


def _clip_camera_polygon(
    camera_positions: np.ndarray,
    world_positions: np.ndarray,
    world_normals: np.ndarray,
    uvs: np.ndarray,
    tangents: np.ndarray,
    near: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    plane_z = -near
    rows = [
        (
            camera_positions[index],
            world_positions[index],
            world_normals[index],
            uvs[index],
            tangents[index],
        )
        for index in range(3)
    ]
    inside = [float(row[0][2]) <= plane_z for row in rows]
    if all(inside):
        return [rows]
    if not any(inside):
        return []
    polygon: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for index in range(3):
        current = rows[index]
        following = rows[(index + 1) % 3]
        current_inside = inside[index]
        following_inside = inside[(index + 1) % 3]
        if current_inside:
            polygon.append(current)
        if current_inside != following_inside:
            z0 = float(current[0][2])
            z1 = float(following[0][2])
            denominator = z1 - z0
            t = 0.0 if abs(denominator) <= 1e-15 else (plane_z - z0) / denominator
            polygon.append(
                (
                    current[0] + (following[0] - current[0]) * t,
                    current[1] + (following[1] - current[1]) * t,
                    current[2] + (following[2] - current[2]) * t,
                    current[3] + (following[3] - current[3]) * t,
                    current[4] + (following[4] - current[4]) * t,
                )
            )
    return [polygon] if len(polygon) >= 3 else []


def _project_point(
    world_position: np.ndarray,
    view_projection: np.ndarray,
    width: int,
    height: int,
) -> tuple[float, float, float] | None:
    clip = view_projection @ np.append(world_position, 1.0)
    if float(clip[3]) <= 1e-9:
        return None
    ndc = clip[:3] / clip[3]
    depth = float(screen_triangle_depth(np.array([ndc[2]]))[0])
    x = (float(ndc[0]) * 0.5 + 0.5) * width
    y = (1.0 - (float(ndc[1]) * 0.5 + 0.5)) * height
    return x, y, depth


def _prepare_primary_triangles(
    scene: Scene,
    world_positions: np.ndarray,
    world_normals: np.ndarray,
    world_tangents: np.ndarray,
    camera: Camera,
    view: np.ndarray,
    projection: np.ndarray,
    counters: Counter,
) -> list[ProjectedTriangle]:
    camera_homogeneous = transform_points(view, world_positions)
    camera_positions = camera_homogeneous[:, :3]
    records: list[ProjectedTriangle] = []

    for primitive in scene.mesh.primitives:
        material = scene.materials[primitive.material]
        for triangle_index in primitive.indices:
            counters.total_triangles += 1
            camera_triangle = camera_positions[triangle_index]
            edge1 = camera_triangle[1] - camera_triangle[0]
            edge2 = camera_triangle[2] - camera_triangle[0]
            face_normal = np.cross(edge1, edge2)
            normal_length = float(np.linalg.norm(face_normal))
            if normal_length <= 1e-12:
                counters.degenerate_triangles += 1
                continue
            centroid = np.mean(camera_triangle, axis=0)
            front_facing = float(np.dot(face_normal, -centroid)) > 0.0
            if camera.cull_mode == "back" and not material.double_sided and not front_facing:
                counters.backface_culled_triangles += 1
                continue

            clipped = _clip_camera_polygon(
                camera_triangle,
                world_positions[triangle_index],
                world_normals[triangle_index],
                scene.mesh.uvs[triangle_index],
                world_tangents[triangle_index],
                camera.near,
            )
            inside_count = int(np.count_nonzero(camera_triangle[:, 2] <= -camera.near))
            if inside_count == 0:
                counters.near_fully_outside_triangles += 1
                continue
            was_near_clipped = inside_count < 3
            if was_near_clipped:
                counters.near_clipped_original_triangles += 1
            if not clipped:
                continue

            camera_depth = float(np.mean(-camera_triangle[:, 2]))
            for polygon in clipped:
                for fan_index in range(1, len(polygon) - 1):
                    selected = (polygon[0], polygon[fan_index], polygon[fan_index + 1])
                    cam_vertices = np.stack([row[0] for row in selected])
                    clip = (projection @ np.concatenate(
                        [cam_vertices, np.ones((3, 1), dtype=np.float64)], axis=1
                    ).T).T
                    w = clip[:, 3]
                    if np.any(w <= 1e-9):
                        continue
                    ndc = clip[:, :3] / w[:, None]
                    depth = screen_triangle_depth(ndc[:, 2])
                    screen = np.empty((3, 2), dtype=np.float64)
                    screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * camera.width
                    screen[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * camera.height
                    bbox = screen_bbox(screen, camera.width, camera.height)
                    if not bbox_intersects(*bbox, camera.width, camera.height):
                        counters.offscreen_triangles += 1
                        continue
                    counters.projected_triangles += 1
                    if was_near_clipped:
                        counters.near_clipped_output_triangles += 1
                    records.append(
                        ProjectedTriangle(
                            screen=screen,
                            depth=depth,
                            inv_w=1.0 / w,
                            world_pos=np.stack([row[1] for row in selected]),
                            world_normal=np.stack([row[2] for row in selected]),
                            uv=np.stack([row[3] for row in selected]),
                            tangent=np.stack([row[4] for row in selected]),
                            material=material,
                            camera_depth=camera_depth,
                            bbox=bbox,
                        )
                    )
    return records


def _ordered_material_groups(
    records: list[ProjectedTriangle], transparent: bool
) -> list[tuple[str, list[ProjectedTriangle]]]:
    groups: dict[str, list[ProjectedTriangle]] = {}
    for record in records:
        groups.setdefault(record.material.name, []).append(record)

    def segment_depth(item: tuple[str, list[ProjectedTriangle]]) -> float:
        values = [triangle.camera_depth for triangle in item[1]]
        return float(sum(values) / len(values))

    ordered = sorted(
        groups.items(),
        key=lambda item: (segment_depth(item), item[0]),
        reverse=transparent,
    )
    for _, triangles in ordered:
        triangles.sort(
            key=lambda triangle: (triangle.camera_depth, tuple(triangle.screen.reshape(-1))),
            reverse=transparent,
        )
    return ordered


def _world_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    center = (minimum + maximum) * 0.5
    corners = np.array(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ]
    )
    radius = float(np.max(np.linalg.norm(corners - center, axis=1)))
    return center, corners, max(radius, 1e-3)


def _shadow_depth_triangles(
    world_triangles: np.ndarray,
    view: np.ndarray,
    projection: np.ndarray,
    resolution: int,
    near: float,
) -> list[DepthTriangle]:
    triangles: list[DepthTriangle] = []
    homogeneous = np.concatenate(
        [world_triangles.reshape((-1, 3)), np.ones((3, 1), dtype=np.float64)], axis=1
    )
    camera_positions = (view @ homogeneous.T).T[:, :3].reshape((3, 3))
    inside = camera_positions[:, 2] <= -near
    polygons: list[list[np.ndarray]] = []
    if np.all(inside):
        polygons = [[camera_positions[0], camera_positions[1], camera_positions[2]]]
    elif np.any(inside):
        polygon: list[np.ndarray] = []
        for index in range(3):
            current = camera_positions[index]
            following = camera_positions[(index + 1) % 3]
            current_inside = bool(inside[index])
            following_inside = bool(inside[(index + 1) % 3])
            if current_inside:
                polygon.append(current)
            if current_inside != following_inside:
                denominator = float(following[2] - current[2])
                t = 0.0 if abs(denominator) <= 1e-15 else (-near - float(current[2])) / denominator
                polygon.append(current + (following - current) * t)
        if len(polygon) >= 3:
            polygons = [polygon]

    for polygon in polygons:
        for fan_index in range(1, len(polygon) - 1):
            camera_triangle = np.stack([polygon[0], polygon[fan_index], polygon[fan_index + 1]])
            clip = (projection @ np.concatenate(
                [camera_triangle, np.ones((3, 1), dtype=np.float64)], axis=1
            ).T).T
            w = clip[:, 3]
            if np.any(w <= 1e-9):
                continue
            ndc = clip[:, :3] / w[:, None]
            depth = screen_triangle_depth(ndc[:, 2])
            screen = np.empty((3, 2), dtype=np.float64)
            screen[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * resolution
            screen[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * resolution
            bbox = screen_bbox(screen, resolution, resolution)
            if not bbox_intersects(*bbox, resolution, resolution):
                continue
            triangles.append(DepthTriangle(screen=screen, depth=depth, bbox=bbox))
    return triangles


def _opaque_world_triangles(scene: Scene, world_positions: np.ndarray) -> list[np.ndarray]:
    triangles: list[np.ndarray] = []
    for primitive in scene.mesh.primitives:
        material = scene.materials[primitive.material]
        if material.transparent:
            continue
        triangles.extend(world_positions[primitive.indices])
    return triangles


def _build_shadows(
    scene: Scene,
    world_positions: np.ndarray,
    counters: Counter,
) -> list[RenderedShadow]:
    center, _, radius = _world_bounds(world_positions)
    opaque_triangles = _opaque_world_triangles(scene, world_positions)
    shadows: list[RenderedShadow] = []
    for light in scene.lights:
        if not light.shadow.enabled or not opaque_triangles:
            continue
        resolution = light.shadow.resolution
        if light.kind == "directional":
            assert light.direction is not None
            direction = normalize(light.direction)
            up = np.array([0.0, 0.0, 1.0]) if abs(float(direction[1])) > 0.95 else np.array([0.0, 1.0, 0.0])
            distance = radius * 2.0 + 1.0
            light_position = center - direction * distance
            view = look_at(light_position, center, up)
            extent = radius * 1.08 + 1e-3
            near = 0.01
            far = distance + radius * 2.0 + 1.0
            projection = orthographic(-extent, extent, -extent, extent, near, far)
            matrix = projection @ view
            depth = np.ones((resolution, resolution), dtype=np.float64)
            submitted = 0
            rasterized = 0
            for world_triangle in opaque_triangles:
                projected = _shadow_depth_triangles(
                    world_triangle, view, projection, resolution, near
                )
                submitted += len(projected)
                rasterized += rasterize_depth_triangles(projected, depth)
            texel_world = (2.0 * extent) / resolution
            face = ShadowFace(
                matrix=matrix,
                depth=depth,
                resolution=resolution,
                bias_scale=texel_world / max(far - near, 1e-6),
            )
            shadows.append(
                RenderedShadow(
                    light=light,
                    kind="directional",
                    faces=(face,),
                    face_directions=None,
                    submitted_triangles=submitted,
                    rasterized_triangles=rasterized,
                )
            )
        else:
            assert light.position is not None and light.light_range is not None
            faces: list[ShadowFace] = []
            submitted = 0
            rasterized = 0
            near = 0.03
            far = float(light.light_range)
            projection = perspective(90.0, 1.0, near, far)
            for face_direction in POINT_FACE_DIRECTIONS:
                if abs(float(face_direction[1])) > 0.5:
                    up = np.array([0.0, 0.0, 1.0])
                else:
                    up = np.array([0.0, 1.0, 0.0])
                view = look_at(light.position, light.position + face_direction, up)
                matrix = projection @ view
                depth = np.ones((resolution, resolution), dtype=np.float64)
                for world_triangle in opaque_triangles:
                    projected = _shadow_depth_triangles(
                        world_triangle, view, projection, resolution, near
                    )
                    submitted += len(projected)
                    rasterized += rasterize_depth_triangles(projected, depth)
                faces.append(
                    ShadowFace(
                        matrix=matrix,
                        depth=depth,
                        resolution=resolution,
                        bias_scale=1.0 / resolution,
                    )
                )
            shadows.append(
                RenderedShadow(
                    light=light,
                    kind="point",
                    faces=tuple(faces),
                    face_directions=POINT_FACE_DIRECTIONS,
                    submitted_triangles=submitted,
                    rasterized_triangles=rasterized,
                )
            )
    return shadows


def _point_face_index(offset: np.ndarray) -> np.ndarray:
    absolute = np.abs(offset)
    major = np.argmax(absolute, axis=1)
    positive = offset[np.arange(offset.shape[0]), major] >= 0.0
    return np.where(
        major == 0,
        np.where(positive, 0, 1),
        np.where(major == 1, np.where(positive, 2, 3), np.where(positive, 4, 5)),
    )


def _pcf_visibility(
    face: ShadowFace,
    u: np.ndarray,
    v: np.ndarray,
    receiver_depth: np.ndarray,
    bias: np.ndarray,
    radius: int,
    valid_projection: np.ndarray,
) -> np.ndarray:
    resolution = face.resolution
    lit = np.zeros(u.shape[0], dtype=np.float64)
    samples = 0
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            samples += 1
            sample_x = np.floor(u * resolution + offset_x).astype(np.int64)
            sample_y = np.floor(v * resolution + offset_y).astype(np.int64)
            inside = (
                valid_projection
                & (sample_x >= 0)
                & (sample_x < resolution)
                & (sample_y >= 0)
                & (sample_y < resolution)
            )
            stored = np.ones(u.shape[0], dtype=np.float64)
            if np.any(inside):
                stored[inside] = face.depth[sample_y[inside], sample_x[inside]]
            lit += (receiver_depth - bias <= stored + 1e-6).astype(np.float64)
    return lit / float(samples)


def _shadow_visibility(
    shadow: RenderedShadow,
    world_position: np.ndarray,
    geometric_normal: np.ndarray,
    shading_normal: np.ndarray,
    light_direction: np.ndarray,
) -> np.ndarray:
    settings = shadow.light.shadow
    shifted = world_position + geometric_normal * settings.normal_bias
    homogeneous = np.concatenate(
        [shifted, np.ones((shifted.shape[0], 1), dtype=np.float64)], axis=1
    )
    n_dot_l = np.clip(np.sum(shading_normal * light_direction, axis=1), 0.0, 1.0)
    slope = np.sqrt(np.clip(1.0 - n_dot_l * n_dot_l, 0.0, 1.0)) / np.maximum(n_dot_l, 0.2)
    slope = np.minimum(slope, 4.0)

    if shadow.kind == "directional":
        face = shadow.faces[0]
        clip = (face.matrix @ homogeneous.T).T
        w = clip[:, 3]
        valid = np.abs(w) > 1e-9
        safe_w = np.where(valid, w, 1.0)
        ndc = clip[:, :3] / safe_w[:, None]
        u = ndc[:, 0] * 0.5 + 0.5
        v = 0.5 - ndc[:, 1] * 0.5
        receiver_depth = ndc[:, 2] * 0.5 + 0.5
        bias = settings.bias + settings.slope_bias * slope * face.bias_scale
        return _pcf_visibility(face, u, v, receiver_depth, bias, settings.pcf_radius, valid)

    assert shadow.light.position is not None
    offset = shifted - shadow.light.position
    face_indices = _point_face_index(offset)
    visibility = np.ones(shifted.shape[0], dtype=np.float64)
    for face_index, face in enumerate(shadow.faces):
        mask = face_indices == face_index
        if not np.any(mask):
            continue
        selected_homogeneous = homogeneous[mask]
        clip = (face.matrix @ selected_homogeneous.T).T
        w = clip[:, 3]
        valid = w > 1e-9
        safe_w = np.where(valid, w, 1.0)
        ndc = clip[:, :3] / safe_w[:, None]
        u = ndc[:, 0] * 0.5 + 0.5
        v = 0.5 - ndc[:, 1] * 0.5
        receiver_depth = ndc[:, 2] * 0.5 + 0.5
        bias = settings.bias + settings.slope_bias * slope[mask] * face.bias_scale
        values = _pcf_visibility(
            face, u, v, receiver_depth, bias, settings.pcf_radius, valid
        )
        visibility[mask] = values
    return visibility


def _interpolate_attributes(
    triangle: ProjectedTriangle, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    world_position = weights @ triangle.world_pos
    geometric_normal = normalize_rows(
        weights @ triangle.world_normal, fallback=np.array([0.0, 1.0, 0.0])
    )
    uv = weights @ triangle.uv
    tangent = weights @ triangle.tangent
    return world_position, geometric_normal, uv, tangent


def _shade_fragments(
    scene: Scene,
    triangle: ProjectedTriangle,
    weights: np.ndarray,
    shadows: list[RenderedShadow],
) -> np.ndarray:
    material = triangle.material
    world_position, geometric_normal, uv, tangent = _interpolate_attributes(triangle, weights)
    diffuse = sample_texture(material.diffuse_map, uv[:, 0], uv[:, 1])
    specular_mask = sample_texture(material.specular_map, uv[:, 0], uv[:, 1])
    normal_sample = sample_texture(material.normal_map, uv[:, 0], uv[:, 1])
    albedo = np.clip(material.base_color[None, :] * diffuse, 0.0, 1.0)

    tangent_vector = tangent[:, :3]
    tangent_vector = tangent_vector - geometric_normal * np.sum(
        geometric_normal * tangent_vector, axis=1, keepdims=True
    )
    tangent_lengths = np.linalg.norm(tangent_vector, axis=1)
    bad_tangent = tangent_lengths <= 1e-10
    if np.any(bad_tangent):
        reference = np.where(
            (np.abs(geometric_normal[bad_tangent, 0]) < 0.9)[:, None],
            np.array([[1.0, 0.0, 0.0]]),
            np.array([[0.0, 1.0, 0.0]]),
        )
        tangent_vector[bad_tangent] = np.cross(geometric_normal[bad_tangent], reference)
    tangent_vector = normalize_rows(tangent_vector, fallback=np.array([1.0, 0.0, 0.0]))
    bitangent = normalize_rows(
        np.cross(geometric_normal, tangent_vector) * tangent[:, 3, None],
        fallback=np.array([0.0, 0.0, 1.0]),
    )
    tangent_normal = normal_sample * 2.0 - 1.0
    tangent_normal[:, 0:2] *= material.normal_strength
    tangent_normal = normalize_rows(tangent_normal, fallback=np.array([0.0, 0.0, 1.0]))
    shading_normal = normalize_rows(
        tangent_vector * tangent_normal[:, 0, None]
        + bitangent * tangent_normal[:, 1, None]
        + geometric_normal * tangent_normal[:, 2, None],
        fallback=geometric_normal,
    )

    view_vector = normalize_rows(
        scene.camera.position[None, :] - world_position,
        fallback=np.array([0.0, 0.0, 1.0]),
    )
    if material.double_sided:
        back_facing = np.sum(shading_normal * view_vector, axis=1) < 0.0
        shading_normal[back_facing] *= -1.0

    specular_factor = np.mean(specular_mask, axis=1) * material.specular
    linear = scene.camera.ambient[None, :] * albedo + material.emission[None, :]
    for shadow in shadows:
        light = shadow.light
        if light.intensity <= 0.0:
            continue
        light_color = light.color * light.intensity
        if light.kind == "directional":
            assert light.direction is not None
            light_direction = np.broadcast_to(light.direction, world_position.shape)
            attenuation = np.ones(world_position.shape[0], dtype=np.float64)
        else:
            assert light.position is not None and light.light_range is not None
            to_light = light.position[None, :] - world_position
            distance = np.linalg.norm(to_light, axis=1)
            safe_distance = np.maximum(distance, 1e-9)
            light_direction = to_light / safe_distance[:, None]
            assert light.attenuation is not None
            constant, linear_term, quadratic = light.attenuation
            attenuation = 1.0 / (
                constant + linear_term * distance + quadratic * distance * distance
            )
            attenuation *= (distance <= light.light_range).astype(np.float64)
        n_dot_l = np.clip(np.sum(shading_normal * light_direction, axis=1), 0.0, 1.0)
        if shadow.faces:
            visibility = _shadow_visibility(
                shadow, world_position, geometric_normal, shading_normal, light_direction
            )
        else:
            visibility = np.ones(world_position.shape[0], dtype=np.float64)
        visibility *= (n_dot_l > 1e-8).astype(np.float64)
        half_vector = normalize_rows(
            light_direction + view_vector, fallback=light_direction
        )
        n_dot_h = np.clip(np.sum(shading_normal * half_vector, axis=1), 0.0, 1.0)
        specular = np.power(n_dot_h, material.shininess) * specular_factor * n_dot_l
        contribution = albedo * n_dot_l[:, None] + specular[:, None]
        linear += contribution * light_color[None, :] * (
            visibility * attenuation
        )[:, None]

    # A light with shadows disabled still needs its direct term. It is present in
    # ``shadows`` as an unshadowed wrapper, constructed below for every light.
    return np.clip(linear, 0.0, 1.0e8)


def _unshadowed_lights(scene: Scene, shadows: list[RenderedShadow]) -> list[RenderedShadow]:
    present = {id(shadow.light) for shadow in shadows}
    for light in scene.lights:
        if id(light) not in present:
            shadows.append(
                RenderedShadow(
                    light=light,
                    kind=light.kind,
                    faces=(),
                    face_directions=None,
                )
            )
    return shadows


def _apply_overlays(
    scene: Scene,
    records: list[ProjectedTriangle],
    color: np.ndarray,
    depth: np.ndarray,
    view_projection: np.ndarray,
    counters: Counter,
) -> None:
    height, width = depth.shape
    wireframe_mask = np.zeros((height, width), dtype=bool)
    normal_mask = np.zeros((height, width), dtype=bool)
    if scene.debug.wireframe:
        for triangle in records:
            for edge in range(3):
                following = (edge + 1) % 3
                start = (
                    float(triangle.screen[edge, 0]),
                    float(triangle.screen[edge, 1]),
                    float(triangle.depth[edge]),
                )
                end = (
                    float(triangle.screen[following, 0]),
                    float(triangle.screen[following, 1]),
                    float(triangle.depth[following]),
                )
                counters.wireframe_pixels += draw_depth_tested_line(
                    color,
                    depth,
                    start,
                    end,
                    scene.debug.wireframe_color,
                    wireframe_mask,
                )
    if scene.debug.normals:
        for triangle in records:
            for vertex in range(3):
                start = (
                    float(triangle.screen[vertex, 0]),
                    float(triangle.screen[vertex, 1]),
                    float(triangle.depth[vertex]),
                )
                endpoint = (
                    triangle.world_pos[vertex]
                    + triangle.world_normal[vertex] * scene.debug.normal_length
                )
                projected = _project_point(endpoint, view_projection, width, height)
                if projected is None:
                    continue
                counters.normal_overlay_pixels += draw_depth_tested_line(
                    color,
                    depth,
                    start,
                    projected,
                    scene.debug.normal_color,
                    normal_mask,
                )


def render_scene(scene: Scene) -> RenderResult:
    camera = scene.camera
    world_positions = transform_points(scene.mesh.model, scene.mesh.vertices)[:, :3]
    world_normals = normalize_rows(
        transform_directions(scene.mesh.normal_transform, scene.mesh.normals),
        fallback=np.array([0.0, 1.0, 0.0]),
    )
    if scene.mesh.tangents is None:
        local_tangents = _generate_tangents(scene)
    else:
        local_tangents = scene.mesh.tangents
    world_tangents = transform_directions(scene.mesh.model, local_tangents[:, :3])
    model_determinant = float(np.linalg.det(scene.mesh.model[:3, :3]))
    world_tangents = np.concatenate(
        [
            normalize_rows(world_tangents, fallback=np.array([1.0, 0.0, 0.0])),
            local_tangents[:, 3:4] * (1.0 if model_determinant >= 0.0 else -1.0),
        ],
        axis=1,
    )

    view = look_at(camera.position, camera.target, camera.up)
    projection = perspective(
        camera.fov_y, camera.width / float(camera.height), camera.near, camera.far
    )
    view_projection = projection @ view
    counters = Counter(shaded_mask=np.zeros((camera.height, camera.width), dtype=bool))
    records = _prepare_primary_triangles(
        scene,
        world_positions,
        world_normals,
        world_tangents,
        camera,
        view,
        projection,
        counters,
    )

    opaque_records = [record for record in records if not record.material.transparent]
    transparent_records = [record for record in records if record.material.transparent]
    opaque_groups = _ordered_material_groups(opaque_records, transparent=False)
    transparent_groups = _ordered_material_groups(transparent_records, transparent=True)
    opaque_order = [record for _, group in opaque_groups for record in group]
    transparent_order = [record for _, group in transparent_groups for record in group]
    counters.opaque_draw_calls = len(opaque_groups)
    counters.transparent_draw_calls = len(transparent_groups)
    counters.transparent_triangles = len(transparent_records)

    depth_buffer = np.ones((camera.height, camera.width), dtype=np.float64)
    for start_index in range(0, len(opaque_order), 4096):
        block = opaque_order[start_index : start_index + 4096]
        depth_triangles = [
            DepthTriangle(screen=record.screen, depth=record.depth, bbox=record.bbox)
            for record in block
        ]
        counters.depth_prepass_triangles += len(depth_triangles)
        counters.depth_prepass_rasterized_triangles += rasterize_depth_triangles(
            depth_triangles, depth_buffer
        )
    hiz = HierarchicalZ(depth_buffer)

    shadows = _build_shadows(scene, world_positions, counters)
    shadows = _unshadowed_lights(scene, shadows)
    linear_color = np.broadcast_to(
        camera.background[None, None, :], (camera.height, camera.width, 3)
    ).copy()

    for _, group in opaque_groups:
        for triangle in group:
            x_min, y_min, x_max, y_max = triangle.bbox
            nearest_depth = float(np.min(triangle.depth))
            if hiz.fully_occluded(
                x_min, y_min, x_max, y_max, nearest_depth
            ):
                counters.hiz_culled_triangles += 1
                continue
            counters.opaque_main_triangles += 1
            pixels, depths, weights = triangle_coverage(
                triangle.screen, triangle.depth, triangle.inv_w, triangle.bbox
            )
            if pixels.size == 0:
                continue
            y = pixels[:, 0]
            x = pixels[:, 1]
            visible = depths <= depth_buffer[y, x] + 1e-7
            if not np.any(visible):
                continue
            selected_weights = weights[visible]
            shaded = _shade_fragments(scene, triangle, selected_weights, shadows)
            selected_y = y[visible]
            selected_x = x[visible]
            linear_color[selected_y, selected_x] = shaded
            counters.shaded_mask[selected_y, selected_x] = True
            counters.shaded_fragments += int(np.count_nonzero(visible))
            counters.opaque_fragments += int(np.count_nonzero(visible))

    for _, group in transparent_groups:
        for triangle in group:
            pixels, depths, weights = triangle_coverage(
                triangle.screen, triangle.depth, triangle.inv_w, triangle.bbox
            )
            if pixels.size == 0:
                continue
            y = pixels[:, 0]
            x = pixels[:, 1]
            visible = depths < depth_buffer[y, x] - 1e-7
            if not np.any(visible):
                continue
            shaded = _shade_fragments(scene, triangle, weights[visible], shadows)
            alpha = triangle.material.opacity
            selected_y = y[visible]
            selected_x = x[visible]
            linear_color[selected_y, selected_x] = (
                shaded * alpha + linear_color[selected_y, selected_x] * (1.0 - alpha)
            )
            counters.shaded_mask[selected_y, selected_x] = True
            counters.shaded_fragments += int(np.count_nonzero(visible))
            counters.transparent_fragments += int(np.count_nonzero(visible))

    _apply_overlays(
        scene,
        records,
        linear_color,
        depth_buffer,
        view_projection,
        counters,
    )

    image = quantize_srgb(linear_color * camera.exposure)
    culled = (
        counters.backface_culled_triangles
        + counters.degenerate_triangles
        + counters.offscreen_triangles
        + counters.near_fully_outside_triangles
        + counters.hiz_culled_triangles
    )
    stats: dict[str, Any] = {
        "schemaVersion": 1,
        "scene": {
            "inputSha256": scene.input_sha256,
            "resolution": [camera.width, camera.height],
            "camera": {
                "position": [float(value) for value in camera.position],
                "target": [float(value) for value in camera.target],
                "fovY": camera.fov_y,
                "exposure": camera.exposure,
            },
        },
        "counts": {
            "total_triangles": counters.total_triangles,
            "culled_triangles": culled,
            "backface_culled_triangles": counters.backface_culled_triangles,
            "degenerate_triangles": counters.degenerate_triangles,
            "offscreen_triangles": counters.offscreen_triangles,
            "near_fully_outside_triangles": counters.near_fully_outside_triangles,
            "near_clipped_original_triangles": counters.near_clipped_original_triangles,
            "near_clipped_output_triangles": counters.near_clipped_output_triangles,
            "projected_triangles": counters.projected_triangles,
            "hiz_culled_triangles": counters.hiz_culled_triangles,
            "depth_prepass_triangles": counters.depth_prepass_triangles,
            "depth_prepass_rasterized_triangles": counters.depth_prepass_rasterized_triangles,
            "opaque_main_triangles": counters.opaque_main_triangles,
            "transparent_triangles": counters.transparent_triangles,
            "opaque_draw_calls": counters.opaque_draw_calls,
            "transparent_draw_calls": counters.transparent_draw_calls,
            "shaded_pixels": int(np.count_nonzero(counters.shaded_mask)),
            "shaded_fragments": counters.shaded_fragments,
            "opaque_fragments": counters.opaque_fragments,
            "transparent_fragments": counters.transparent_fragments,
            "wireframe_pixels": counters.wireframe_pixels,
            "normal_overlay_pixels": counters.normal_overlay_pixels,
        },
        "hiz": {
            "levels": hiz.level_count,
            "full_resolution_depth": [camera.width, camera.height],
        },
        "lights": [],
        "output": {"format": "PNG", "colorSpace": "sRGB", "gamma": "IEC 61966-2-1"},
    }
    shadow_by_light = {id(shadow.light): shadow for shadow in shadows}
    for light in scene.lights:
        shadow = shadow_by_light[id(light)]
        has_shadow_map = bool(shadow.faces)
        stats["lights"].append(
            {
                "name": light.name,
                "type": light.kind,
                "shadowEnabled": light.shadow.enabled,
                "shadowMapResolution": (
                    [light.shadow.resolution, light.shadow.resolution]
                    if has_shadow_map
                    else None
                ),
                "shadowMapFaces": len(shadow.faces),
                "pcfRadius": light.shadow.pcf_radius,
                "pcfKernel": [2 * light.shadow.pcf_radius + 1, 2 * light.shadow.pcf_radius + 1],
                "shadowSubmittedTriangles": shadow.submitted_triangles,
                "shadowRasterizedTriangles": shadow.rasterized_triangles,
            }
        )
    stats["summary"] = {
        "triangle_count": counters.total_triangles,
        "culled_triangle_count": culled,
        "shaded_pixel_count": int(np.count_nonzero(counters.shaded_mask)),
        "shadow_maps": [
            {
                "light": item["name"],
                "resolution": item["shadowMapResolution"],
                "faces": item["shadowMapFaces"],
            }
            for item in stats["lights"]
        ],
    }
    stats["determinism"] = {
        "pixelQuantization": "round-half-up to 8-bit sRGB",
        "randomness": "none",
    }
    return RenderResult(image=image, stats=stats)


def render_scene_file(scene_path: str | Path) -> RenderResult:
    return render_scene(load_scene(scene_path))
