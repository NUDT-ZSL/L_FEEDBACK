from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math3d import F64, safe_normalize, transform_points
from .raster import rasterize_gbuffer
from .scene import Scene
from .shadow import ShadowContext, build_shadow_context


SHADING_ROW_CHUNK = 64

ISSUE_KEYS = (
    "degenerate_clip_count",
    "clipped_near_triangles",
    "clipped_far_triangles",
    "invalid_triangle_count",
    "invalid_projected_triangle_count",
    "degenerate_triangle_count",
    "invalid_raster_pixel_count",
    "invalid_normal_pixel_count",
    "rasterized_pixel_count",
    "shadow_invalid_triangle_count",
    "shadow_degenerate_triangle_count",
    "shadow_rasterized_pixel_count",
    "shadow_invalid_sample_count",
    "singular_bone_matrix_count",
    "invalid_vertex_count",
    "nan_vertex_count",
    "invalid_pixel_count",
    "nan_pixel_count",
)


@dataclass(frozen=True)
class PoseResult:
    world_vertices: np.ndarray
    normal_vertices: np.ndarray
    view_vertices: np.ndarray


@dataclass(frozen=True)
class FrameRender:
    image: np.ndarray
    stats: dict


def _new_issues() -> dict[str, int]:
    return {key: 0 for key in ISSUE_KEYS}


def _pose(scene: Scene, frame: int, issues: dict[str, int]) -> PoseResult:
    local_matrices = scene.local_by_frame[frame]
    bone_count = len(scene.bones)
    world_matrices = np.zeros((bone_count, 4, 4), dtype=F64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for bone_index in scene.topo_order:
            parent = int(scene.parents[bone_index])
            if parent < 0:
                world_matrices[bone_index] = local_matrices[bone_index]
            else:
                world_matrices[bone_index] = (
                    world_matrices[parent] @ local_matrices[bone_index]
                )
        skin_matrices = world_matrices @ scene.inverse_bind
        affine = skin_matrices[:, :3, :3]
        determinant = np.linalg.det(affine)
        good = np.isfinite(determinant) & (np.abs(determinant) > 1e-12)
        inverse_affine = np.zeros_like(affine)
        if np.any(good):
            inverse_affine[good] = np.linalg.inv(affine[good])
        bad_indices = np.flatnonzero(~good)
        for bone_index in bad_indices:
            issues["singular_bone_matrix_count"] += 1
            matrix = affine[int(bone_index)]
            if np.all(np.isfinite(matrix)):
                try:
                    inverse_affine[int(bone_index)] = np.linalg.pinv(matrix)
                except np.linalg.LinAlgError:
                    pass
        normal_matrices = inverse_affine.transpose(0, 2, 1)

        selected_skin = skin_matrices[scene.skin_indices]
        homogeneous_vertices = np.concatenate(
            [
                scene.vertices,
                np.ones((scene.vertices.shape[0], 1), dtype=F64),
            ],
            axis=1,
        )
        transformed = np.einsum(
            "nkij,nj->nki", selected_skin, homogeneous_vertices
        )
        world_vertices = np.sum(
            transformed[:, :, :3] * scene.skin_weights[:, :, None],
            axis=1,
        )
        selected_normals = normal_matrices[scene.skin_indices]
        transformed_normals = np.einsum(
            "nkij,nj->nki", selected_normals, scene.normals
        )
        normal_vertices = np.sum(
            transformed_normals * scene.skin_weights[:, :, None],
            axis=1,
        )
        normal_vertices = safe_normalize(normal_vertices)
        view_vertices = transform_points(scene.camera.view, world_vertices)

    invalid_vertices = ~np.all(np.isfinite(world_vertices), axis=1)
    invalid_vertices |= ~np.all(np.isfinite(normal_vertices), axis=1)
    invalid_vertices |= ~np.all(np.isfinite(view_vertices), axis=1)
    issues["invalid_vertex_count"] = int(np.count_nonzero(invalid_vertices))
    issues["nan_vertex_count"] = int(
        np.count_nonzero(
            np.any(np.isnan(world_vertices), axis=1)
            | np.any(np.isnan(normal_vertices), axis=1)
            | np.any(np.isnan(view_vertices), axis=1)
        )
    )
    return PoseResult(
        world_vertices=world_vertices,
        normal_vertices=normal_vertices,
        view_vertices=view_vertices,
    )


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    clamped = np.clip(linear, 0.0, 1.0)
    return np.where(
        clamped <= 0.0031308,
        12.92 * clamped,
        1.055 * np.power(clamped, 1.0 / 2.4) - 0.055,
    )


def _background_bytes(scene: Scene) -> np.ndarray:
    return np.rint(scene.background * 255.0).astype(np.uint8)


def _shade(
    scene: Scene,
    shadow: ShadowContext,
    depth_buffer: np.ndarray,
    world_buffer: np.ndarray,
    normal_buffer: np.ndarray,
    material_buffer: np.ndarray,
    issues: dict[str, int],
) -> np.ndarray:
    height, width = depth_buffer.shape
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[...] = _background_bytes(scene)
    material_diffuse = np.stack([material.diffuse for material in scene.materials])
    material_specular = np.stack(
        [material.specular for material in scene.materials]
    )
    material_shininess = np.asarray(
        [material.shininess for material in scene.materials], dtype=F64
    )
    light_direction = scene.directional.direction
    point_position = scene.point.position

    for row_start in range(0, height, SHADING_ROW_CHUNK):
        row_end = min(height, row_start + SHADING_ROW_CHUNK)
        depth_chunk = depth_buffer[row_start:row_end]
        coverage = np.isfinite(depth_chunk)
        if not np.any(coverage):
            continue
        world = world_buffer[row_start:row_end][coverage]
        normals = normal_buffer[row_start:row_end][coverage]
        view_depths = depth_chunk[coverage]
        material_indices = material_buffer[row_start:row_end][coverage]
        valid_material = (material_indices >= 0) & (
            material_indices < len(scene.materials)
        )
        if not np.all(valid_material):
            issues["invalid_pixel_count"] += int(
                np.count_nonzero(~valid_material)
            )
        safe_material = np.clip(material_indices, 0, len(scene.materials) - 1)
        diffuse = material_diffuse[safe_material]
        specular = material_specular[safe_material]
        shininess = material_shininess[safe_material]

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            world_valid = np.all(np.isfinite(world), axis=1)
            normal_valid = np.all(np.isfinite(normals), axis=1)
            normals = safe_normalize(normals)
            view_vectors = safe_normalize(scene.camera.position[None, :] - world)
            facing = np.sum(normals * view_vectors, axis=1)
            normals = np.where(
                (facing < 0.0)[:, None], -normals, normals
            )
            color = diffuse * scene.ambient[None, :]

            if scene.directional.intensity > 0.0:
                ndl = np.maximum(0.0, normals @ light_direction)
                visibility = np.ones(world.shape[0], dtype=F64)
                if scene.directional.cast_shadow:
                    visibility, invalid_samples = shadow.visibility(
                        scene,
                        world,
                        normals,
                        view_depths,
                    )
                    issues["shadow_invalid_sample_count"] += invalid_samples
                half_vector = safe_normalize(
                    light_direction[None, :] + view_vectors
                )
                ndh = np.maximum(0.0, np.sum(normals * half_vector, axis=1))
                diffuse_directional = (
                    diffuse
                    * scene.directional.color[None, :]
                    * (scene.directional.intensity * ndl)[:, None]
                )
                specular_directional = (
                    specular
                    * scene.directional.color[None, :]
                    * (
                        scene.directional.intensity
                        * np.power(ndh, shininess)
                        * (ndl > 0.0)
                    )[:, None]
                )
                color += visibility[:, None] * (
                    diffuse_directional + specular_directional
                )

            if scene.point.intensity > 0.0:
                point_vectors = point_position[None, :] - world
                point_distance = np.linalg.norm(point_vectors, axis=1)
                attenuation = 1.0 / (
                    1.0 + scene.point.falloff * point_distance * point_distance
                )
                point_direction = safe_normalize(point_vectors)
                point_ndl = np.maximum(0.0, np.sum(normals * point_direction, axis=1))
                point_half = safe_normalize(
                    point_direction + view_vectors
                )
                point_ndh = np.maximum(
                    0.0, np.sum(normals * point_half, axis=1)
                )
                point_energy = scene.point.intensity * attenuation
                diffuse_point = (
                    diffuse
                    * scene.point.color[None, :]
                    * (point_energy * point_ndl)[:, None]
                )
                specular_point = (
                    specular
                    * scene.point.color[None, :]
                    * (
                        point_energy
                        * np.power(point_ndh, shininess)
                        * (point_ndl > 0.0)
                    )[:, None]
                )
                color += diffuse_point + specular_point

            bad = ~np.isfinite(color).all(axis=1)
            bad |= ~world_valid | ~normal_valid | ~valid_material
            nan_bad = np.any(np.isnan(color), axis=1)
            issues["invalid_pixel_count"] += int(np.count_nonzero(bad))
            issues["nan_pixel_count"] += int(np.count_nonzero(nan_bad))
            color = np.nan_to_num(
                color, nan=0.0, posinf=0.0, neginf=0.0
            )
            srgb = _linear_to_srgb(color)
            display = np.rint(srgb * 255.0).astype(np.uint8)

        display_chunk = np.empty((row_end - row_start, width, 3), dtype=np.uint8)
        display_chunk[...] = _background_bytes(scene)
        display_chunk[coverage] = display
        display[bad] = _background_bytes(scene)
        display_chunk[coverage] = display
        image[row_start:row_end] = display_chunk
    return image


def render_frame(scene: Scene, frame: int) -> FrameRender:
    issues = _new_issues()
    pose = _pose(scene, frame, issues)
    shadow = build_shadow_context(scene, pose.world_vertices, issues)
    (
        depth_buffer,
        world_buffer,
        normal_buffer,
        material_buffer,
    ) = rasterize_gbuffer(
        scene,
        pose.world_vertices,
        pose.normal_vertices,
        pose.view_vertices,
        issues,
    )
    image = _shade(
        scene,
        shadow,
        depth_buffer,
        world_buffer,
        normal_buffer,
        material_buffer,
        issues,
    )

    coverage = np.isfinite(depth_buffer)
    rendered_pixel_count = int(np.count_nonzero(coverage))
    if rendered_pixel_count:
        depth_min = float(np.min(depth_buffer[coverage]))
        depth_max = float(np.max(depth_buffer[coverage]))
    else:
        depth_min = 0.0
        depth_max = 0.0
    if rendered_pixel_count:
        cascade_indices = np.searchsorted(
            scene.shadow.splits, depth_buffer[coverage], side="left"
        )
        cascade_indices = np.clip(cascade_indices, 0, 2)
        cascade_pixel_counts = np.bincount(
            cascade_indices, minlength=3
        ).tolist()
    else:
        cascade_pixel_counts = [0, 0, 0]
    visible_materials = material_buffer[material_buffer >= 0]
    material_pixel_counts = np.bincount(
        visible_materials, minlength=len(scene.materials)
    ).tolist()
    shadow_ranges = [
        {
            "index": cascade.index,
            "camera_near": cascade.camera_near,
            "camera_far": cascade.camera_far,
            "valid_pixel_ratio": cascade.valid_ratio,
            "valid_texels": cascade.valid_texels,
        }
        for cascade in shadow.cascades
    ]
    stats = {
        "schema_version": 1,
        "frame": frame,
        "frame_hash": scene.frame_hashes[frame],
        "structure_hash": scene.structure_hash,
        "width": scene.width,
        "height": scene.height,
        "rendered_pixel_count": rendered_pixel_count,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "vertex_count": int(scene.vertices.shape[0]),
        "triangle_count": int(scene.triangles.shape[0]),
        "bone_count": len(scene.bones),
        "material_count": len(scene.materials),
        "shadow_resolution": scene.shadow.resolution,
        "shadow_bias": scene.shadow.bias + scene.shadow.resolution_depth_bias,
        "shadow_normal_offset": (
            scene.shadow.normal_offset * scene.shadow.normal_offset_scale
        ),
        "shadow_cascade_valid_ratio": shadow.valid_ratios,
        "shadow_cascades": shadow_ranges,
        "shadow_cascade_pixel_counts": cascade_pixel_counts,
        "material_pixel_counts": material_pixel_counts,
        "invalid_pixel_count": issues["invalid_pixel_count"],
        "nan_pixel_count": issues["nan_pixel_count"],
        "issues": {key: int(value) for key, value in issues.items()},
    }
    return FrameRender(image=image, stats=stats)
