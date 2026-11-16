from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .gbuffer import GBuffer
from .linear import normalize
from .scene import Light, Material, Scene
from .shadows import ShadowCollection


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return (value >= edge1).astype(np.float64)
    t = np.clip((value - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _material_table(
    materials: List[Material],
) -> Dict[str, np.ndarray]:
    return {
        "base_color": np.stack([item.base_color for item in materials]).astype(np.float64),
        "specular_color": np.stack([item.specular_color for item in materials]).astype(
            np.float64
        ),
        "metallic": np.asarray([item.metallic for item in materials], dtype=np.float64),
        "exponent_tangent": np.asarray(
            [item.exponent_tangent for item in materials], dtype=np.float64
        ),
        "exponent_bitangent": np.asarray(
            [item.exponent_bitangent for item in materials], dtype=np.float64
        ),
        "rotation_rad": np.radians(
            np.asarray(
                [item.anisotropy_rotation_deg for item in materials], dtype=np.float64
            )
        ),
    }


def _anisotropic_specular(
    normal: np.ndarray,
    tangent: np.ndarray,
    bitangent: np.ndarray,
    light_direction: np.ndarray,
    view_direction: np.ndarray,
    base_color: np.ndarray,
    specular_color: np.ndarray,
    exponent_tangent: np.ndarray,
    exponent_bitangent: np.ndarray,
    rotation_rad: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    cos_rotation = np.cos(rotation_rad)[:, None]
    sin_rotation = np.sin(rotation_rad)[:, None]
    rotated_tangent = tangent * cos_rotation + bitangent * sin_rotation
    rotated_bitangent = -tangent * sin_rotation + bitangent * cos_rotation

    half_vector = normalize(view_direction + light_direction)
    n_dot_h = np.clip(np.sum(normal * half_vector, axis=1), 0.0, 1.0)
    t_dot_h = np.sum(rotated_tangent * half_vector, axis=1)
    b_dot_h = np.sum(rotated_bitangent * half_vector, axis=1)
    v_dot_h = np.clip(np.sum(view_direction * half_vector, axis=1), 0.0, 1.0)

    lobe_exponent = (
        exponent_tangent * t_dot_h * t_dot_h
        + exponent_bitangent * b_dot_h * b_dot_h
    )
    normalization = np.sqrt(
        (exponent_tangent + 2.0) * (exponent_bitangent + 2.0)
    ) / (8.0 * np.pi)
    distribution = normalization * np.power(n_dot_h, np.maximum(lobe_exponent, 0.0))
    fresnel = specular_color + (1.0 - specular_color) * np.power(1.0 - v_dot_h, 5.0)[
        :, None
    ]
    return distribution, fresnel


def shade_deferred(
    gbuffer: GBuffer,
    scene: Scene,
    shadows: ShadowCollection,
    camera_view: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, Dict[str, int]], int]:
    height, width = gbuffer.depth.shape
    hdr = np.zeros((height * width, 3), dtype=np.float64)
    covered = np.nonzero(gbuffer.coverage.reshape(-1) != 0)[0]
    interactions: Dict[str, int] = {}
    details: Dict[str, Dict[str, int]] = {}
    total_shadow_queries = 0
    if covered.size == 0:
        for light in scene.lights:
            interactions[light.light_id] = 0
            details[light.light_id] = {
                "candidate_pixels": 0,
                "shadow_queries": 0,
                "fully_shadowed_pixels": 0,
                "lit_pixels": 0,
            }
        return (
            hdr.reshape(height, width, 3).astype(np.float32),
            interactions,
            details,
            total_shadow_queries,
        )

    positions = gbuffer.position.reshape(-1, 3)[covered].astype(np.float64)
    normals = normalize(gbuffer.normal.reshape(-1, 3)[covered].astype(np.float64))
    tangents = normalize(
        gbuffer.tangent.reshape(-1, 3)[covered].astype(np.float64)
    )
    bitangents = normalize(
        gbuffer.bitangent.reshape(-1, 3)[covered].astype(np.float64)
    )
    material_indices = gbuffer.material_index.reshape(-1)[covered]
    if np.any(material_indices < 0) or np.any(material_indices >= len(scene.materials)):
        raise ValueError("GBuffer contains an invalid material index")

    tables = _material_table(scene.materials)
    base_color = tables["base_color"][material_indices]
    specular_color = tables["specular_color"][material_indices]
    metallic = tables["metallic"][material_indices, None]
    exponent_tangent = tables["exponent_tangent"][material_indices]
    exponent_bitangent = tables["exponent_bitangent"][material_indices]
    rotation_rad = tables["rotation_rad"][material_indices]

    view_vector = scene.camera.position[None, :] - positions
    view_vector = normalize(view_vector)

    for light in scene.lights:
        if light.light_type == "directional":
            light_direction = np.broadcast_to(
                -normalize(light.direction)[None, :], positions.shape
            )
            attenuation = np.ones((covered.size,), dtype=np.float64)
            candidate = np.ones((covered.size,), dtype=bool)
        else:
            delta = positions - light.position[None, :]
            distance_squared = np.sum(delta * delta, axis=1)
            distance = np.sqrt(distance_squared)
            light_direction = delta / np.maximum(distance[:, None], 1.0e-12)
            attenuation = 1.0 / np.maximum(distance_squared, 1.0e-8)
            candidate = distance_squared <= float(light.range) ** 2
            if light.light_type == "spot":
                cosine_theta = np.sum(
                    light_direction * normalize(light.direction)[None, :], axis=1
                )
                cosine_inner = np.cos(np.radians(light.inner_angle_deg))
                cosine_outer = np.cos(np.radians(light.outer_angle_deg))
                cone_factor = _smoothstep(cosine_outer, cosine_inner, cosine_theta)
                candidate &= cosine_theta >= cosine_outer
                attenuation = attenuation * cone_factor

        n_dot_l = np.clip(np.sum(normals * light_direction, axis=1), 0.0, 1.0)
        shading_mask = candidate & (n_dot_l > 0.0)
        visibility = np.ones((covered.size,), dtype=np.float64)
        if np.any(shading_mask):
            shading_indices = np.nonzero(shading_mask)[0]
            visibility[shading_indices] = shadows.visibility(
                light, positions[shading_indices], camera_view
            )
        query_count = int(np.count_nonzero(shading_mask))
        total_shadow_queries += query_count

        contribution = np.zeros_like(positions)
        if np.any(shading_mask):
            shading_indices = np.nonzero(shading_mask)[0]
            distribution, fresnel = _anisotropic_specular(
                normals[shading_indices],
                tangents[shading_indices],
                bitangents[shading_indices],
                light_direction[shading_indices],
                view_vector[shading_indices],
                base_color[shading_indices],
                specular_color[shading_indices],
                exponent_tangent[shading_indices],
                exponent_bitangent[shading_indices],
                rotation_rad[shading_indices],
            )
            lambert = base_color[shading_indices] * (1.0 - metallic[shading_indices])
            diffuse = lambert * n_dot_l[shading_indices, None]
            specular = (
                fresnel
                * distribution[:, None]
                * n_dot_l[shading_indices, None]
            )
            local_attenuation = attenuation[shading_indices, None]
            local_visibility = visibility[shading_indices, None]
            radiance = light.color[None, :] * light.intensity
            contribution[shading_indices] = (
                (diffuse + specular) * local_attenuation * local_visibility * radiance
            )
        hdr[covered] += contribution

        interaction_count = int(np.count_nonzero(candidate))
        interactions[light.light_id] = interaction_count
        details[light.light_id] = {
            "candidate_pixels": interaction_count,
            "shadow_queries": query_count,
            "fully_shadowed_pixels": int(
                np.count_nonzero(shading_mask & (visibility < 0.5))
            ),
            "lit_pixels": int(
                np.count_nonzero(shading_mask & (visibility >= 0.5))
            ),
        }

    hdr = np.maximum(hdr, 0.0)
    if not np.all(np.isfinite(hdr)):
        raise ValueError("deferred lighting produced non-finite HDR values")
    return (
        hdr.reshape(height, width, 3).astype(np.float32),
        interactions,
        details,
        total_shadow_queries,
    )
