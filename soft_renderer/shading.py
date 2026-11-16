from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .geometry import ViewCamera
from .mathutil import clamp, normalize, smoothstep
from .scene import Light, Material, Scene
from .shadows import ShadowMap, light_visibility


def empty_shadow_statistics(scene: Scene) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}
    for light in scene.lights:
        result[str(light.id)] = {
            "fully_lit_fragment_samples": 0,
            "penumbra_fragment_samples": 0,
            "fully_shadowed_fragment_samples": 0,
        }
    return result


def _distance_falloff(light: Light, distances: np.ndarray) -> np.ndarray:
    attenuation = light.attenuation
    denominator = (
        attenuation.constant
        + attenuation.linear * distances
        + attenuation.quadratic * distances * distances
    )
    falloff = 1.0 / np.maximum(denominator, 1.0e-12)
    if attenuation.range is not None:
        normalized = clamp(distances / attenuation.range)
        window = 1.0 - normalized * normalized * normalized * normalized
        falloff = falloff * window * window
    return falloff


def _spot_angular_falloff(light: Light, directions_to_light: np.ndarray) -> np.ndarray:
    if light.kind != "spot":
        return np.ones(len(directions_to_light), dtype=np.float64)
    cosine = np.sum(-directions_to_light * light.direction[None, :], axis=-1)
    return smoothstep(light.outer_cos, light.inner_cos, cosine)


def shade_gbuffer(
    gbuffer: Dict[str, np.ndarray],
    scene: Scene,
    camera: ViewCamera,
    materials: Dict[int, Material],
    shadow_maps: Dict[int, Optional[ShadowMap]],
    pcf_samples: int,
    statistics: Optional[Dict[str, Dict[str, int]]] = None,
    fixed_material: Optional[Material] = None,
) -> np.ndarray:
    height, width = gbuffer["material"].shape
    result = np.zeros((height, width, 3), dtype=np.float64)
    coverage = gbuffer["material"] >= 0
    if not np.any(coverage):
        return result

    positions = gbuffer["position"][coverage]
    normals = normalize(gbuffer["normal"][coverage])
    material_ids = gbuffer["material"][coverage]
    view_vectors = normalize(camera.position[None, :] - positions)
    back_facing = np.sum(normals * view_vectors, axis=-1) < 0.0
    normals[back_facing] *= -1.0

    if fixed_material is not None:
        base_color = np.broadcast_to(
            fixed_material.base_color[None, :], (len(positions), 3)
        )
        roughness = np.full(len(positions), fixed_material.roughness, dtype=np.float64)
        metallic = np.full(len(positions), fixed_material.metallic, dtype=np.float64)
    else:
        unique_ids = np.unique(material_ids)
        material_index = np.searchsorted(unique_ids, material_ids)
        base_colors = np.stack(
            [materials[int(value)].base_color for value in unique_ids], axis=0
        )
        roughness_values = np.asarray(
            [materials[int(value)].roughness for value in unique_ids], dtype=np.float64
        )
        metallic_values = np.asarray(
            [materials[int(value)].metallic for value in unique_ids], dtype=np.float64
        )
        base_color = base_colors[material_index]
        roughness = roughness_values[material_index]
        metallic = metallic_values[material_index]

    accumulated = base_color * scene.ambient[None, :]
    for light in scene.lights:
        delta = light.position[None, :] - positions
        distance = np.sqrt(np.sum(delta * delta, axis=-1))
        directions_to_light = delta / np.maximum(distance, 1.0e-12)[..., None]
        angular = _spot_angular_falloff(light, directions_to_light)
        attenuation = _distance_falloff(light, distance)
        active = (angular > 0.0) & (attenuation > 0.0)

        visibility = light_visibility(
            light,
            shadow_maps.get(light.id),
            positions,
            normals,
            directions_to_light,
            distance,
            pcf_samples,
        )
        if statistics is not None:
            light_statistics = statistics[str(light.id)]
            count_mask = active
            visible_values = visibility[count_mask]
            light_statistics["fully_lit_fragment_samples"] += int(
                np.count_nonzero(visible_values >= 1.0 - 1.0e-12)
            )
            light_statistics["penumbra_fragment_samples"] += int(
                np.count_nonzero(
                    (visible_values > 1.0e-12) & (visible_values < 1.0 - 1.0e-12)
                )
            )
            light_statistics["fully_shadowed_fragment_samples"] += int(
                np.count_nonzero(visible_values <= 1.0e-12)
            )

        ndotl = np.maximum(np.sum(normals * directions_to_light, axis=-1), 0.0)
        ndotv = np.maximum(np.sum(normals * view_vectors, axis=-1), 1.0e-6)
        half_vector = normalize(directions_to_light + view_vectors)
        ndoth = np.maximum(np.sum(normals * half_vector, axis=-1), 0.0)
        vdoth = np.maximum(np.sum(view_vectors * half_vector, axis=-1), 0.0)

        alpha = np.maximum(roughness * roughness, 1.0e-4)
        alpha_squared = alpha * alpha
        denominator = ndoth * ndoth * (alpha_squared - 1.0) + 1.0
        distribution = alpha_squared / np.maximum(np.pi * denominator * denominator, 1.0e-12)

        k = (roughness + 1.0) * (roughness + 1.0) / 8.0
        geometry_view = ndotv / np.maximum(ndotv * (1.0 - k) + k, 1.0e-12)
        geometry_light = ndotl / np.maximum(ndotl * (1.0 - k) + k, 1.0e-12)
        geometry = geometry_view * geometry_light

        f0 = 0.04 * (1.0 - metallic[:, None]) + base_color * metallic[:, None]
        fresnel = f0 + (1.0 - f0) * np.power(1.0 - vdoth[:, None], 5.0)
        specular = distribution[:, None] * geometry[:, None] * fresnel
        specular /= np.maximum(4.0 * ndotv[:, None] * ndotl[:, None], 1.0e-12)
        diffuse = (1.0 - fresnel) * (1.0 - metallic[:, None]) * base_color / np.pi
        brdf = diffuse + specular

        radiance = light.color[None, :] * light.intensity
        contribution = (
            brdf
            * ndotl[:, None]
            * (visibility * attenuation * angular)[:, None]
            * radiance[None, :]
        )
        accumulated = accumulated + contribution

    result[coverage] = accumulated
    return result
