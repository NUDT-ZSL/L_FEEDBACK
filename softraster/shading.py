"""Layered metallic-roughness shading with one shadow reduction per light."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math3d import normalize
from .scene import Layer, Light, Scene
from .shadows import ShadowMap, compute_visibility


@dataclass(frozen=True)
class LightSample:
    direction: np.ndarray
    radiance: np.ndarray


def shade_scene(
    scene: Scene,
    gbuffer,
    shadow_maps: tuple[ShadowMap | None, ...],
) -> np.ndarray:
    height = scene.render.height
    width = scene.render.width
    background = scene.background
    flat_background = np.broadcast_to(
        background, (height * width, 3)
    ).astype(np.float64).copy()

    hit_pixels = np.flatnonzero(gbuffer.hit_mask.ravel())
    if hit_pixels.size == 0:
        return flat_background.reshape((height, width, 3)).astype(np.float32)

    all_positions = gbuffer.world_position.reshape((-1, 3))
    all_normals = gbuffer.normal.reshape((-1, 3))
    all_mesh_indices = gbuffer.mesh_index.ravel()
    positions = all_positions[hit_pixels]
    normal = all_normals[hit_pixels]
    mesh_indices = all_mesh_indices[hit_pixels]
    view = normalize(scene.camera.eye[None, :] - positions)

    visibility = [
        compute_visibility(light, shadow_map, positions, normal)
        for light, shadow_map in zip(scene.lights, shadow_maps)
    ]
    light_samples = [
        _sample_light(
            scene,
            light,
            index,
            positions,
            visibility[index],
        )
        for index, light in enumerate(scene.lights)
    ]

    for mesh_index, mesh in enumerate(scene.meshes):
        local = np.flatnonzero(mesh_indices == mesh_index)
        if local.size == 0:
            continue
        material = scene.materials[mesh.material_index]
        points = positions[local]
        geometric_normal = normal[local]
        view_direction = view[local]
        local_lights = [
            LightSample(
                direction=sample.direction[local],
                radiance=sample.radiance[local],
            )
            for sample in light_samples
        ]

        result = np.broadcast_to(
            background, (local.size, 3)
        ).astype(np.float64).copy()
        for layer in material.layers:
            layer_normal = perturbed_layer_normal(geometric_normal, layer)
            layer_radiance = _evaluate_layer(
                scene,
                layer,
                points,
                layer_normal,
                view_direction,
                local_lights,
            )
            alpha = layer.opacity
            result = result * (1.0 - alpha) + layer_radiance * alpha

        flat_background[hit_pixels[local]] = np.maximum(result, 0.0)

    image = flat_background.reshape((height, width, 3)) * scene.render.exposure
    if not np.all(np.isfinite(image)):
        raise FloatingPointError("shading produced non-finite pixel values")
    return image.astype(np.float32)


def perturbed_layer_normal(
    geometric_normal: np.ndarray, layer: Layer
) -> np.ndarray:
    """Compute a layer normal only from the geometric normal.

    Earlier layers are intentionally absent from this function: a bottom-layer
    bump therefore cannot propagate into, or reverse, an upper clearcoat.
    """
    geometric_normal = normalize(np.asarray(geometric_normal, dtype=np.float64))
    reference = np.broadcast_to(
        np.asarray([0.0, 1.0, 0.0]), geometric_normal.shape
    ).copy()
    parallel = np.abs(geometric_normal[:, 1]) > 0.98
    reference[parallel] = np.asarray([1.0, 0.0, 0.0])
    tangent = normalize(np.cross(reference, geometric_normal))
    bitangent = normalize(np.cross(geometric_normal, tangent))
    perturbed = (
        geometric_normal * layer.normal[2]
        + layer.normal_strength * layer.normal[0] * tangent
        + layer.normal_strength * layer.normal[1] * bitangent
    )
    return normalize(perturbed)


def _sample_light(
    scene: Scene,
    light: Light,
    light_index: int,
    positions: np.ndarray,
    visibility: np.ndarray,
) -> LightSample:
    if light.kind == "directional":
        assert light.direction is not None
        direction = np.broadcast_to(
            -normalize(light.direction), positions.shape
        ).copy()
        radiance = np.broadcast_to(
            light.color * light.intensity, positions.shape
        ).copy()
    else:
        assert light.position is not None
        assert light.range is not None
        delta = light.position[None, :] - positions
        distance_squared = np.sum(delta * delta, axis=1, keepdims=True)
        distance = np.sqrt(distance_squared)
        direction = delta / np.maximum(distance, 1.0e-12)
        normalized_distance = distance / light.range
        range_window = np.clip(
            1.0 - normalized_distance**4, 0.0, 1.0
        ) ** 2
        attenuation = range_window / np.maximum(distance_squared, 1.0e-8)
        radiance = (
            (light.color * light.intensity)[None, :] * attenuation
        )
    radiance = radiance * visibility[:, None]
    return LightSample(direction=direction, radiance=radiance)


def _evaluate_layer(
    scene: Scene,
    layer: Layer,
    positions: np.ndarray,
    normal: np.ndarray,
    view: np.ndarray,
    lights: list[LightSample],
) -> np.ndarray:
    del positions
    base_color = layer.base_color[None, :]
    metallic = layer.metallic
    roughness = max(layer.roughness, 1.0e-3)
    diffuse_color = base_color * (1.0 - metallic)
    f0 = 0.04 * (1.0 - metallic) + base_color * metallic
    ndotv = np.clip(np.sum(normal * view, axis=1), 1.0e-5, 1.0)

    ambient = np.broadcast_to(
        scene.ambient_color[None, :] * scene.ambient_intensity,
        (normal.shape[0], 3),
    ).copy()
    fresnel_ambient = f0 + (np.maximum(1.0 - roughness, f0) - f0) * (
        (1.0 - ndotv) ** 5
    )[:, None]
    result = diffuse_color * ambient
    result += fresnel_ambient * ambient * (1.0 - roughness) ** 2

    clearcoat_roughness = max(0.03, roughness * 0.55)
    ambient_clearcoat = (
        0.04
        * ambient
        * layer.clearcoat
        * ((1.0 - ndotv) ** 5)[:, None]
        * (1.0 - clearcoat_roughness) ** 2
    )
    result += ambient_clearcoat

    for sample in lights:
        light_direction = sample.direction
        half_vector = normalize(view + light_direction)
        ndotl = np.clip(np.sum(normal * light_direction, axis=1), 0.0, 1.0)
        ndoth = np.clip(np.sum(normal * half_vector, axis=1), 0.0, 1.0)
        vdoth = np.clip(np.sum(view * half_vector, axis=1), 0.0, 1.0)
        active = ndotl > 0.0

        distribution = _ggx_distribution(ndoth, roughness)
        geometry = _smith_geometry(ndotl, ndotv, roughness)
        fresnel = f0 + (1.0 - f0) * ((1.0 - vdoth) ** 5)[:, None]
        specular = (
            fresnel
            * distribution[:, None]
            * geometry[:, None]
            / np.maximum(4.0 * ndotl * ndotv, 1.0e-7)[:, None]
        )
        diffuse = diffuse_color * (1.0 - fresnel)
        contribution = (diffuse / np.pi + specular) * (
            ndotl[:, None] * sample.radiance
        )

        clearcoat_ndoth = ndoth
        clearcoat_distribution = _ggx_distribution(
            clearcoat_ndoth, clearcoat_roughness
        )
        clearcoat_geometry = _smith_geometry(
            ndotl, ndotv, clearcoat_roughness
        )
        clearcoat_fresnel = 0.04 + 0.96 * ((1.0 - vdoth) ** 5)
        clearcoat_specular = (
            clearcoat_fresnel[:, None]
            * clearcoat_distribution[:, None]
            * clearcoat_geometry[:, None]
            / np.maximum(4.0 * ndotl * ndotv, 1.0e-7)[:, None]
        )
        contribution += (
            clearcoat_specular
            * layer.clearcoat
            * ndotl[:, None]
            * sample.radiance
        )
        result += np.where(active[:, None], contribution, 0.0)

    return result


def _ggx_distribution(ndoth: np.ndarray, roughness: float) -> np.ndarray:
    alpha = roughness * roughness
    alpha_squared = alpha * alpha
    denominator = ndoth * ndoth * (alpha_squared - 1.0) + 1.0
    return alpha_squared / np.maximum(
        np.pi * denominator * denominator, 1.0e-12
    )


def _smith_geometry(
    ndotl: np.ndarray, ndotv: np.ndarray, roughness: float
) -> np.ndarray:
    k = ((roughness + 1.0) ** 2) / 8.0
    geometry_l = ndotl / np.maximum(ndotl * (1.0 - k) + k, 1.0e-12)
    geometry_v = ndotv / np.maximum(ndotv * (1.0 - k) + k, 1.0e-12)
    return geometry_l * geometry_v
