"""Deterministic deferred material shading and frame composition."""

from __future__ import annotations

from dataclasses import dataclass

from .gbuffer import GBufferTile
from .math3d import dot, length, normalize, sub
from .scene import Light, Scene
from .shadows import ShadowMaps, shadow_visibility


@dataclass(frozen=True)
class TileShading:
    color: list[float]
    light_usage: tuple[str, ...]


def _point_attenuation(distance: float, light_range: float | None) -> float:
    if light_range is not None:
        if distance >= light_range:
            return 0.0
        ratio = distance / light_range
        ratio2 = ratio * ratio
        ratio4 = ratio2 * ratio2
        window = 1.0 - ratio4
        window *= window
    else:
        window = 1.0
    inverse_falloff = 1.0 / (1.0 + 0.12 * distance + 0.02 * distance * distance)
    return inverse_falloff * window


def shade_tile(scene: Scene, gbuffer: GBufferTile, shadow_maps: ShadowMaps) -> TileShading:
    pixel_count = gbuffer.width * gbuffer.height
    color = [0.0] * (pixel_count * 3)
    light_usage: set[str] = set()
    for local_y in range(gbuffer.height):
        for local_x in range(gbuffer.width):
            pixel_index = gbuffer.pixel_index(local_x, local_y)
            depth = gbuffer.depth[pixel_index]
            if depth == float("inf"):
                continue
            material_index = gbuffer.material_ids[pixel_index]
            material = scene.material(material_index)
            if material is None:
                continue
            normal = normalize(
                (
                    gbuffer.normal[pixel_index * 3],
                    gbuffer.normal[pixel_index * 3 + 1],
                    gbuffer.normal[pixel_index * 3 + 2],
                )
            )
            world_position = gbuffer.world_at(scene, local_x, local_y)
            albedo = (
                gbuffer.albedo[pixel_index * 3],
                gbuffer.albedo[pixel_index * 3 + 1],
                gbuffer.albedo[pixel_index * 3 + 2],
            )
            radiance_r = albedo[0] * scene.ambient + material.emissive[0]
            radiance_g = albedo[1] * scene.ambient + material.emissive[1]
            radiance_b = albedo[2] * scene.ambient + material.emissive[2]

            for light in scene.lights:
                if light.kind == "directional":
                    light_vector = normalize((-light.direction[0], -light.direction[1], -light.direction[2]))
                    attenuation = 1.0
                    distance = 0.0
                else:
                    to_light = sub(light.position, world_position)
                    distance = length(to_light)
                    if distance <= 1.0e-12:
                        light_vector = normal
                    else:
                        light_vector = normalize(to_light)
                    attenuation = _point_attenuation(distance, light.range)
                normal_dot_light = dot(normal, light_vector)
                if normal_dot_light <= 0.0 or attenuation <= 0.0:
                    continue
                light_usage.add(light.light_id)
                visibility = shadow_visibility(
                    scene,
                    light,
                    shadow_maps,
                    world_position,
                    normal,
                    normal_dot_light,
                )
                contribution = normal_dot_light * attenuation * visibility * light.intensity
                radiance_r += albedo[0] * light.color[0] * contribution
                radiance_g += albedo[1] * light.color[1] * contribution
                radiance_b += albedo[2] * light.color[2] * contribution

            color[pixel_index * 3] = min(1.0, max(0.0, radiance_r))
            color[pixel_index * 3 + 1] = min(1.0, max(0.0, radiance_g))
            color[pixel_index * 3 + 2] = min(1.0, max(0.0, radiance_b))
    ordered_light_ids = tuple(light.light_id for light in scene.lights if light.light_id in light_usage)
    return TileShading(color=color, light_usage=ordered_light_ids)


def compose_frame(
    scene: Scene,
    tiles: list[GBufferTile],
    shading: dict[tuple[int, int], TileShading],
) -> list[float]:
    frame = [
        component
        for _pixel in range(scene.width * scene.height)
        for component in scene.background
    ]
    for tile in tiles:
        tile_shading = shading.get((tile.x, tile.y))
        if tile_shading is None:
            continue
        for local_y in range(tile.height):
            frame_y = tile.y + local_y
            for local_x in range(tile.width):
                frame_x = tile.x + local_x
                frame_index = (frame_y * scene.width + frame_x) * 3
                tile_index = tile.pixel_index(local_x, local_y) * 3
                frame[frame_index] = tile_shading.color[tile_index]
                frame[frame_index + 1] = tile_shading.color[tile_index + 1]
                frame[frame_index + 2] = tile_shading.color[tile_index + 2]
    return frame


def add_light_state(state: dict[str, object], light: Light) -> None:
    state[light.light_id] = {
        "type": light.kind,
        "color": list(light.color),
        "intensity": light.intensity,
        "position": list(light.position),
        "direction": list(light.direction),
        "range": light.range,
    }
