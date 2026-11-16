from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .geometry import barycentric_coordinates, perspective_weights
from .mathutil import normalize


GBuffer = Dict[str, np.ndarray]
ShadeCallback = Callable[[GBuffer], np.ndarray]


def _triangle_bounds(
    screen: np.ndarray, x0: int, y0: int, x1: int, y1: int
) -> Tuple[int, int, int, int]:
    minimum = np.min(screen, axis=0)
    maximum = np.max(screen, axis=0)
    xa = max(x0, int(np.ceil(float(minimum[0]) - 0.5)))
    xb = min(x1, int(np.floor(float(maximum[0]) - 0.5)) + 1)
    ya = max(y0, int(np.ceil(float(minimum[1]) - 0.5)))
    yb = min(y1, int(np.floor(float(maximum[1]) - 0.5)) + 1)
    return xa, xb, ya, yb


def _rasterize_into_tile(
    triangles: List[Dict[str, object]],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    far: float,
) -> Tuple[np.ndarray, GBuffer]:
    tile_height = y1 - y0
    tile_width = x1 - x0
    depth = np.full((tile_height, tile_width), np.inf, dtype=np.float64)
    position = np.zeros((tile_height, tile_width, 3), dtype=np.float64)
    normal = np.zeros((tile_height, tile_width, 3), dtype=np.float64)
    uv = np.zeros((tile_height, tile_width, 2), dtype=np.float64)
    material = np.full((tile_height, tile_width), -1, dtype=np.int64)

    for triangle in triangles:
        screen = np.asarray(triangle["screen"], dtype=np.float64)
        xa, xb, ya, yb = _triangle_bounds(screen, x0, y0, x1, y1)
        if xa >= xb or ya >= yb:
            continue
        local_x = np.arange(xa - x0, xb - x0, dtype=np.float64) + 0.5 + x0
        local_y = np.arange(ya - y0, yb - y0, dtype=np.float64) + 0.5 + y0
        pixel_x, pixel_y = np.meshgrid(local_x, local_y)
        weights = barycentric_coordinates(screen, pixel_x, pixel_y)
        inside = np.all(weights >= -1.0e-10, axis=-1)
        if not np.any(inside):
            continue
        triangle_depth = np.asarray(triangle["depth"], dtype=np.float64)
        inverse_depth = np.sum(weights * (1.0 / triangle_depth), axis=-1)
        pixel_depth = 1.0 / np.maximum(inverse_depth, 1.0e-30)
        valid = (
            inside
            & (pixel_depth >= 1.0e-12)
            & (pixel_depth <= far)
        )
        if not np.any(valid):
            continue
        ys = slice(ya - y0, yb - y0)
        xs = slice(xa - x0, xb - x0)
        current_depth = depth[ys, xs]
        update = valid & (pixel_depth < current_depth)
        if not np.any(update):
            continue
        attributes = perspective_weights(weights, triangle_depth)
        triangle_world = np.asarray(triangle["world"], dtype=np.float64)
        triangle_normal = np.asarray(triangle["normal"], dtype=np.float64)
        triangle_uv = np.asarray(triangle["uv"], dtype=np.float64)
        interpolated_position = np.sum(
            attributes[..., None] * triangle_world[None, None, :, :], axis=2
        )
        interpolated_normal = normalize(
            np.sum(attributes[..., None] * triangle_normal[None, None, :, :], axis=2)
        )
        interpolated_uv = np.sum(
            attributes[..., None] * triangle_uv[None, None, :, :], axis=2
        )

        tile_depth = depth[ys, xs]
        tile_depth[update] = pixel_depth[update]
        position_view = position[ys, xs]
        position_view[update] = interpolated_position[update]
        normal_view = normal[ys, xs]
        normal_view[update] = interpolated_normal[update]
        uv_view = uv[ys, xs]
        uv_view[update] = interpolated_uv[update]
        material_view = material[ys, xs]
        material_view[update] = int(triangle["material"])
        depth[ys, xs] = tile_depth
        position[ys, xs] = position_view
        normal[ys, xs] = normal_view
        uv[ys, xs] = uv_view
        material[ys, xs] = material_view

    return depth, {
        "position": position,
        "normal": normal,
        "uv": uv,
        "material": material,
    }


def render_blocks(
    triangles: List[Dict[str, object]],
    width: int,
    height: int,
    block_size: int,
    far: float,
    shade_tile: ShadeCallback,
    background: Optional[np.ndarray] = None,
    blocks: Optional[Sequence[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if background is None:
        background_value = np.zeros(3, dtype=np.float64)
    else:
        background_value = np.asarray(background, dtype=np.float64)
    color = np.empty((height, width, 3), dtype=np.float64)
    color[:] = background_value
    full_depth = np.full((height, width), np.inf, dtype=np.float64)

    if blocks is None:
        block_coordinates = [
            (x0, y0)
            for y0 in range(0, height, block_size)
            for x0 in range(0, width, block_size)
        ]
    else:
        block_coordinates = sorted(set(blocks))
    block_set = set(block_coordinates)
    block_triangles: Dict[Tuple[int, int], List[Dict[str, object]]] = {
        block: [] for block in block_coordinates
    }
    for triangle in triangles:
        screen = np.asarray(triangle["screen"], dtype=np.float64)
        xa, xb, ya, yb = _triangle_bounds(screen, 0, 0, width, height)
        if xa >= xb or ya >= yb:
            continue
        first_x = xa // block_size
        last_x = (xb - 1) // block_size
        first_y = ya // block_size
        last_y = (yb - 1) // block_size
        for block_y in range(first_y, last_y + 1):
            for block_x in range(first_x, last_x + 1):
                block = (block_x * block_size, block_y * block_size)
                if block in block_set:
                    block_triangles[block].append(triangle)
    for x0, y0 in block_coordinates:
        x1 = min(width, x0 + block_size)
        y1 = min(height, y0 + block_size)
        tile_depth, gbuffer = _rasterize_into_tile(
            block_triangles[(x0, y0)], x0, y0, x1, y1, far
        )
        tile_color = np.asarray(shade_tile(gbuffer), dtype=np.float64)
        if tile_color.shape != (y1 - y0, x1 - x0, 3):
            raise RuntimeError("shade callback returned an invalid tile shape")
        color[y0:y1, x0:x1] = tile_color
        full_depth[y0:y1, x0:x1] = tile_depth
    return color, full_depth


def touched_blocks(
    triangles: List[Dict[str, object]],
    width: int,
    height: int,
    block_size: int,
) -> List[Tuple[int, int]]:
    blocks = set()
    for triangle in triangles:
        screen = np.asarray(triangle["screen"], dtype=np.float64)
        xa, xb, ya, yb = _triangle_bounds(screen, 0, 0, width, height)
        if xa >= xb or ya >= yb:
            continue
        first_x = xa // block_size
        last_x = (xb - 1) // block_size
        first_y = ya // block_size
        last_y = (yb - 1) // block_size
        for block_y in range(first_y, last_y + 1):
            for block_x in range(first_x, last_x + 1):
                blocks.add((block_x * block_size, block_y * block_size))
    return sorted(blocks)


def rasterize_depth_only(
    triangles: List[Dict[str, object]],
    width: int,
    height: int,
    far: float,
) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.full((height, width), np.inf, dtype=np.float64)
    radial = np.full((height, width), np.inf, dtype=np.float64)
    for triangle in triangles:
        screen = np.asarray(triangle["screen"], dtype=np.float64)
        xa, xb, ya, yb = _triangle_bounds(screen, 0, 0, width, height)
        if xa >= xb or ya >= yb:
            continue
        pixel_x, pixel_y = np.meshgrid(
            np.arange(xa, xb, dtype=np.float64) + 0.5,
            np.arange(ya, yb, dtype=np.float64) + 0.5,
        )
        weights = barycentric_coordinates(screen, pixel_x, pixel_y)
        inside = np.all(weights >= -1.0e-10, axis=-1)
        if not np.any(inside):
            continue
        triangle_depth = np.asarray(triangle["depth"], dtype=np.float64)
        inverse_depth = np.sum(weights * (1.0 / triangle_depth), axis=-1)
        pixel_depth = 1.0 / np.maximum(inverse_depth, 1.0e-30)
        valid = inside & (pixel_depth >= 1.0e-12) & (pixel_depth <= far)
        update = valid & (pixel_depth < depth[ya:yb, xa:xb])
        if not np.any(update):
            continue
        attributes = perspective_weights(weights, triangle_depth)
        world = np.asarray(triangle["world"], dtype=np.float64)
        interpolated_world = np.sum(
            attributes[..., None] * world[None, None, :, :], axis=2
        )
        radial_distance = np.sqrt(
            np.sum((interpolated_world - np.asarray(triangle["light_position"])) ** 2, axis=-1)
        )
        depth_view = depth[ya:yb, xa:xb]
        radial_view = radial[ya:yb, xa:xb]
        depth_view[update] = pixel_depth[update]
        radial_view[update] = radial_distance[update]
        depth[ya:yb, xa:xb] = depth_view
        radial[ya:yb, xa:xb] = radial_view
    return depth, radial
