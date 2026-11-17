from __future__ import annotations

import math

import numpy as np

from deferred_renderer.culling import (
    build_tile_frustum,
    build_tile_lights,
    coc_radius_array,
    sphere_intersects_tile,
)
from deferred_renderer.math3d import reconstruct_world
from deferred_renderer.scene import default_scene_dict, parse_scene


def test_exact_sphere_frustum_vertex_contact() -> None:
    scene = parse_scene(default_scene_dict())
    frustum = build_tile_frustum(
        -0.4,
        0.1,
        -0.3,
        0.2,
        4.0,
        8.0,
        scene.inv_projection,
        scene.camera.inv_view,
    )
    assert sphere_intersects_tile(frustum.center, 0.01, frustum)
    far_away = frustum.center + np.array([1000.0, 1000.0, 1000.0])
    assert not sphere_intersects_tile(far_away, 1.0, frustum)

    vertex = frustum.vertices[0]
    outward = vertex - frustum.center
    outward /= np.linalg.norm(outward)
    outside = vertex + outward * 0.25
    assert not sphere_intersects_tile(outside, 0.1, frustum)
    assert sphere_intersects_tile(outside, 0.30, frustum)


def test_tile_lists_have_no_false_negatives_for_visible_pixels() -> None:
    scene = parse_scene(default_scene_dict())
    width, height = 64, 48
    xs = np.linspace(4.0, 11.0, width, dtype=np.float32)
    depth = np.broadcast_to(xs[None, :], (height, width)).copy()
    depth[: height // 6, : width // 5] = 0.0

    result = build_tile_lights(scene, depth)
    world = reconstruct_world(depth, scene.inv_projection, scene.camera.inv_view)
    tile_size = scene.render.tile_size
    by_index = {record["tile"]: set(record["lights"]) for record in result.tile_records}
    tiles_x = result.tiles_x

    for y in range(height):
        for x in range(width):
            if depth[y, x] <= 0.0:
                continue
            tile_index = (y // tile_size) * tiles_x + (x // tile_size)
            listed = by_index[tile_index]
            for light_index, light in enumerate(scene.lights):
                delta = world[y, x] - np.asarray(light.position)
                if float(np.dot(delta, delta)) <= light.radius * light.radius:
                    assert light_index in listed


def test_flat_depth_tiles_do_not_create_degenerate_frusta() -> None:
    scene = parse_scene(default_scene_dict())
    depth = np.full((8, 8), 5.0, dtype=np.float32)
    result = build_tile_lights(scene, depth)
    assert len(result.tile_records) == 1
    assert result.tile_records[0]["lights"]


def test_coc_uses_the_same_linear_depth_convention() -> None:
    scene = parse_scene(default_scene_dict())
    depth = np.array(
        [
            [0.0, scene.render.focus_distance],
            [scene.render.focus_distance * 0.5, scene.render.focus_distance * 2.0],
        ],
        dtype=np.float32,
    )
    radii = coc_radius_array(depth, scene.render)
    assert math.isclose(float(radii[0, 1]), 0.0, abs_tol=1e-5)
    assert float(radii[1, 0]) > 0.0
    assert float(radii[1, 1]) > 0.0
    assert math.isclose(
        float(radii[0, 0]),
        scene.render.max_blur_radius * 0.75,
        rel_tol=0.0,
        abs_tol=1e-6,
    )

