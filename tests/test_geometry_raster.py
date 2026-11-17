from __future__ import annotations

import numpy as np

from terrain_softrast.config import (
    Camera,
    CascadeSettings,
    DirectionalLight,
    PointLight,
    RenderSettings,
)
from terrain_softrast.geometry import build_geometry
from terrain_softrast.math3d import clip_to_frustum, to_homogeneous
from terrain_softrast.renderer import render_scene
from terrain_softrast.samples import generate_weight_maps


def test_heightfield_normals_scale_and_weight_normalization():
    height = np.arange(25, dtype=np.float64).reshape(5, 5) / 10.0
    weights = np.zeros((2, 5, 5), dtype=np.float64)
    weights[0] = 2.0
    weights[1] = 1.0
    geometry = build_geometry(height, weights, unit_scale=0.5, height_scale=3.0)
    assert geometry.vertices.shape == (5, 5, 3)
    assert np.allclose(geometry.weights[..., 0], 2.0 / 3.0)
    assert np.allclose(geometry.weights[..., 1], 1.0 / 3.0)
    assert np.all(np.isclose(np.linalg.norm(geometry.normals, axis=-1), 1.0))
    assert np.all(geometry.normals[..., 1] > 0.0)
    assert "MATERIAL_WEIGHTS_NOT_NORMALIZED" in geometry.warnings.codes


def test_homogeneous_frustum_clipping_behind_partial_and_outside():
    partial = np.asarray(
        [
            (-0.2, -0.2, -0.5, 1.0),
            (0.0, 0.0, -2.0, 1.0),
            (0.5, -0.2, -0.5, 1.0),
        ],
        dtype=np.float64,
    )
    clipped, _ = clip_to_frustum(partial)
    assert clipped.shape[0] >= 4
    assert np.all(clipped[:, 3] > 0.0)
    assert np.all(clipped[:, 2] + clipped[:, 3] >= -1e-12)

    behind = np.asarray(
        [
            (-0.2, -0.2, -2.0, 1.0),
            (0.0, 0.0, -3.0, 1.0),
            (0.5, -0.2, -2.5, 1.0),
        ],
        dtype=np.float64,
    )
    clipped_behind, _ = clip_to_frustum(behind)
    assert clipped_behind.shape[0] == 0


def test_shared_edges_have_no_holes_and_flat_depth():
    size = 5
    height = np.zeros((size, size), dtype=np.float64)
    weights = generate_weight_maps(height, 4, 99)
    camera = Camera(eye=(0.0, 10.0, 0.0), target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0), fov_y_deg=45.0)
    settings = RenderSettings(
        width=64,
        height=64,
        tile_size=16,
        cascades=CascadeSettings(count=1, shadow_size=64),
    )
    result = render_scene(
        height,
        weights,
        camera=camera,
        directional_light=DirectionalLight(direction=(-0.3, -1.0, -0.2)),
        point_lights=(PointLight((-2.0, 2.0, 0.0)), PointLight((2.0, 2.0, 0.0))),
        settings=settings,
        unit_scale=0.5,
        height_scale=1.0,
    )
    finite = np.isfinite(result.depth_linear)
    assert np.any(finite)
    rows, columns = np.nonzero(finite)
    y0, y1 = int(rows.min()), int(rows.max() + 1)
    x0, x1 = int(columns.min()), int(columns.max() + 1)
    assert np.all(finite[y0:y1, x0:x1])
    center = result.depth_linear[y0:y1, x0:x1]
    assert np.max(np.abs(center - center[0, 0])) < 1e-4
    finite_visibility = result.visibility[np.isfinite(result.depth_linear)]
    assert np.all(finite_visibility > 0.999)


def test_repeated_render_arrays_are_identical(deterministic_height, deterministic_weights):
    settings = RenderSettings(
        width=96,
        height=72,
        tile_size=24,
        cascades=CascadeSettings(count=2, shadow_size=64),
    )
    camera = Camera(eye=(0.0, 5.0, 7.0), target=(0.0, 0.1, 0.0), far=40.0)
    first = render_scene(
        deterministic_height,
        deterministic_weights,
        camera=camera,
        settings=settings,
        unit_scale=0.3,
        height_scale=2.0,
    )
    second = render_scene(
        deterministic_height,
        deterministic_weights,
        camera=camera,
        settings=settings,
        unit_scale=0.3,
        height_scale=2.0,
    )
    assert np.array_equal(first.color_u8, second.color_u8)
    assert np.array_equal(first.depth_u16, second.depth_u16)
    assert np.array_equal(first.shadow_u8, second.shadow_u8)
    assert np.array_equal(first.color_linear, second.color_linear)
    assert np.array_equal(first.depth_linear, second.depth_linear, equal_nan=True)
    assert first.manifest["summary_sha256"] == second.manifest["summary_sha256"]
