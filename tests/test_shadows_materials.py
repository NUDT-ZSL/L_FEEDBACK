from __future__ import annotations

import numpy as np

from terrain_softrast.config import (
    Camera,
    CascadeSettings,
    DirectionalLight,
    Material,
    PointLight,
    RenderSettings,
)
from terrain_softrast.renderer import render_scene
from terrain_softrast.samples import generate_weight_maps
from terrain_softrast.shade import material_layer_colors
from terrain_softrast.shadows import build_cascades, cascade_weights


def test_cascades_increase_and_overlap_blends_two_levels(ridge_case):
    cascades = build_cascades(
        ridge_case["camera"],
        ridge_case["light"],
        ridge_case["settings"].width,
        ridge_case["settings"].height,
        ridge_case["settings"].cascades,
    )
    assert len(cascades) == 3
    splits = [cascade.split_far for cascade in cascades]
    assert splits == sorted(splits)
    assert splits[-1] == ridge_case["camera"].far
    first = cascades[0]
    overlap = (first.split_far - first.split_near) * ridge_case["settings"].cascades.overlap_fraction
    depth = np.asarray([first.split_far - overlap * 0.5])
    weights = cascade_weights(depth, cascades, ridge_case["settings"].cascades.overlap_fraction)
    assert weights.shape == (1, 3)
    assert np.isclose(weights.sum(), 1.0)
    assert np.count_nonzero(weights[0] > 1e-12) == 2


def test_ridge_casts_visible_shadow_and_cascades_are_recorded(ridge_case):
    result = render_scene(
        ridge_case["height"],
        ridge_case["weights"],
        camera=ridge_case["camera"],
        directional_light=ridge_case["light"],
        settings=ridge_case["settings"],
        unit_scale=ridge_case["unit_scale"],
        height_scale=ridge_case["height_scale"],
    )
    finite = np.isfinite(result.depth_linear)
    visible_pixels = result.visibility[finite]
    shadow_fraction = float(np.mean(visible_pixels < 0.5))
    assert 0.04 < shadow_fraction < 0.75
    assert np.all((result.visibility >= 0.0) & (result.visibility <= 1.0))
    assert result.manifest["cascades"]["count"] == 3
    assert len(result.manifest["cascades"]["levels"]) == 3
    assert all(level["shadow_size"] == [128, 128] for level in result.manifest["cascades"]["levels"])


def test_single_cell_ridge_preserves_thin_shadow_detail():
    height = np.zeros((33, 33), dtype=np.float64)
    height[:, 16] = 1.2
    weights = generate_weight_maps(height, 4, 17)
    settings = RenderSettings(
        width=128,
        height=96,
        tile_size=16,
        cascades=CascadeSettings(count=2, shadow_size=128, overlap_fraction=0.12),
    )
    camera = Camera(eye=(0.0, 10.0, 13.0), target=(0.0, 0.0, 0.0), fov_y_deg=50.0, far=80.0)
    light = DirectionalLight(direction=(-1.0, -0.25, -0.08), intensity=1.2)
    result = render_scene(
        height,
        weights,
        camera=camera,
        directional_light=light,
        settings=settings,
        unit_scale=0.35,
        height_scale=2.0,
    )
    finite = np.isfinite(result.depth_linear)
    shadow_fraction = float(np.mean(result.visibility[finite] < 0.5))
    partial_fraction = float(np.mean((result.visibility[finite] > 0.01) & (result.visibility[finite] < 0.99)))
    assert shadow_fraction > 0.01
    assert partial_fraction > 0.005


def test_unnormalized_weights_emit_business_warning(deterministic_height, deterministic_weights):
    settings = RenderSettings(width=64, height=48, tile_size=16)
    camera = Camera(eye=(0.0, 5.0, 7.0), target=(0.0, 0.0, 0.0))
    result = render_scene(
        deterministic_height,
        deterministic_weights * 3.7,
        camera=camera,
        settings=settings,
        unit_scale=0.3,
        height_scale=2.0,
    )
    assert "MATERIAL_WEIGHTS_NOT_NORMALIZED" in result.manifest["warning_codes"]
    assert any(
        warning["code"] == "MATERIAL_WEIGHTS_NOT_NORMALIZED"
        for warning in result.manifest["warnings"]
    )


def test_material_blend_equals_weighted_one_hot_layers():
    materials = (
        Material("a", (0.2, 0.3, 0.1), 0.8, 0.15, 1.2, 0.0),
        Material("b", (0.5, 0.4, 0.3), 0.55, 0.25, 2.1, 0.7),
        Material("c", (0.7, 0.75, 0.8), 0.65, 0.05, 0.9, 1.3),
    )
    world = np.asarray(
        [
            (-1.0, 0.0, 0.5),
            (0.0, 0.2, -1.0),
            (1.0, -0.1, 1.2),
            (0.4, 0.6, 0.1),
        ],
        dtype=np.float64,
    )
    normals = np.tile(np.asarray((0.0, 1.0, 0.0)), (world.shape[0], 1))
    visibility = np.asarray((1.0, 0.7, 0.4, 0.9))
    camera = Camera(eye=(0.0, 5.0, 6.0), target=(0.0, 0.0, 0.0))
    settings = RenderSettings()
    points = (PointLight((-2.0, 2.0, 0.0)), PointLight((2.0, 1.5, 1.0)))
    layer_colors, _ = material_layer_colors(
        world,
        normals,
        visibility,
        materials,
        camera,
        DirectionalLight(),
        points,
        settings,
    )
    weights = np.asarray(
        [
            (0.2, 0.5, 0.3),
            (0.6, 0.1, 0.3),
            (0.15, 0.25, 0.6),
            (0.34, 0.33, 0.33),
        ],
        dtype=np.float64,
    )
    blended = np.sum(layer_colors * weights[:, :, None], axis=1)
    one_hot_sum = np.zeros_like(blended)
    for index in range(len(materials)):
        one_hot = np.zeros_like(weights)
        one_hot[:, index] = 1.0
        one_hot_sum += np.sum(layer_colors * one_hot[:, :, None], axis=1) * weights[:, index : index + 1]
    assert np.allclose(blended, one_hot_sum, rtol=0.0, atol=1e-14)
