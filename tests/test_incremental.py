from __future__ import annotations

import numpy as np

from terrain_softrast.config import Camera
from terrain_softrast.renderer import render_scene


def _assert_same_image(first, second):
    assert np.array_equal(first.color_u8, second.color_u8)
    assert np.array_equal(first.depth_u16, second.depth_u16)
    assert np.array_equal(first.shadow_u8, second.shadow_u8)
    assert np.array_equal(first.color_linear, second.color_linear)
    assert np.array_equal(first.depth_linear, second.depth_linear, equal_nan=True)
    assert np.array_equal(first.visibility, second.visibility)


def test_incremental_dirty_tiles_match_full_render(ridge_case):
    height = ridge_case["height"]
    weights = ridge_case["weights"]
    args = {
        "camera": ridge_case["camera"],
        "directional_light": ridge_case["light"],
        "settings": ridge_case["settings"],
        "unit_scale": ridge_case["unit_scale"],
        "height_scale": ridge_case["height_scale"],
    }
    baseline = render_scene(height, weights, **args)
    changed = height.copy()
    changed[6:11, 3:7] += 0.9
    dirty = (2, 5, 8, 12)
    incremental = render_scene(changed, weights, cache=baseline.cache, dirty_rect=dirty, **args)
    full = render_scene(changed, weights, **args)
    _assert_same_image(incremental, full)
    assert incremental.stats["incremental"]["mode"] == "incremental"
    assert incremental.stats["incremental"]["full_rebuild"] is False
    assert incremental.stats["incremental"]["rendered_tiles"] < incremental.stats["incremental"]["total_tiles"]
    assert incremental.stats["incremental"]["shadow_tiles_updated"] > 0


def test_incomplete_dirty_rect_falls_back_to_full(ridge_case):
    height = ridge_case["height"]
    weights = ridge_case["weights"]
    args = {
        "camera": ridge_case["camera"],
        "directional_light": ridge_case["light"],
        "settings": ridge_case["settings"],
        "unit_scale": ridge_case["unit_scale"],
        "height_scale": ridge_case["height_scale"],
    }
    baseline = render_scene(height, weights, **args)
    changed = height.copy()
    changed[4:8, 4:8] += 0.5
    changed[18:20, 18:20] += 0.2
    incremental = render_scene(
        changed,
        weights,
        cache=baseline.cache,
        dirty_rect=(3, 3, 9, 9),
        **args,
    )
    full = render_scene(changed, weights, **args)
    _assert_same_image(incremental, full)
    assert incremental.stats["incremental"]["full_rebuild"] is True
    assert "DIRTY_RECT_INCOMPLETE" in incremental.manifest["warning_codes"]


def test_camera_matrix_change_forces_full_rebuild(ridge_case):
    height = ridge_case["height"]
    weights = ridge_case["weights"]
    args = {
        "directional_light": ridge_case["light"],
        "settings": ridge_case["settings"],
        "unit_scale": ridge_case["unit_scale"],
        "height_scale": ridge_case["height_scale"],
    }
    baseline = render_scene(height, weights, camera=ridge_case["camera"], **args)
    changed = height.copy()
    changed[8:12, 8:12] += 0.5
    moved_camera = Camera(
        eye=(1.5, 8.5, 10.5),
        target=(0.0, 0.2, 0.0),
        fov_y_deg=50.0,
        far=80.0,
    )
    incremental = render_scene(
        changed,
        weights,
        camera=moved_camera,
        cache=baseline.cache,
        dirty_rect=(7, 7, 13, 13),
        **args,
    )
    full = render_scene(changed, weights, camera=moved_camera, **args)
    _assert_same_image(incremental, full)
    assert incremental.stats["incremental"]["full_rebuild"] is True
    assert incremental.stats["incremental"]["reason"] == "structural_signature_changed"


def test_geometry_scale_change_forces_full_rebuild(ridge_case):
    height = ridge_case["height"]
    weights = ridge_case["weights"]
    baseline = render_scene(
        height,
        weights,
        camera=ridge_case["camera"],
        directional_light=ridge_case["light"],
        settings=ridge_case["settings"],
        unit_scale=ridge_case["unit_scale"],
        height_scale=ridge_case["height_scale"],
    )
    result = render_scene(
        height,
        weights,
        camera=ridge_case["camera"],
        directional_light=ridge_case["light"],
        settings=ridge_case["settings"],
        unit_scale=ridge_case["unit_scale"],
        height_scale=ridge_case["height_scale"] * 1.5,
        cache=baseline.cache,
    )
    full = render_scene(
        height,
        weights,
        camera=ridge_case["camera"],
        directional_light=ridge_case["light"],
        settings=ridge_case["settings"],
        unit_scale=ridge_case["unit_scale"],
        height_scale=ridge_case["height_scale"] * 1.5,
    )
    _assert_same_image(result, full)
    assert result.stats["incremental"]["full_rebuild"] is True
    assert result.stats["incremental"]["reason"] == "structural_signature_changed"
