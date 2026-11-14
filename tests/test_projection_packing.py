from __future__ import annotations

import copy

import numpy as np
import pytest

from skin_bake.bake import bake_scene
from skin_bake.basis import build_bind_basis
from skin_bake.errors import BusinessError
from skin_bake.packing import actual_rate_for_count, block_payload_dimensions, plan_and_pack
from skin_bake.projection import (
    project_to_bind_basis,
    sample_input_curves,
)
from skin_bake.scene import compute_content_hash
from skin_bake.skinning import animation_frame_times


def _project(scene):
    basis = build_bind_basis(scene.normals, scene.tangents)
    times = animation_frame_times(scene)
    return basis, project_to_bind_basis(
        scene, basis, times, sample_input_curves(scene, times)
    )


def _assert_no_overlap(blocks, width, height):
    rectangles = []
    for block in blocks:
        x, y = block.x, block.y
        w, h = block.padded_width, block.padded_height
        assert x >= 0 and y >= 0 and x + w <= width and y + h <= height
        for other in rectangles:
            assert not (
                x < other[0] + other[2]
                and other[0] < x + w
                and y < other[1] + other[3]
                and other[1] < y + h
            )
        rectangles.append((x, y, w, h))


def test_projection_is_stable_and_loop_closes(scene_path):
    from skin_bake.scene import load_scene

    scene = load_scene(scene_path)
    basis, projection = _project(scene)
    assert basis.fallback_vertices == (7,)
    assert np.all(np.isfinite(projection.curves))
    assert projection.overall_max_relative_error < 1e-12
    assert projection.loop_closure_error == 0.0
    assert np.allclose(projection.curves[0], 0.0)


def test_projection_tolerance_failure_writes_nothing(tmp_path, scene_document):
    from skin_bake.scene import write_scene

    document = copy.deepcopy(scene_document)
    document["bake"]["errorTolerance"] = 1e-6
    document["displacement"]["vertexCurves"][0]["keys"][3]["offset"][2] = 0.01
    document["contentHash"] = compute_content_hash(document)
    scene_path = tmp_path / "bad_projection.json"
    write_scene(scene_path, document)
    output = tmp_path / "must_not_exist"
    with pytest.raises(BusinessError) as error:
        bake_scene(scene_path, output)
    assert error.value.code == "E_TOLERANCE_EXCEEDED"
    assert not output.exists()
    assert not any(tmp_path.glob(".must_not_exist.tmp-*"))


def test_maxrects_pack_is_deterministic_and_non_overlapping():
    y = np.linspace(0.0, 1.0, 36)
    curves = np.zeros((36, 24, 2), dtype=np.float64)
    curves[:, :, 0] = np.sin(y)[:, None] * np.linspace(0.1, 0.3, 24)[None, :]
    curves[:, :, 1] = np.cos(y)[:, None] * np.linspace(0.2, 0.5, 24)[None, :]
    kwargs = dict(
        curves=curves,
        duration=1.5,
        loop=True,
        texture_sample_rate=50.0,
        atlas_width=128,
        atlas_height=128,
        min_block_size=1,
        padding=1,
    )
    first = plan_and_pack(**kwargs)
    second = plan_and_pack(**kwargs)
    assert first == second
    assert len(first.blocks) == 24
    _assert_no_overlap(first.blocks, 128, 128)


def test_uniform_downsampling_fits_small_atlas():
    y = np.linspace(0.0, 1.0, 96)
    curves = np.zeros((96, 120, 2), dtype=np.float64)
    curves[:, :, 0] = np.sin(y * 8.0)[:, None] * 0.08
    curves[:, :, 1] = np.cos(y * 5.0)[:, None] * 0.06
    result = plan_and_pack(
        curves=curves,
        duration=2.0,
        loop=True,
        texture_sample_rate=120.0,
        atlas_width=96,
        atlas_height=96,
        min_block_size=1,
        padding=1,
    )
    assert result.attempts > 1
    assert result.downsample_factor < 1.0
    assert result.actual_sample_rate == pytest.approx(
        actual_rate_for_count(result.sample_count, 2.0, True)
    )
    assert all(block.payload_width * block.payload_height >= result.sample_count for block in result.blocks)
    _assert_no_overlap(result.blocks, 96, 96)


def test_extreme_curve_aspect_does_not_prevent_minimum_downsampling():
    width, height = block_payload_dimensions(1, 0.0, 1.0, 1)
    assert (width, height) == (1, 1)
    width, height = block_payload_dimensions(2, 0.0, 1.0, 1)
    assert (width, height) == (1, 2)


def test_atlas_capacity_error_is_stable():
    curves = np.zeros((20, 30, 2), dtype=np.float64)
    curves[..., 0] = 0.1
    with pytest.raises(BusinessError) as error:
        plan_and_pack(
            curves=curves,
            duration=1.0,
            loop=True,
            texture_sample_rate=30.0,
            atlas_width=8,
            atlas_height=8,
            min_block_size=1,
            padding=1,
        )
    assert error.value.code == "E_ATLAS_CAPACITY"

