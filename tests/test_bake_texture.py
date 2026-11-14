from __future__ import annotations

import json
import copy

import numpy as np
import pytest

from skin_bake.bake import bake_scene
from skin_bake.basis import build_bind_basis
from skin_bake.errors import BusinessError
from skin_bake.projection import (
    project_to_bind_basis,
    resample_projection,
    sample_input_curves,
)
from skin_bake.runtime import load_baked_asset
from skin_bake.scene import load_scene
from skin_bake.skinning import animation_frame_times, interpolate_curve
from skin_bake.texture import read_texture


def test_end_to_end_texture_mapping_and_report(scene_path, baked_dir):
    scene = load_scene(scene_path)
    report = json.loads((baked_dir / "report.json").read_text(encoding="utf-8"))
    atlas_map = json.loads((baked_dir / "atlas_map.json").read_text(encoding="utf-8"))
    texture = read_texture(baked_dir / "displacement.rg16f")

    assert report["status"] == "ok"
    assert report["inputFileSha256"] == scene.input_file_sha256
    assert report["fallbackVertices"] == [7]
    assert report["frameCount"] == 48
    assert report["actualSampleRate"] < report["baseSampleRate"]
    assert report["downsampleFactor"] < 1.0
    assert report["maxProjectionError"] < report["errorTolerance"]
    assert len(report["maxProjectionErrorByVertex"]) == scene.vertex_count
    assert texture.width == 128 and texture.height == 128
    assert texture.header["format"] == "skin-bake-rg16f"
    assert len(atlas_map["blocks"]) == scene.vertex_count

    basis = build_bind_basis(scene.normals, scene.tangents)
    times = animation_frame_times(scene)
    projection = project_to_bind_basis(
        scene, basis, times, sample_input_curves(scene, times)
    )
    expected = resample_projection(scene, projection, atlas_map["sampleCount"])
    for block in atlas_map["blocks"]:
        vertex = block["vertex"]
        px, py = block["pixelOffset"]
        width, height = block["blockSize"]
        payload = texture.pixels[
            py : py + height, px : px + width, :
        ].reshape(-1, 2)[: block["sampleCount"]]
        assert np.allclose(payload, expected[:, vertex, :].astype(np.float16), atol=0.0, rtol=0.0)


def test_padding_is_accumulated_and_outside_is_zero(scene_path, baked_dir):
    atlas_map = json.loads((baked_dir / "atlas_map.json").read_text(encoding="utf-8"))
    texture = read_texture(baked_dir / "displacement.rg16f")
    occupied = np.zeros((texture.height, texture.width), dtype=bool)
    for block in atlas_map["blocks"]:
        px, py = block["pixelOffset"]
        w, h = block["blockSize"]
        pad = atlas_map["packing"]["padding"]
        occupied[py - pad : py + h + pad, px - pad : px + w + pad] = True
        payload = texture.pixels[py : py + h, px : px + w].copy()
        assert np.allclose(texture.pixels[py - 1, px : px + w], payload[0])
        assert np.allclose(texture.pixels[py + h, px : px + w], payload[-1])
        assert np.allclose(texture.pixels[py : py + h, px - 1], payload[:, 0])
        assert np.allclose(texture.pixels[py : py + h, px + w], payload[:, -1])
    assert np.all(texture.pixels[~occupied] == 0.0)


def test_clamp_scene_extends_boundaries_and_reports_extra_frame(clamp_scene_path, tmp_path):
    output = tmp_path / "clamp_bake"
    bake_scene(clamp_scene_path, output)
    scene = load_scene(clamp_scene_path)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    atlas_map = json.loads((output / "atlas_map.json").read_text(encoding="utf-8"))
    assert scene.loop_mode == "clamp"
    assert report["frameCount"] == int(round(scene.duration * scene.fps)) + 1
    assert atlas_map["loopMode"] == "clamp"
    curve = scene.curves[0]
    before = interpolate_curve(
        curve.times, curve.vectors, -0.4, scene.duration, loop=False
    )
    after = interpolate_curve(
        curve.times, curve.vectors, scene.duration + 0.4, scene.duration, loop=False
    )
    assert np.array_equal(before, curve.vectors[0])
    assert np.array_equal(after, curve.vectors[-1])


def test_loop_endpoint_key_is_force_closed(tmp_path, scene_document):
    from skin_bake.scene import write_scene

    document = copy.deepcopy(scene_document)
    document["displacement"]["vertexCurves"][0]["keys"][-1]["offset"] = [9.0, 9.0, 0.0]
    path = tmp_path / "forced_loop.json"
    write_scene(path, document)
    scene = load_scene(path)
    curve = scene.curves[0]
    assert np.array_equal(curve.vectors[-1], curve.vectors[0])


def test_loop_displacement_interpolation_wraps_missing_end_key():
    times = np.array([0.2, 0.8])
    values = np.array([[1.0, 0.0], [3.0, 0.0]])
    middle = interpolate_curve(times, values, 0.5, 1.0, loop=True)
    before_start = interpolate_curve(times, values, 0.1, 1.0, loop=True)
    after_end = interpolate_curve(times, values, 0.9, 1.0, loop=True)
    assert np.allclose(middle, [2.0, 0.0])
    assert before_start[0] == pytest.approx(1.5)
    assert after_end[0] == pytest.approx(2.5)
    assert np.allclose(
        interpolate_curve(times, values, 0.0, 1.0, loop=True),
        interpolate_curve(times, values, 1.0, 1.0, loop=True),
    )


def test_texture_corruption_is_detected(scene_path, baked_dir):
    texture_path = baked_dir / "displacement.rg16f"
    data = bytearray(texture_path.read_bytes())
    data[-1] ^= 0x01
    texture_path.write_bytes(data)
    with pytest.raises(BusinessError) as error:
        read_texture(texture_path)
    assert error.value.code == "E_TEXTURE_CORRUPT"


def test_runtime_asset_decodes_all_vertices(scene_path, baked_dir):
    asset = load_baked_asset(scene_path, baked_dir)
    assert asset.frame_count == 48
    for frame in (0, 12, 47):
        displacement = asset.frame_displacements(asset.frame_time(frame))
        assert displacement.shape == (asset.scene.vertex_count, 2)
        assert np.all(np.isfinite(displacement))
    assert asset.bind_basis.fallback_vertices == (7,)


def test_absolute_integration_mode_bakes(tmp_path, scene_document):
    from skin_bake.scene import write_scene

    document = copy.deepcopy(scene_document)
    document["displacement"]["integration"] = "absolute"
    path = tmp_path / "absolute.json"
    write_scene(path, document)
    output = tmp_path / "absolute_bake"
    bake_scene(path, output)
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["precision"]["accumulation"] == "direct"
    assert report["status"] == "ok"


def test_atlas_capacity_failure_writes_nothing(tmp_path, scene_document):
    from skin_bake.scene import write_scene

    document = copy.deepcopy(scene_document)
    document["bake"]["atlasWidth"] = 4
    document["bake"]["atlasHeight"] = 4
    path = tmp_path / "tiny_atlas.json"
    write_scene(path, document)
    output = tmp_path / "no_atlas"
    with pytest.raises(BusinessError) as error:
        bake_scene(path, output)
    assert error.value.code == "E_ATLAS_CAPACITY"
    assert not output.exists()

