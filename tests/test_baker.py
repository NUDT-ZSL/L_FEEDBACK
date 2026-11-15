"""End-to-end and unit tests for the terrain baker."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from terrain_baker import cache, errors  # noqa: E402
from terrain_baker.baker import run_bake  # noqa: E402
from terrain_baker.scene import load_scene  # noqa: E402

SAMPLE_SCENE = os.path.join(REPO_ROOT, "samples", "scene.json")


def make_scene_dict(**overrides):
    base = {
        "seed": 42,
        "resolution": {"width": 128, "height": 96},
        "tile_size": 32,
        "normal_strength": 1.2,
        "materials": ["grass", "rock", "sand"],
        "layers": [
            {"material_id": "grass", "tile_scale": 4.0, "rotation_deg": 0.0,
             "blend_mode": "over", "height_scale": 1.0, "opacity": 1.0},
            {"material_id": "rock", "tile_scale": 8.0, "rotation_deg": 30.0,
             "blend_mode": "add", "height_scale": 0.5, "opacity": 0.8},
            {"material_id": "sand", "tile_scale": 5.0, "rotation_deg": -15.0,
             "blend_mode": "multiply", "height_scale": 0.3, "opacity": 0.6},
        ],
    }
    base.update(overrides)
    return base


def write_scene(tmp_path, scene_dict):
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(scene_dict), encoding="utf-8")
    return str(path)


def bake(tmp_path, scene_dict, out_name="out", tile_size=None):
    from dataclasses import replace
    scene = load_scene(write_scene(tmp_path, scene_dict))
    if tile_size is not None:
        scene = replace(scene, tile_size=tile_size)
    out_dir = tmp_path / out_name
    manifest = run_bake(scene, str(out_dir))
    return manifest, out_dir


def read_png(out_dir, name):
    return np.array(Image.open(out_dir / name))


def test_outputs_and_manifest(tmp_path):
    manifest, out_dir = bake(tmp_path, make_scene_dict())
    for name in ("diffuse.png", "normal.png", "weights.png", "manifest.json"):
        assert (out_dir / name).is_file(), name
    diffuse = read_png(out_dir, "diffuse.png")
    normal = read_png(out_dir, "normal.png")
    weights = read_png(out_dir, "weights.png")
    assert diffuse.shape == (96, 128, 3)
    assert normal.shape == (96, 128, 3)
    assert weights.shape == (96, 128, 4)
    # Normal z channel must be non-negative (>= 128 in 8-bit encoding).
    assert normal[:, :, 2].min() >= 128
    # Weight channels sum to 255 per pixel (8-bit normalized).
    sums = weights.astype(np.int64).sum(axis=-1)
    assert np.abs(sums - 255).max() <= 2
    # Manifest contents.
    assert manifest["generator"].startswith("terrain-baker/")
    assert "time" not in json.dumps(manifest).lower()
    assert len(manifest["layers"]) == 3
    for entry in manifest["layers"]:
        assert entry["min_height"] <= entry["max_height"]
        assert len(entry["mean_albedo"]) == 3
    for key in ("diffuse", "normal", "weights"):
        assert len(manifest["outputs"][key]["sha256"]) == 64
    assert 0.9 < manifest["normal_correction_factor"] < 1.1


def test_determinism(tmp_path):
    m1, d1 = bake(tmp_path, make_scene_dict(), out_name="a")
    m2, d2 = bake(tmp_path, make_scene_dict(), out_name="b")
    assert m1 == m2
    for name in ("diffuse.png", "normal.png", "weights.png"):
        assert (d1 / name).read_bytes() == (d2 / name).read_bytes()


def test_chunked_matches_full(tmp_path):
    scene_dict = make_scene_dict()
    manifest_full, dir_full = bake(tmp_path, scene_dict, out_name="full",
                                   tile_size=4096)
    manifest_tiled, dir_tiled = bake(tmp_path, scene_dict, out_name="tiled",
                                     tile_size=32)
    for name in ("diffuse.png", "normal.png", "weights.png"):
        a = read_png(dir_full, name)
        b = read_png(dir_tiled, name)
        assert np.array_equal(a, b), f"{name} differs between chunk sizes"
    # Manifest stats are rounded, so they must match exactly too.
    assert manifest_full["layers"] == manifest_tiled["layers"]
    assert (manifest_full["normal_correction_factor"]
            == manifest_tiled["normal_correction_factor"])
    assert manifest_full["outputs"] == manifest_tiled["outputs"]


def test_non_square_chunked_matches_full(tmp_path):
    scene_dict = make_scene_dict()
    scene_dict["resolution"] = {"width": 150, "height": 110}
    _, dir_full = bake(tmp_path, scene_dict, out_name="full", tile_size=4096)
    _, dir_tiled = bake(tmp_path, scene_dict, out_name="tiled", tile_size=48)
    for name in ("diffuse.png", "normal.png", "weights.png"):
        assert np.array_equal(read_png(dir_full, name),
                              read_png(dir_tiled, name)), name


def test_cache_roundtrip_and_reuse(tmp_path):
    manifest1, out_dir = bake(tmp_path, make_scene_dict())
    cache_dir = out_dir / ".tbcache"
    assert (cache_dir / "height.tbc").is_file()
    assert (cache_dir / "normals.tbc").is_file()
    # Second run hits the cache and must produce identical outputs.
    manifest2, _ = bake(tmp_path, make_scene_dict())
    assert manifest1 == manifest2


def test_cache_version_mismatch_rebuilds(tmp_path):
    manifest1, out_dir = bake(tmp_path, make_scene_dict())
    height_cache = out_dir / ".tbcache" / "height.tbc"
    raw = bytearray(height_cache.read_bytes())
    raw[8:12] = (0).to_bytes(4, "little")  # forge old version
    height_cache.write_bytes(bytes(raw))
    manifest2, _ = bake(tmp_path, make_scene_dict())
    assert manifest1 == manifest2  # rebuilt, not misread
    # Cache was rewritten with the current version.
    assert height_cache.read_bytes()[8:12] == cache.VERSION.to_bytes(4, "little")


def test_cache_crc_corruption_rebuilds(tmp_path):
    manifest1, out_dir = bake(tmp_path, make_scene_dict())
    normal_cache = out_dir / ".tbcache" / "normals.tbc"
    raw = bytearray(normal_cache.read_bytes())
    raw[-1] ^= 0xFF  # corrupt payload
    normal_cache.write_bytes(bytes(raw))
    manifest2, _ = bake(tmp_path, make_scene_dict())
    assert manifest1 == manifest2


def test_error_scene_missing(tmp_path):
    result = run_cli(tmp_path / "nope.json", tmp_path / "out")
    assert result.returncode == 2
    assert json.loads(result.stderr)["error"] == "E_SCENE_MISSING"
    assert not (tmp_path / "out").exists() or not os.listdir(tmp_path / "out")


def test_error_scene_invalid_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    result = run_cli(bad, tmp_path / "out")
    assert result.returncode == 3
    assert json.loads(result.stderr)["error"] == "E_SCENE_INVALID"


def test_error_unknown_material(tmp_path):
    scene = make_scene_dict()
    scene["layers"][0]["material_id"] = "lava"
    result = run_cli(write_scene(tmp_path, scene), tmp_path / "out")
    assert result.returncode == 4
    assert json.loads(result.stderr)["error"] == "E_UNKNOWN_MATERIAL"


def test_error_resolution_limit(tmp_path):
    scene = make_scene_dict()
    scene["resolution"] = {"width": 2048, "height": 2048}
    scene["max_pixels"] = 1024 * 1024
    result = run_cli(write_scene(tmp_path, scene), tmp_path / "out")
    assert result.returncode == 5
    assert json.loads(result.stderr)["error"] == "E_RESOLUTION_LIMIT"


def test_error_target_exists(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "stray.txt").write_text("not ours", encoding="utf-8")
    result = run_cli(write_scene(tmp_path, make_scene_dict()), out)
    assert result.returncode == 6
    assert json.loads(result.stderr)["error"] == "E_TARGET_EXISTS"
    assert (out / "stray.txt").is_file()  # untouched


def test_rebake_into_own_output_allowed(tmp_path):
    scene_path = write_scene(tmp_path, make_scene_dict())
    out = tmp_path / "out"
    assert run_cli(scene_path, out).returncode == 0
    assert run_cli(scene_path, out).returncode == 0  # marker present


def test_error_path_traversal(tmp_path):
    scene_path = write_scene(tmp_path, make_scene_dict())
    result = run_cli(scene_path, str(tmp_path / ".." / "escape"))
    assert result.returncode == 7
    assert json.loads(result.stderr)["error"] == "E_PATH_UNSAFE"


def test_error_path_symlink(tmp_path):
    scene_path = write_scene(tmp_path, make_scene_dict())
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    os.symlink(real, link)
    result = run_cli(scene_path, str(link / "out"))
    assert result.returncode == 7
    assert json.loads(result.stderr)["error"] == "E_PATH_UNSAFE"


def run_cli(scene_path, out_dir):
    return subprocess.run(
        [sys.executable, "-m", "terrain_baker", str(scene_path),
         "--output", str(out_dir)],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_success(tmp_path):
    scene_path = write_scene(tmp_path, make_scene_dict())
    out = tmp_path / "out"
    result = run_cli(scene_path, out)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "ok"
    assert (out / "manifest.json").is_file()
