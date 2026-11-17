from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "deferred_renderer", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
        timeout=120,
        check=False,
    )


def write_scene(path: Path) -> dict:
    payload = run_cli("--write-default-scene", str(path))
    assert payload.returncode == 0, payload.stderr
    return json.loads(path.read_text())


def test_headless_pipeline_statistics_and_cpu_reference(tmp_path: Path) -> None:
    scene = tmp_path / "scene.json"
    write_scene(scene)
    output = tmp_path / "headless"
    result = run_cli(
        "--scene",
        str(scene),
        "--out",
        str(output),
        "--mode",
        "headless",
        "--width",
        "96",
        "--height",
        "54",
    )
    assert result.returncode == 0, result.stderr
    stats = json.loads((output / "stats.json").read_text())
    assert stats["ok"] is True
    assert stats["resolution"] == {"width": 96, "height": 54}

    attachments = {item["name"]: item for item in stats["g_buffer"]["attachments"]}
    assert set(attachments) == {
        "albedo_roughness",
        "normal_side",
        "linear_depth",
        "material_id",
    }
    assert attachments["normal_side"]["format"] == "RGBA16F"
    assert attachments["linear_depth"]["format"] == "R32F"
    assert attachments["material_id"]["format"] == "R32UI"

    measured = stats["g_buffer"]["measured"]
    assert measured["valid_geometry_pixels"] > 0
    assert measured["linear_depth_geometry"]["min"] >= stats["scene"]["camera"]["near"]
    assert measured["linear_depth_geometry"]["max"] <= stats["scene"]["camera"]["far"]
    assert measured["normal_length"]["max"] <= 1.001
    assert measured["front_side_pixels"] > 0
    assert measured["back_side_pixels"] > 0
    assert measured["material_ids_present"] == [1, 2, 3, 4, 5, 6]

    tiling = stats["tile_lighting"]
    assert tiling["tile_size"] == 16
    assert len(tiling["lists"]) == tiling["tile_count"]
    assert tiling["average_lights_per_tile"] >= 0.0
    assert tiling["max_lights_in_tile"] <= stats["scene"]["light_count"]
    assert sum(tiling["light_count_histogram"].values()) == tiling["tile_count"]

    reference = stats["reference_check"]
    assert reference["passed"] is True
    assert reference["max_abs_error"] <= reference["threshold_abs"]

    dof = stats["dof"]
    assert dof["composition_order"] == ["background", "foreground"]
    assert len(dof["bin_edges_px"]) == len(dof["all_pixel_counts"]) + 1
    assert sum(dof["all_pixel_counts"]) == 96 * 54
    assert "pixel_sha256" in stats["output"]

    image = np.asarray(Image.open(output / "frame_0000.png"))
    assert image.shape == (54, 96, 4)


def test_repeat_and_window_rendering_are_pixel_identical(tmp_path: Path) -> None:
    pytest.importorskip("glfw")
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("window parity requires an X11 or Wayland display")

    scene = tmp_path / "scene.json"
    write_scene(scene)
    outputs = [tmp_path / "repeat_a", tmp_path / "repeat_b", tmp_path / "window"]
    commands = [
        ("--mode", "headless"),
        ("--mode", "headless"),
        ("--mode", "window", "--window-hidden"),
    ]
    hashes: list[str] = []
    for output, mode_args in zip(outputs, commands, strict=True):
        result = run_cli(
            "--scene",
            str(scene),
            "--out",
            str(output),
            "--width",
            "80",
            "--height",
            "48",
            *mode_args,
        )
        if result.returncode == 3 and "E_WINDOW_CREATE" in result.stderr:
            pytest.skip("GLFW cannot create a window in this environment")
        assert result.returncode == 0, result.stderr
        stats = json.loads((output / "stats.json").read_text())
        hashes.append(stats["output"]["pixel_sha256"])
    assert len(set(hashes)) == 1
    images = [np.asarray(Image.open(output / "frame_0000.png")) for output in outputs]
    assert np.array_equal(images[0], images[1])
    assert np.array_equal(images[0], images[2])

