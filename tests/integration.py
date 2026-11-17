#!/usr/bin/env python3
"""End-to-end deterministic and error-path checks for the software rasterizer."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run(executable: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), *map(str, args)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_ppm(path: Path) -> tuple[int, int, bytes]:
    with path.open("rb") as stream:
        magic = stream.readline().strip()
        dimensions = stream.readline().split()
        maximum = stream.readline().strip()
        payload = stream.read()
    require(magic == b"P6", f"{path} is not binary PPM")
    require(maximum == b"255", f"{path} does not use 8-bit channels")
    require(len(dimensions) == 2, f"{path} has invalid dimensions")
    width, height = map(int, dimensions)
    require(len(payload) == width * height * 3, f"{path} payload size mismatch")
    return width, height, payload


def write_scene(path: Path, scene: dict) -> None:
    path.write_text(json.dumps(scene, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def expect_error(
    executable: Path,
    work: Path,
    name: str,
    scene_path: Path,
    expected_code: int,
    expected_name: str,
) -> None:
    output = work / f"{name}.ppm"
    stats = work / f"{name}.stats.json"
    for path in (output, stats):
        if path.exists():
            path.unlink()
    result = run(executable, scene_path, output, stats)
    require(result.returncode == expected_code,
            f"{name}: exit code {result.returncode}, expected {expected_code}")
    require(result.stdout == "", f"{name}: error path must not write to stdout")
    payload = json.loads(result.stderr)
    require(payload["error"]["code"] == expected_code, f"{name}: JSON error code mismatch")
    require(payload["error"]["name"] == expected_name, f"{name}: JSON error name mismatch")
    require(not output.exists(), f"{name}: a partial image was created")
    require(not stats.exists(), f"{name}: a partial statistics file was created")


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: integration.py <softraster-executable> <work-directory>", file=sys.stderr)
        return 2
    executable = Path(sys.argv[1]).resolve()
    work = Path(sys.argv[2]).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    sample_path = work / "sample.json"
    generated = run(executable, "--generate-sample", sample_path)
    require(generated.returncode == 0, f"sample generation failed: {generated.stderr}")
    require(sample_path.exists(), "sample generator did not write a scene")

    bad_usage = run(executable)
    require(bad_usage.returncode == 5, "invalid CLI usage did not return code 5")
    require(bad_usage.stdout == "", "invalid CLI usage wrote to stdout")
    require(json.loads(bad_usage.stderr)["error"]["name"] == "INVALID_VALUE",
            "invalid CLI usage did not report INVALID_VALUE")

    first_ppm = work / "first.ppm"
    first_stats = work / "first.stats.json"
    second_ppm = work / "second.ppm"
    second_stats = work / "second.stats.json"
    first_run = run(executable, sample_path, first_ppm, first_stats)
    second_run = run(executable, sample_path, second_ppm, second_stats)
    require(first_run.returncode == 0, f"first render failed: {first_run.stderr}")
    require(second_run.returncode == 0, f"second render failed: {second_run.stderr}")
    require(first_run.stdout == "" and first_run.stderr == "",
            "successful render must be quiet")
    require(first_ppm.read_bytes() == second_ppm.read_bytes(),
            "repeated runs produced different PPM bytes")
    require(first_stats.read_bytes() == second_stats.read_bytes(),
            "repeated runs produced different statistics JSON")

    width, height, payload = parse_ppm(first_ppm)
    require((width, height) == (320, 320), "sample dimensions are not 320x320")
    stats = json.loads(first_stats.read_text(encoding="utf-8"))
    digest = hashlib.sha256(first_ppm.read_bytes()).hexdigest()
    require(stats["outputImageSha256"] == digest, "stats SHA-256 does not match the PPM")
    require(stats["ppmBytes"] == first_ppm.stat().st_size, "stats PPM byte count mismatch")
    require(stats["width"] == width and stats["height"] == height, "stats dimensions mismatch")
    require(stats["shadowMapSize"] == 512, "stats shadow map size does not match the scene")
    require(stats["rasterizedTriangles"] > 0, "rasterized triangle count is zero")
    require(stats["shadowSampleCount"] > 0, "shadow sample count is zero")
    require(stats["transparentTriangles"] > 0, "no transparent triangles were processed")
    require(stats["transparentSortSwaps"] > 0, "transparent sorting did not record swaps")
    require(stats["transparentFragments"] > 0, "no transparent fragments were blended")
    require(stats["peakMemoryEstimatedBytes"] > 0, "peak memory estimate is zero")
    require(len(payload) == width * height * 3, "PPM payload is incomplete")

    viewport_scene = copy.deepcopy(json.loads(sample_path.read_text(encoding="utf-8")))
    viewport_scene["directionalLight"]["shadow"]["mapSize"] = 256
    viewport_scene["image"]["viewport"] = {
        "x": 16,
        "y": 12,
        "width": 240,
        "height": 200,
    }
    viewport_path = work / "viewport.json"
    viewport_ppm = work / "viewport.ppm"
    viewport_stats_path = work / "viewport.stats.json"
    write_scene(viewport_path, viewport_scene)
    viewport_run = run(executable, viewport_path, viewport_ppm, viewport_stats_path)
    require(viewport_run.returncode == 0, f"viewport render failed: {viewport_run.stderr}")
    viewport_width, viewport_height, _ = parse_ppm(viewport_ppm)
    require((viewport_width, viewport_height) == (320, 320),
            "viewport changed the output image dimensions")
    viewport_stats = json.loads(viewport_stats_path.read_text(encoding="utf-8"))
    require(viewport_stats["viewport"] == {
        "x": 16, "y": 12, "width": 240, "height": 200
    }, "statistics do not report the configured viewport")
    require(viewport_stats["shadowMapSize"] == 256,
            "statistics do not report the configured shadow map size")
    require(viewport_stats["shadowSampleCount"] > 0,
            "configured viewport produced no shadow samples")

    sample_text = sample_path.read_text(encoding="utf-8")
    sample = json.loads(sample_text)

    missing_scene = run(
        executable,
        work / "does_not_exist.json",
        work / "io_error.ppm",
        work / "io_error.stats.json",
    )
    require(missing_scene.returncode == 1, "missing scene did not return IO_ERROR")
    require(json.loads(missing_scene.stderr)["error"]["name"] == "IO_ERROR",
            "missing scene did not report IO_ERROR as JSON")
    require(not (work / "io_error.ppm").exists(),
            "missing scene unexpectedly created an image")

    unwritable_ppm = work / "no_such_directory" / "output.ppm"
    unwritable_stats = work / "unwritable.stats.json"
    unwritable = run(executable, sample_path, unwritable_ppm, unwritable_stats)
    require(unwritable.returncode == 11, "unwritable output did not return code 11")
    require(json.loads(unwritable.stderr)["error"]["name"] == "OUTPUT_WRITE_ERROR",
            "unwritable output did not report OUTPUT_WRITE_ERROR")
    require(not unwritable_ppm.exists() and not unwritable_stats.exists(),
            "output write failure left partial output files")

    missing = copy.deepcopy(sample)
    del missing["camera"]["target"]
    missing_path = work / "missing.json"
    write_scene(missing_path, missing)
    expect_error(executable, work, "missing", missing_path, 3, "MISSING_FIELD")

    bad_index = copy.deepcopy(sample)
    bad_index["meshes"][0]["indices"][0] = 999999
    bad_index_path = work / "bad_index.json"
    write_scene(bad_index_path, bad_index)
    expect_error(executable, work, "bad_index", bad_index_path, 7, "INDEX_OUT_OF_RANGE")

    nonfinite_path = work / "nonfinite.json"
    nonfinite_text = re.sub(
        r'("fovYDegrees"\s*:\s*)[^,\n]+', r"\g<1>1e999", sample_text, count=1
    )
    require(nonfinite_text != sample_text, "failed to inject a non-finite value")
    nonfinite_path.write_text(nonfinite_text, encoding="utf-8")
    expect_error(executable, work, "nonfinite", nonfinite_path, 6, "NON_FINITE_FLOAT")

    large_shadow = copy.deepcopy(sample)
    large_shadow["directionalLight"]["shadow"]["mapSize"] = 8193
    large_shadow_path = work / "large_shadow.json"
    write_scene(large_shadow_path, large_shadow)
    expect_error(executable, work, "large_shadow", large_shadow_path, 8,
                 "SHADOW_MAP_TOO_LARGE")

    duplicate = copy.deepcopy(sample)
    transparent_object = next(
        obj for obj in duplicate["objects"]
        if duplicate["materials"][obj["material"]]["opacity"] < 1.0
    )
    duplicated_object = copy.deepcopy(transparent_object)
    duplicated_object["name"] = "duplicate_transparent_object"
    duplicate["objects"].append(duplicated_object)
    duplicate_path = work / "duplicate_depth.json"
    write_scene(duplicate_path, duplicate)
    expect_error(executable, work, "duplicate_depth", duplicate_path, 9,
                 "DUPLICATE_TRANSPARENT_DEPTH")

    memory_scene = copy.deepcopy(sample)
    memory_scene["limits"]["maxMemoryBytes"] = 1024
    memory_path = work / "memory_limit.json"
    write_scene(memory_path, memory_scene)
    expect_error(executable, work, "memory_limit", memory_path, 10,
                 "MEMORY_LIMIT_EXCEEDED")

    fallback = copy.deepcopy(sample)
    head_mesh = fallback["meshes"][1]
    for vertex in head_mesh["vertices"]:
        scale = 3.0
        vertex["normal"] = [component * scale for component in vertex["normal"]]
        vertex["uv"] = [0.25, 0.75]
    fallback_path = work / "fallback.json"
    fallback_ppm = work / "fallback.ppm"
    fallback_stats = work / "fallback.stats.json"
    write_scene(fallback_path, fallback)
    fallback_run = run(executable, fallback_path, fallback_ppm, fallback_stats)
    require(fallback_run.returncode == 0,
            f"normal/UV fallback path failed: {fallback_run.stderr}")
    require(fallback_ppm.exists() and fallback_stats.exists(),
            "normal/UV fallback did not produce complete outputs")
    parse_ppm(fallback_ppm)

    print("integration checks passed")
    print(f"sample_sha256={digest}")
    print(f"rasterized={stats['rasterizedTriangles']} clipped={stats['clippedTriangles']} "
          f"shadowSamples={stats['shadowSampleCount']} "
          f"sortSwaps={stats['transparentSortSwaps']} "
          f"peakMemory={stats['peakMemoryEstimatedBytes']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"integration failure: {error}", file=sys.stderr)
        raise SystemExit(1)
