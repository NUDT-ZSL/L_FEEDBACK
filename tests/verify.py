#!/usr/bin/env python3
"""End-to-end acceptance checks for the CPU terrain prototype."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_renderer(executable: Path, config: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(executable), "--config", str(config)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def write_config(source: dict, directory: Path, name: str, image_name: str) -> Path:
    config = copy.deepcopy(source)
    config["output"]["image"] = str(directory / image_name)
    config["output"]["report"] = str(directory / "report.json")
    path = directory / name
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def make_matrix_case(
    source: dict, directory: Path, size: int, supersampling: int, width: int, height: int
) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(source)
    config["heightfield"]["size"] = size
    config["render"]["width"] = width
    config["render"]["height"] = height
    config["render"]["supersampling"] = supersampling
    config["camera"]["position"] = [size * 0.5, 12.0 + size * 0.20, size + 36.0]
    config["camera"]["target"] = [size * 0.5, 5.0, size * 0.5]
    config["camera"]["far"] = size * 4.0 + 100.0
    image_name = "terrain.bmp" if supersampling != 4 else "terrain.ppm"
    config["output"]["image"] = str(directory / image_name)
    config["output"]["report"] = str(directory / "report.json")
    path = directory / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path, image_name


def validate_bmp(path: Path, width: int, height: int) -> None:
    data = path.read_bytes()
    if data[:2] != b"BM":
        fail("BMP signature is invalid")
    file_size, = struct.unpack_from("<I", data, 2)
    data_offset, = struct.unpack_from("<I", data, 10)
    image_width, image_height = struct.unpack_from("<ii", data, 18)
    bits = struct.unpack_from("<H", data, 28)[0]
    if file_size != len(data):
        fail("BMP header file size does not match file length")
    if image_width != width or image_height != height or bits != 24:
        fail("BMP dimensions or bit depth are invalid")
    row_bytes = width * 3
    padded_row = (row_bytes + 3) // 4 * 4
    if data_offset + padded_row * height != len(data):
        fail("BMP pixel payload length is invalid")


def validate_ppm(path: Path, width: int, height: int) -> None:
    data = path.read_bytes()
    magic, dimensions, maximum = data.split(b"\n", 2)
    if magic != b"P6" or maximum[:3] != b"255":
        fail("PPM header is invalid")
    w, h = (int(value) for value in dimensions.split())
    if w != width or h != height:
        fail("PPM dimensions are invalid")
    payload = data[data.find(b"\n", data.find(b"255")) + 1 :]
    if len(payload) != width * height * 3:
        fail("PPM payload length is invalid")


def verify_report(report: dict, config: dict) -> None:
    triangles = report["triangles"]
    lod = report["lod"]
    raster = report["rasterization"]
    gbuffer = report["g_buffer"]
    aniso = report["anisotropic_sampling"]

    if triangles["total"] != triangles["terrain"] + triangles["transition"]:
        fail("triangle accounting does not add up")
    if report["total_triangles"] != triangles["total"]:
        fail("top-level total_triangles alias is inconsistent")
    if report["clipped_triangles"] != triangles["after_frustum_clip"]:
        fail("top-level clipped_triangles alias is inconsistent")
    if report["culled_triangles"] != triangles["culled"]:
        fail("top-level culled_triangles alias is inconsistent")
    if triangles["total"] <= 0 or triangles["transition"] <= 0:
        fail("the sample must exercise LOD transition triangles")
    if triangles["culled"] != (
        triangles["frustum_culled"] + triangles["backface_culled"]
    ):
        fail("culled triangle count is inconsistent")
    if triangles["clipped_triangle_count"] != triangles["after_frustum_clip"]:
        fail("clipped_triangle_count does not match the clipping pass")
    if triangles["after_clipping"] > triangles["after_frustum_clip"]:
        fail("post-culling triangle count exceeds the clipping output")

    distribution = lod["distribution"]
    for key in ("lod0_blocks", "lod1_blocks", "lod2_blocks"):
        if distribution[key] <= 0:
            fail(f"sample did not exercise {key}")
    if sum(distribution.values()) != lod["block_columns"] * lod["block_rows"]:
        fail("LOD block distribution does not cover the block grid")
    if report["lod_distribution"] != {
        "0": distribution["lod0_blocks"],
        "1": distribution["lod1_blocks"],
        "2": distribution["lod2_blocks"],
    }:
        fail("top-level lod_distribution alias is inconsistent")
    if lod["seam_method"] != "transition_triangles":
        fail("unexpected seam method")
    topology = report["topology"]
    if topology["non_manifold_edges"] != 0 or topology["hanging_edges"] != 0:
        fail("topology audit found non-manifold or hanging edges")
    if topology["edges"] != topology["boundary_edges"] + topology["manifold_edges"]:
        fail("topology edge accounting is inconsistent")

    if raster["width"] != config["render"]["width"]:
        fail("report width differs from config")
    if raster["height"] != config["render"]["height"]:
        fail("report height differs from config")
    if raster["supersampling"] != config["render"]["supersampling"]:
        fail("report supersampling differs from config")
    expected_internal = (
        config["render"]["width"] * config["render"]["supersampling"],
        config["render"]["height"] * config["render"]["supersampling"],
    )
    if (raster["internal_width"], raster["internal_height"]) != expected_internal:
        fail("internal resolution differs from SSAA settings")
    if raster["covered_samples"] <= 0:
        fail("no G-buffer samples were covered")
    if triangles["unique_vertices"] <= 0:
        fail("unique vertex count is invalid")

    if gbuffer["bytes_per_pixel"] <= 0 or gbuffer["total_bytes"] <= 0:
        fail("G-buffer memory estimate is invalid")
    if report["gbuffer_memory_bytes"] != gbuffer["total_bytes"]:
        fail("top-level gbuffer_memory_bytes alias is inconsistent")
    expected_bytes = (
        raster["internal_width"] * raster["internal_height"] * gbuffer["bytes_per_pixel"]
    )
    if gbuffer["pixel_count"] != raster["internal_width"] * raster["internal_height"]:
        fail("G-buffer pixel count is invalid")
    if gbuffer["total_bytes"] != expected_bytes:
        fail("G-buffer total bytes are inconsistent")

    if not (1.0 <= aniso["average_samples"] <= 16.0):
        fail("average anisotropic sample count is outside the sample budget")
    if report["average_anisotropic_samples"] != aniso["average_samples"]:
        fail("top-level anisotropic sample alias is inconsistent")
    if aniso["max_samples"] < 1 or aniso["max_samples"] > 16:
        fail("maximum anisotropic sample count is outside the sample budget")


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: verify.py <renderer> <default-config> <work-directory>", file=sys.stderr)
        return 2
    executable = Path(sys.argv[1]).resolve()
    source_config_path = Path(sys.argv[2]).resolve()
    work = Path(sys.argv[3]).resolve()
    work.mkdir(parents=True, exist_ok=True)

    source = json.loads(source_config_path.read_text(encoding="utf-8"))
    bmp_config = write_config(source, work, "deterministic.json", "terrain-a.bmp")
    first = run_renderer(executable, bmp_config)
    if first.returncode != 0:
        fail(f"first valid render failed: {first.stderr.strip()}")
    image_a = work / "terrain-a.bmp"
    report_a = work / "report.json"
    hashes_a = (sha256(image_a), sha256(report_a))

    second = run_renderer(executable, bmp_config)
    if second.returncode != 0:
        fail(f"second valid render failed: {second.stderr.strip()}")
    hashes_b = (sha256(image_a), sha256(report_a))
    if hashes_a != hashes_b:
        fail("same config produced different image or report bytes")

    report = json.loads(report_a.read_text(encoding="utf-8"))
    validate_bmp(image_a, source["render"]["width"], source["render"]["height"])
    verify_report(report, source)

    ppm_config = write_config(source, work, "ppm.json", "terrain.ppm")
    ppm_run = run_renderer(executable, ppm_config)
    if ppm_run.returncode != 0:
        fail(f"PPM render failed: {ppm_run.stderr.strip()}")
    validate_ppm(work / "terrain.ppm", source["render"]["width"], source["render"]["height"])

    invalid = copy.deepcopy(source)
    invalid["camera"]["fov_degrees"] = 180.0
    invalid["output"]["image"] = str(work / "invalid.bmp")
    invalid["output"]["report"] = str(work / "invalid.json")
    invalid_path = work / "invalid.json.config"
    invalid_path.write_text(json.dumps(invalid, indent=2) + "\n", encoding="utf-8")
    invalid_run = run_renderer(executable, invalid_path)
    if invalid_run.returncode == 0:
        fail("invalid configuration exited with code 0")
    if "camera.fov_degrees" not in invalid_run.stderr:
        fail("invalid configuration error message is not specific")
    if (work / "invalid.bmp").exists() or (work / "invalid.json").exists():
        fail("invalid configuration still produced output files")

    missing = copy.deepcopy(source)
    del missing["lighting"]["point"]
    missing_path = work / "missing-lights.json"
    missing_path.write_text(json.dumps(missing, indent=2) + "\n", encoding="utf-8")
    missing_run = run_renderer(executable, missing_path)
    if missing_run.returncode == 0 or "lighting.point" not in missing_run.stderr:
        fail("missing required light list did not produce a clear configuration error")

    matrix_cases = (
        (64, 1, 192, 108),
        (128, 4, 128, 72),
        (512, 1, 240, 135),
    )
    for size, supersampling, width, height in matrix_cases:
        directory = work / f"matrix-{size}-{supersampling}x"
        matrix_config, image_name = make_matrix_case(
            source, directory, size, supersampling, width, height
        )
        matrix_run = run_renderer(executable, matrix_config)
        if matrix_run.returncode != 0:
            fail(f"matrix case {size}/{supersampling}x failed: {matrix_run.stderr.strip()}")
        matrix_report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if matrix_report["topology"]["hanging_edges"] != 0:
            fail(f"matrix case {size}/{supersampling}x has hanging edges")
        if matrix_report["topology"]["non_manifold_edges"] != 0:
            fail(f"matrix case {size}/{supersampling}x has non-manifold edges")
        if matrix_report["triangles"]["total"] <= 0:
            fail(f"matrix case {size}/{supersampling}x emitted no triangles")
        if supersampling == 4:
            validate_ppm(directory / image_name, width, height)
        else:
            validate_bmp(directory / image_name, width, height)

    print(
        "PASS deterministic "
        f"image={hashes_a[0]} report={hashes_a[1]} "
        f"triangles={report['triangles']['total']} "
        f"lod={report['lod']['distribution']} "
        f"avg_aniso={report['anisotropic_sampling']['average_samples']:.6f} "
        "matrix=64x1,128x4,512x1"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exception:
        print(f"FAIL: {exception}", file=sys.stderr)
        sys.exit(1)
