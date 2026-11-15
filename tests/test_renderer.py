#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path


class TestFailure(RuntimeError):
    pass


def run_process(command, expected_exit=0):
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != expected_exit:
        raise TestFailure(
            f"exit {result.returncode}, expected {expected_exit}\n"
            f"command: {' '.join(str(part) for part in command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def decode_png(path):
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise TestFailure(f"not a PNG file: {path}")
    position = 8
    idat = bytearray()
    width = height = None
    while position < len(data):
        length = struct.unpack(">I", data[position:position + 4])[0]
        chunk_type = data[position + 4:position + 8]
        chunk_data = data[position + 8:position + 8 + length]
        position += 12 + length
        if chunk_type == b"IHDR":
            width, height, depth, color_type, compression, filtering, interlace = \
                struct.unpack(">IIBBBBB", chunk_data)
            if (depth, color_type, compression, filtering, interlace) != (8, 6, 0, 0, 0):
                raise TestFailure(f"unsupported PNG mode in {path}")
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None:
        raise TestFailure(f"missing PNG header: {path}")
    raw = zlib.decompress(bytes(idat))
    stride = width * 4
    expected_size = height * (stride + 1)
    if len(raw) != expected_size:
        raise TestFailure(f"unexpected PNG scanline size in {path}")
    pixels = []
    for y in range(height):
        row_start = y * (stride + 1)
        if raw[row_start] != 0:
            raise TestFailure(f"unexpected PNG filter type in {path}")
        row = raw[row_start + 1:row_start + 1 + stride]
        pixels.append([tuple(row[x * 4:x * 4 + 4]) for x in range(width)])
    return width, height, pixels


def linear_channel(value):
    value = value / 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def linear_luminance(pixel):
    return (
        0.2126 * linear_channel(pixel[0])
        + 0.7152 * linear_channel(pixel[1])
        + 0.0722 * linear_channel(pixel[2])
    )


def luminance_error(path_a, path_b):
    width_a, height_a, pixels_a = decode_png(path_a)
    width_b, height_b, pixels_b = decode_png(path_b)
    if (width_a, height_a) != (width_b, height_b):
        raise TestFailure("luminance comparison dimensions differ")
    total = 0.0
    for y in range(height_a):
        for x in range(width_a):
            total += abs(
                linear_luminance(pixels_a[y][x])
                - linear_luminance(pixels_b[y][x])
            )
    return total / (width_a * height_a)


def require(condition, message):
    if not condition:
        raise TestFailure(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer", required=True)
    parser.add_argument("--scene-gen", required=True)
    parser.add_argument("--shader-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    renderer = str(Path(args.renderer).resolve())
    scene_gen = str(Path(args.scene_gen).resolve())
    shader_dir = str(Path(args.shader_dir).resolve())
    work = Path(args.work_dir).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    sample_dir = work / "sample"
    run_process([scene_gen, str(sample_dir)])
    sample_copy_dir = work / "sample-copy"
    run_process([scene_gen, str(sample_copy_dir)])
    require(
        (sample_dir / "scene.json").read_bytes()
        == (sample_copy_dir / "scene.json").read_bytes(),
        "scene generator is not deterministic",
    )
    require(
        (sample_dir / "noise.raw").read_bytes()
        == (sample_copy_dir / "noise.raw").read_bytes(),
        "noise generator is not deterministic",
    )
    require(
        (sample_dir / "noise.raw").stat().st_size == 32 * 32 * 32 * 4,
        "generated noise raw has the wrong size",
    )
    scene_path = sample_dir / "scene.json"
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    width = scene["output"]["width"]
    height = scene["output"]["height"]

    def render(tag, extra=None, recovery_suffix=None, expected_exit=0):
        extra = [] if extra is None else extra
        recovery = work / f"recovery-{recovery_suffix or tag}"
        out_dir = work / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [
            renderer,
            "--scene", str(scene_path),
            "--shader-dir", shader_dir,
            "--recovery-dir", str(recovery),
            "--out", str(out_dir / "render.png"),
            "--stats", str(out_dir / "stats.json"),
            "--reference-out", str(out_dir / "reference.png"),
            "--validation", "on",
        ] + extra
        return run_process(command, expected_exit=expected_exit), out_dir

    first, first_dir = render("run-a")
    second, second_dir = render("run-b")

    png_a = (first_dir / "render.png").read_bytes()
    png_b = (second_dir / "render.png").read_bytes()
    require(png_a == png_b, "separate executions did not produce byte-identical PNGs")
    require(
        sha256_bytes(png_a) == sha256_bytes(png_b),
        "separate execution PNG SHA-256 values differ",
    )
    decode_png(first_dir / "render.png")
    decode_png(first_dir / "reference.png")

    stats = json.loads((first_dir / "stats.json").read_text(encoding="utf-8"))
    require(stats["status"] == "ok", "stats status is not ok")
    require(stats["output"]["framesIdentical"] is True, "frame identity flag is false")
    require(
        stats["output"]["pngSha256"] == sha256_bytes(png_a),
        "stats PNG hash does not match the output file",
    )
    require(
        stats["output"]["frameASha256"] == stats["output"]["frameBSha256"],
        "frame A/B PNG hashes differ",
    )
    require(
        stats["determinism"]["framePixelSha256A"]
        == stats["determinism"]["framePixelSha256B"],
        "frame A/B raw-pixel hashes differ",
    )
    require(
        stats["statistics"]["volumeSamplesFrameA"] > 0,
        "frame A volume sample count is zero",
    )
    require(
        stats["statistics"]["volumeSamplesFrameA"]
        == stats["statistics"]["volumeSamplesFrameB"],
        "frame A/B volume sample counts differ",
    )
    require(
        stats["statistics"]["shadowMapMilliseconds"] > 0.0,
        "shadow map time is not positive",
    )
    require(stats["reference"]["enabled"] is True, "reference frame was not enabled")
    require(stats["reference"]["passed"] is True, "brightness reference did not pass")
    require(
        stats["reference"]["meanAbsoluteLuminanceError"] <= 0.02,
        "reference luminance error exceeds 0.02",
    )
    require(
        "reference stepMeters=0.030" in stats["reference"]["source"],
        "reference source is not the documented fixed source",
    )

    bad_scene_path = work / "bad-schema.json"
    bad_scene = dict(scene)
    bad_scene["schemaVersion"] = 99
    bad_scene_path.write_text(json.dumps(bad_scene), encoding="utf-8")
    result = run_process(
        [
            renderer,
            "--scene", str(bad_scene_path),
            "--shader-dir", shader_dir,
            "--recovery-dir", str(work / "recovery-schema"),
            "--validation", "off",
        ],
        expected_exit=2,
    )
    require("E_SCHEMA" in result.stderr, "schema error did not emit E_SCHEMA")
    require("resources_live=0" in result.stderr, "schema path leaked resources")

    injection_cases = [
        ("shader", 3, "E_SHADER"),
        ("alloc", 4, "E_ALLOC"),
        ("desc", 5, "E_DESC"),
    ]
    for injection, exit_code, code in injection_cases:
        result, _ = render(
            f"inject-{injection}",
            extra=["--inject", injection],
            expected_exit=exit_code,
        )
        require(code in result.stderr, f"{injection} injection did not emit {code}")
        require(
            "resources_live=0" in result.stderr,
            f"{injection} injection leaked resources",
        )

    lost_result, lost_dir = render(
        "device-lost",
        extra=["--inject", "device-lost", "--no-reference"],
        expected_exit=6,
    )
    require("E_DEVICE_LOST" in lost_result.stderr, "device-loss path did not emit code")
    recovery_dir = work / "recovery-device-lost"
    raw_path = recovery_dir / "volume_light_recovery.rgba"
    metadata_path = recovery_dir / "volume_light_recovery.json"
    ready_path = recovery_dir / "volume_light_recovery.ready"
    require(raw_path.is_file(), "recovery raw frame is missing")
    require(metadata_path.is_file(), "recovery metadata is missing")
    require(ready_path.is_file(), "recovery ready marker is missing")
    require(
        raw_path.stat().st_size == width * height * 4,
        "recovery frame has the wrong size",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata["complete"] is True, "recovery metadata is not complete")
    require(metadata["width"] == width and metadata["height"] == height,
            "recovery dimensions differ from the scene")
    require("resources_live=0" in lost_result.stderr, "device-loss path leaked resources")

    prompt_result, prompt_dir = render(
        "recovery-prompt",
        extra=["--no-reference", "--clear-recovery"],
        recovery_suffix="device-lost",
    )
    require(
        "recovery: complete frame already exists" in prompt_result.stderr,
        "next startup did not announce the recoverable frame",
    )
    require(not raw_path.exists(), "--clear-recovery did not remove recovery files")

    reference_hq_path = work / "step-high-quality.png"
    result = run_process(
        [
            renderer,
            "--scene", str(scene_path),
            "--shader-dir", shader_dir,
            "--recovery-dir", str(work / "recovery-step-hq"),
            "--out", str(reference_hq_path),
            "--stats", str(work / "step-high-quality.json"),
            "--volume-step", "0.03",
            "--volume-max-steps", "768",
            "--no-reference",
            "--validation", "off",
        ]
    )
    step_errors = []
    for step in ("0.18", "0.12", "0.08"):
        step_path = work / f"step-{step}.png"
        result = run_process(
            [
                renderer,
                "--scene", str(scene_path),
                "--shader-dir", shader_dir,
                "--recovery-dir", str(work / f"recovery-step-{step}"),
                "--out", str(step_path),
                "--stats", str(work / f"step-{step}.json"),
                "--volume-step", step,
                "--volume-max-steps", "768",
                "--no-reference",
                "--validation", "off",
            ]
        )
        step_errors.append(luminance_error(step_path, reference_hq_path))
    require(
        step_errors[0] > step_errors[1] > step_errors[2],
        f"step refinement is not monotonic: {step_errors}",
    )
    require(
        step_errors[2] < 0.02,
        f"finest tested step error is too large: {step_errors[2]}",
    )

    print("PASS")
    print(f"png_sha256={sha256_bytes(png_a)}")
    print(f"reference_luminance_error={stats['reference']['meanAbsoluteLuminanceError']}")
    print(f"step_errors={[round(value, 8) for value in step_errors]}")
    print("injection_codes=E_SHADER,E_ALLOC,E_DESC,E_DEVICE_LOST")
    print("recovery=verified")


if __name__ == "__main__":
    try:
        main()
    except TestFailure as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print(f"FAIL: unexpected {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
