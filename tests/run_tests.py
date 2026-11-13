#!/usr/bin/env python3

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import struct
import subprocess
import sys
import zlib


class TestFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise TestFailure(message)


def make_scene(frame_count=12, width=48, height=48):
    directions = []
    for i in range(frame_count):
        angle = 2.0 * math.pi * i / frame_count
        y = 0.22 * math.sin(3.0 * angle)
        radial = math.sqrt(max(0.0, 1.0 - y * y))
        directions.append(
            {"direction": [radial * math.cos(angle), y, radial * math.sin(angle)]}
        )
    return {
        "resolution": {"width": width, "height": height},
        "samples": directions,
        "layers": [
            {
                "baseColor": [0.76, 0.18, 0.10],
                "roughness": 0.34,
                "normalStrength": 0.18,
                "emissionColor": [0.02, 0.005, 0.0],
            },
            {
                "baseColor": [0.10, 0.46, 0.84],
                "roughness": 0.67,
                "normalStrength": 0.36,
                "emissionColor": [0.0, 0.012, 0.035],
            },
            {
                "baseColor": [0.94, 0.70, 0.24],
                "roughness": 0.15,
                "normalStrength": 0.52,
                "emissionColor": [0.025, 0.018, 0.0],
            },
        ],
        "lights": [
            {
                "direction": [0.6, 0.8, 0.35],
                "color": [1.0, 0.95, 0.86],
                "intensity": 1.0,
            },
            {
                "direction": [-0.55, 0.25, -0.8],
                "color": [0.35, 0.5, 1.0],
                "intensity": 0.65,
            },
        ],
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_baker(baker, scene, output, cache, timeout=180):
    completed = subprocess.run(
        [str(baker), "--scene", str(scene), "--output", str(output), "--cache", str(cache)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    return completed.returncode, completed.stdout, completed.stderr


def parse_summary(stdout):
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    summary = [line for line in lines if line.startswith("SUMMARY ")]
    require(summary, "missing SUMMARY line")
    parts = summary[-1].split()
    require(len(parts) == 4, "malformed SUMMARY line: " + summary[-1])
    return {
        "baked": int(parts[1]),
        "cached": int(parts[2]),
        "total": int(parts[3]),
        "frames": len([line for line in lines if line.startswith("FRAME ")]),
    }


def snapshot(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            result[str(path.relative_to(directory)).replace("\\", "/")] = path.read_bytes()
    return result


def require_same_snapshot(actual_dir, expected_dir, context):
    actual = snapshot(actual_dir)
    expected = snapshot(expected_dir)
    require(set(actual) == set(expected),
            context + ": file sets differ: actual=" + repr(sorted(actual)) +
            " expected=" + repr(sorted(expected)))
    for name in expected:
        require(actual[name] == expected[name],
                context + ": byte mismatch in " + name)


def parse_png(path):
    data = path.read_bytes()
    require(data[:8] == b"\x89PNG\r\n\x1a\n", "bad PNG signature: " + str(path))
    position = 8
    idat = bytearray()
    chunks = []
    width = height = bit_depth = color_type = None
    while position < len(data):
        require(position + 12 <= len(data), "truncated PNG chunk")
        length = struct.unpack(">I", data[position:position + 4])[0]
        chunk_type = data[position + 4:position + 8]
        payload = data[position + 8:position + 8 + length]
        require(position + 12 + length <= len(data), "truncated PNG payload")
        chunks.append(chunk_type)
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", payload[:10])
        elif chunk_type == b"IDAT":
            idat.extend(payload)
        position += 12 + length

    require(b"IHDR" in chunks and b"IEND" in chunks, "missing PNG required chunks")
    require(b"tIME" not in chunks and b"tEXt" not in chunks and b"zTXt" not in chunks,
            "PNG contains non-deterministic metadata: " + str(path))
    require(bit_depth == 8 and color_type == 6, "PNG must be 8-bit RGBA")

    filtered = zlib.decompress(bytes(idat))
    stride = width * 4
    expected_size = height * (stride + 1)
    require(len(filtered) == expected_size, "unexpected PNG scanline size")
    rows = []
    previous = bytearray(stride)
    offset = 0
    for _ in range(height):
        filter_type = filtered[offset]
        offset += 1
        row = bytearray(filtered[offset:offset + stride])
        offset += stride
        for i in range(stride):
            left = row[i - 4] if i >= 4 else 0
            up = previous[i]
            upper_left = previous[i - 4] if i >= 4 else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = up
            elif filter_type == 3:
                predictor = (left + up) // 2
            elif filter_type == 4:
                p = left + up - upper_left
                pa = abs(p - left)
                pb = abs(p - up)
                pc = abs(p - upper_left)
                predictor = left if pa <= pb and pa <= pc else (up if pb <= pc else upper_left)
            else:
                raise TestFailure("unknown PNG filter " + str(filter_type))
            row[i] = (row[i] + predictor) & 0xFF
        rows.append(bytes(row))
        previous = row
    return width, height, b"".join(rows)


def verify_outputs(output_dir, scene):
    manifest_path = output_dir / "manifest.json"
    require(manifest_path.is_file(), "manifest.json missing")
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    frames = manifest.get("frames")
    require(isinstance(frames, list) and len(frames) == len(scene["samples"]),
            "manifest frame count mismatch")
    expected_names = set()
    for index, entry in enumerate(frames):
        name = f"frame_{index:06d}.png"
        expected_names.add(name)
        require(entry["index"] == index, "manifest index mismatch")
        require(entry["path"] == name, "manifest relative path mismatch")
        average_match = re.search(
            r'"path":\s*"' + re.escape(name) +
            r'",\s*"average_luminance":\s*(\d+\.\d{2})',
            manifest_text,
        )
        require(average_match is not None,
                "average luminance is not formatted with two decimals")
        png_path = output_dir / name
        width, height, rgba = parse_png(png_path)
        require(width == scene["resolution"]["width"] and
                height == scene["resolution"]["height"], "PNG resolution mismatch")
        rgb_sum = sum(rgba[i] for i in range(0, len(rgba), 4))
        rgb_sum += sum(rgba[i] for i in range(1, len(rgba), 4))
        rgb_sum += sum(rgba[i] for i in range(2, len(rgba), 4))
        average = rgb_sum / (width * height * 3.0)
        require(abs(entry["average_luminance"] - average) <= 0.0051,
                "manifest average luminance mismatch for " + name)
    actual_names = {path.name for path in output_dir.glob("frame_*.png")}
    require(actual_names == expected_names, "unexpected or missing output frames")


def expect_failure(baker, scene_path, output, cache, expected_code, expected_name,
                   work):
    code, stdout, stderr = run_baker(baker, scene_path, output, cache)
    require(code == expected_code,
            f"expected exit {expected_code}, got {code}: {stderr}")
    require(stderr.startswith(expected_name + ":"),
            f"expected stderr prefix {expected_name}, got: {stderr}")
    require("FRAME " not in stdout, "failure path emitted FRAME output")
    if output.exists() and output.is_dir():
        require(not list(output.glob("frame_*.png")),
                "failure path produced frame files")


def read_until_frame(process, count):
    lines = []
    while len([line for line in lines if line.startswith("FRAME ")]) < count:
        line = process.stdout.readline()
        if not line:
            remaining = process.stdout.read()
            raise TestFailure(
                "process ended before producing frames; stdout=" +
                "".join(lines) + remaining + " stderr=" + process.stderr.read()
            )
        lines.append(line)
    return lines


def run_interrupt_test(baker, scene_path, output, cache, baseline_dir, mode):
    output.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [str(baker), "--scene", str(scene_path), "--output", str(output),
         "--cache", str(cache)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    lines = read_until_frame(process, 2 if mode == "sigint" else 3)
    if mode == "sigint":
        process.send_signal(signal.SIGINT)
    else:
        process.kill()
    try:
        stdout_rest, stderr_rest = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise TestFailure(mode + ": baker did not terminate")
    all_stdout = "".join(lines) + (stdout_rest or "")
    if mode == "sigint":
        require(process.returncode == 0,
                "SIGINT must exit 0, got " + str(process.returncode) +
                " stderr=" + (stderr_rest or ""))
        summary = parse_summary(all_stdout)
        require(summary["total"] == 300, "SIGINT summary total mismatch")
        require(summary["frames"] < 300, "SIGINT did not stop after the current frame")
        require(not (output / "manifest.json").exists(),
                "partial SIGINT output must not publish a manifest")
    else:
        require(process.returncode != 0, "SIGKILL should not report success")

    code, stdout, stderr = run_baker(baker, scene_path, output, cache)
    require(code == 0, "resume failed: " + stderr)
    summary = parse_summary(stdout)
    require(summary["baked"] + summary["cached"] == 300,
            "resume summary does not cover all frames")
    require(summary["baked"] > 0 and summary["cached"] > 0,
            "resume should both reuse committed frames and render missing frames")
    require_same_snapshot(output, baseline_dir, mode + " recovered output")


def main():
    require(len(sys.argv) == 4, "usage: run_tests.py <repo> <baker> <work>")
    repo = Path(sys.argv[1]).resolve()
    baker = Path(sys.argv[2]).resolve()
    work = Path(sys.argv[3]).resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    scenes = work / "scenes"
    scenes.mkdir()

    base_scene = make_scene(12, 48, 48)
    base_scene_path = scenes / "base.json"
    write_json(base_scene_path, base_scene)

    baseline_out = work / "out-baseline"
    baseline_cache = work / "cache-baseline"
    code, stdout, stderr = run_baker(baker, base_scene_path, baseline_out, baseline_cache)
    require(code == 0, "baseline bake failed: " + stderr)
    summary = parse_summary(stdout)
    require(summary == {"baked": 12, "cached": 0, "total": 12, "frames": 12},
            "unexpected baseline summary: " + repr(summary))
    verify_outputs(baseline_out, base_scene)
    primary_global_key = next(p.name for p in baseline_cache.iterdir() if p.is_dir())
    require(re.fullmatch(r"[0-9a-f]{64}", primary_global_key), "global key is not SHA-256")
    for subkey_dir in (baseline_cache / primary_global_key).iterdir():
        require(re.fullmatch(r"[0-9a-f]{64}", subkey_dir.name),
                "frame subkey is not SHA-256")
    repair_cache = work / "cache-repair"
    shutil.copytree(baseline_cache, repair_cache)

    hit_out = work / "out-hit"
    code, stdout, stderr = run_baker(baker, base_scene_path, hit_out, baseline_cache)
    require(code == 0, "cache-hit bake failed: " + stderr)
    summary = parse_summary(stdout)
    require(summary == {"baked": 0, "cached": 12, "total": 12, "frames": 12},
            "unexpected cache-hit summary: " + repr(summary))
    require_same_snapshot(hit_out, baseline_out, "cache hit")

    direction_scene = copy.deepcopy(base_scene)
    direction_scene["samples"][5]["direction"] = [0.28, 0.16, 0.946]
    direction_path = scenes / "direction-tweak.json"
    write_json(direction_path, direction_scene)
    direction_out = work / "out-direction"
    code, stdout, stderr = run_baker(baker, direction_path, direction_out, baseline_cache)
    require(code == 0, "direction tweak failed: " + stderr)
    summary = parse_summary(stdout)
    require(summary == {"baked": 1, "cached": 11, "total": 12, "frames": 12},
            "direction tweak should invalidate one subkey: " + repr(summary))
    for index in range(12):
        name = f"frame_{index:06d}.png"
        same = (direction_out / name).read_bytes() == (baseline_out / name).read_bytes()
        require(same == (index != 5),
                "direction tweak changed the wrong frame set at " + name)

    material_scene = copy.deepcopy(base_scene)
    material_scene["layers"][0]["roughness"] = 0.41
    material_path = scenes / "material-tweak.json"
    write_json(material_path, material_scene)
    material_out = work / "out-material"
    code, stdout, stderr = run_baker(baker, material_path, material_out, baseline_cache)
    require(code == 0, "material tweak failed: " + stderr)
    summary = parse_summary(stdout)
    require(summary["baked"] == 12 and summary["cached"] == 0,
            "material change must invalidate every frame subkey: " + repr(summary))

    missing_field = copy.deepcopy(base_scene)
    del missing_field["layers"][0]["roughness"]
    missing_field_path = scenes / "missing-field.json"
    write_json(missing_field_path, missing_field)
    expect_failure(baker, missing_field_path, work / "out-missing", work / "cache-invalid",
                   2, "E_SCENE_PARSE", work)

    wrong_type = copy.deepcopy(base_scene)
    wrong_type["layers"][0]["roughness"] = "smooth"
    wrong_type_path = scenes / "wrong-type.json"
    write_json(wrong_type_path, wrong_type)
    expect_failure(baker, wrong_type_path, work / "out-wrong-type", work / "cache-invalid",
                   2, "E_SCENE_PARSE", work)

    malformed_path = scenes / "malformed.json"
    malformed_path.write_text('{"resolution": [48, 48],', encoding="utf-8")
    expect_failure(baker, malformed_path, work / "out-malformed", work / "cache-invalid",
                   2, "E_SCENE_PARSE", work)

    bad_roughness = copy.deepcopy(base_scene)
    bad_roughness["layers"][0]["roughness"] = 1.5
    bad_roughness_path = scenes / "bad-roughness.json"
    write_json(bad_roughness_path, bad_roughness)
    expect_failure(baker, bad_roughness_path, work / "out-bad-rough", work / "cache-invalid",
                   3, "E_SCENE_VALID", work)

    bad_strength = copy.deepcopy(base_scene)
    bad_strength["layers"][1]["normalStrength"] = -0.01
    bad_strength_path = scenes / "bad-strength.json"
    write_json(bad_strength_path, bad_strength)
    expect_failure(baker, bad_strength_path, work / "out-bad-strength", work / "cache-invalid",
                   3, "E_SCENE_VALID", work)

    eight_bit_scene = copy.deepcopy(base_scene)
    eight_bit_scene["layers"][0]["baseColor"] = [194, 51, 26]
    eight_bit_scene["lights"][0]["color"] = [255, 242, 219]
    eight_bit_path = scenes / "eight-bit-colors.json"
    write_json(eight_bit_path, eight_bit_scene)
    code, stdout, stderr = run_baker(
        baker, eight_bit_path, work / "out-eight-bit", work / "cache-eight-bit"
    )
    require(code == 0, "0-255 color input should be accepted: " + stderr)
    summary = parse_summary(stdout)
    require(summary["baked"] == 12, "0-255 color bake should complete")

    too_many_layers = copy.deepcopy(base_scene)
    too_many_layers["layers"] = [
        copy.deepcopy(base_scene["layers"][0]) for _ in range(33)
    ]
    too_many_layers_path = scenes / "too-many-layers.json"
    write_json(too_many_layers_path, too_many_layers)
    expect_failure(baker, too_many_layers_path, work / "out-too-many", work / "cache-invalid",
                   3, "E_SCENE_VALID", work)

    zero_direction = copy.deepcopy(base_scene)
    zero_direction["samples"][0]["direction"] = [0.0, 0.0, 0.0]
    zero_direction_path = scenes / "zero-direction.json"
    write_json(zero_direction_path, zero_direction)
    expect_failure(baker, zero_direction_path, work / "out-zero", work / "cache-invalid",
                   3, "E_SCENE_VALID", work)

    blocked_output = work / "output-is-a-file"
    blocked_output.write_text("not a directory", encoding="utf-8")
    code, stdout, stderr = run_baker(baker, base_scene_path, blocked_output, work / "cache-io")
    require(code == 4 and stderr.startswith("E_OUTPUT_IO:"),
            "unwritable output should return E_OUTPUT_IO")
    require("FRAME " not in stdout, "unwritable output must fail before rendering")

    blocked_cache = work / "cache-is-a-file"
    blocked_cache.write_text("not a directory", encoding="utf-8")
    code, stdout, stderr = run_baker(baker, base_scene_path, work / "out-cache-io", blocked_cache)
    require(code == 5 and stderr.startswith("E_CACHE_IO:"),
            "unwritable cache should return E_CACHE_IO")
    require("FRAME " not in stdout, "unwritable cache must fail before rendering")

    metadata_files = sorted(repair_cache.rglob("*.meta.json"))
    require(len(metadata_files) == 12, "expected 12 cache metadata files")
    removed_meta = metadata_files[3]
    removed_meta.unlink()
    temp_leftover = removed_meta.parent / ".tmp-leftover"
    temp_leftover.write_bytes(b"half-finished")
    repair_out = work / "out-meta-repair"
    code, stdout, stderr = run_baker(baker, base_scene_path, repair_out, repair_cache)
    require(code == 0, "metadata repair run failed: " + stderr)
    summary = parse_summary(stdout)
    require(summary == {"baked": 1, "cached": 11, "total": 12, "frames": 12},
            "missing metadata should rerender exactly one frame: " + repr(summary))
    require(not temp_leftover.exists(), "stale temporary cache file was not cleaned")
    require_same_snapshot(repair_out, baseline_out, "metadata repair")

    rgba_files = sorted(repair_cache.rglob("*.rgba"))
    corrupted = rgba_files[7]
    corrupted.write_bytes(b"corrupt cache pixel data")
    corrupted_out = work / "out-rgba-repair"
    code, stdout, stderr = run_baker(baker, base_scene_path, corrupted_out, repair_cache)
    require(code == 0, "checksum repair run failed: " + stderr)
    summary = parse_summary(stdout)
    require(summary["baked"] == 1 and summary["cached"] == 11,
            "bad pixel checksum should rerender exactly one frame: " + repr(summary))
    require_same_snapshot(corrupted_out, baseline_out, "checksum repair")

    resolution_scene = copy.deepcopy(base_scene)
    resolution_scene["resolution"] = {"width": 44, "height": 40}
    resolution_path = scenes / "resolution-tweak.json"
    write_json(resolution_path, resolution_scene)
    resolution_out = work / "out-resolution"
    code, stdout, stderr = run_baker(baker, resolution_path, resolution_out, baseline_cache)
    require(code == 0, "resolution tweak failed: " + stderr)
    summary = parse_summary(stdout)
    require(summary["baked"] == 12 and summary["cached"] == 0,
            "resolution is global and must invalidate every frame: " + repr(summary))
    require(len([p for p in baseline_cache.iterdir() if p.is_dir()]) >= 2,
            "resolution change did not create a new global cache key")

    stale_out = work / "out-stale"
    shutil.copytree(baseline_out, stale_out)
    shortened_scene = copy.deepcopy(base_scene)
    shortened_scene["samples"] = shortened_scene["samples"][:10]
    shortened_path = scenes / "shortened.json"
    write_json(shortened_path, shortened_scene)
    code, stdout, stderr = run_baker(
        baker, shortened_path, stale_out, work / "cache-shortened"
    )
    require(code == 0, "shortened frame-list bake failed: " + stderr)
    verify_outputs(stale_out, shortened_scene)
    require(len(list(stale_out.glob("frame_*.png"))) == 10,
            "stale sequence frames were not removed")

    stress_scene = make_scene(300, 48, 48)
    stress_path = scenes / "stress.json"
    write_json(stress_path, stress_scene)
    stress_baseline = work / "stress-baseline"
    stress_baseline_cache = work / "stress-baseline-cache"
    code, stdout, stderr = run_baker(baker, stress_path, stress_baseline,
                                     stress_baseline_cache)
    require(code == 0, "stress baseline failed: " + stderr)
    summary = parse_summary(stdout)
    require(summary["baked"] == 300 and summary["cached"] == 0,
            "stress baseline summary mismatch")
    verify_outputs(stress_baseline, stress_scene)

    if os.name == "nt":
        print("SKIP: Ctrl+C/SIGKILL process tests are POSIX-only in this harness")
    else:
        run_interrupt_test(
            baker, stress_path, work / "stress-sigint", work / "stress-sigint-cache",
            stress_baseline, "sigint"
        )
        run_interrupt_test(
            baker, stress_path, work / "stress-sigkill", work / "stress-sigkill-cache",
            stress_baseline, "sigkill"
        )

    print("integration tests passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except TestFailure as error:
        print("TEST FAILURE: " + str(error), file=sys.stderr)
        sys.exit(1)
