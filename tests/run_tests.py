#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def fail(message):
    raise AssertionError(message)


def parse_json_output(text, label):
    text = text.strip()
    if not text:
        fail(f"{label} was empty")
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        fail(f"{label} was not a single JSON object: {error}: {text!r}")


def read_ppm(path):
    data = Path(path).read_bytes()
    if not data.startswith(b"P6\n"):
        fail(f"{path} does not start with P6 header")
    try:
        header, dimensions, maximum, pixels = data.split(b"\n", 3)
    except ValueError:
        fail(f"{path} has a truncated PPM header")
    if header != b"P6":
        fail(f"{path} has wrong magic")
    width, height = [int(value) for value in dimensions.split()]
    if maximum != b"255":
        fail(f"{path} is not 8-bit PPM")
    expected = width * height * 3
    if len(pixels) != expected:
        fail(f"{path} has {len(pixels)} bytes, expected {expected}")
    return width, height, pixels


def assert_no_render_artifacts(prefix):
    prefix = Path(prefix)
    for suffix in ("_geometry.ppm", "_lighting.ppm", "_full.ppm", "_stats.json"):
        path = Path(str(prefix) + suffix)
        if path.exists():
            fail(f"unexpected output artifact after rejected input: {path}")
    if Path(str(prefix) + ".render-in-progress.json").exists():
        fail("render marker remained after a rejected input")
    for path in prefix.parent.glob(prefix.name + "_*.tmp.*"):
        fail(f"temporary output remained after a rejected input: {path}")


class Harness:
    def __init__(self, renderer, generator):
        self.renderer = str(Path(renderer).resolve())
        self.generator = str(Path(generator).resolve())
        self.environment = os.environ.copy()
        self.environment["LIBGL_ALWAYS_SOFTWARE"] = "1"
        self.temp = tempfile.TemporaryDirectory(prefix="orv-test-")
        self.root = Path(self.temp.name)
        self.scene = self.root / "scene.json"
        self.no_lights_scene = self.root / "scene_no_lights.json"
        self.generate(self.scene, "default")
        self.generate(self.no_lights_scene, "no-lights")

    def close(self):
        self.temp.cleanup()

    def generate(self, path, variant):
        result = subprocess.run(
            [self.generator, "--output", str(path), "--variant", variant],
            text=True,
            capture_output=True,
            timeout=30,
            env=self.environment,
        )
        if result.returncode != 0:
            fail(f"scene generator failed: {result.stderr}")

    def run(self, scene, prefix, pass_name="full", width=192, height=108,
            shadow_size=512, extra_env=None, timeout=90):
        env = self.environment.copy()
        if extra_env:
            env.update(extra_env)
        command = [
            self.renderer,
            "--scene", str(scene),
            "--output", str(prefix),
            "--pass", pass_name,
            "--width", str(width),
            "--height", str(height),
            "--shadow-size", str(shadow_size),
        ]
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )

    def run_success(self, scene, prefix, pass_name="full"):
        result = self.run(scene, prefix, pass_name)
        if result.returncode != 0:
            fail(
                f"render failed for pass {pass_name}: exit={result.returncode}, "
                f"stdout={result.stdout!r}, stderr={result.stderr!r}"
            )
        state = parse_json_output(result.stdout, "final state")
        if state.get("state") != "completed" or state.get("code") != "RENDER_OK":
            fail(f"unexpected final state: {state}")
        stats_path = Path(str(prefix) + "_stats.json")
        if not stats_path.exists():
            fail(f"stats file missing: {stats_path}")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        return result, state, stats


def load_scene(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_scene(path, scene):
    Path(path).write_text(json.dumps(scene), encoding="utf-8")


def assert_error(test, result, exit_code, code):
    if result.returncode != exit_code:
        fail(
            f"{test}: expected exit {exit_code}, got {result.returncode}; "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
    payload = parse_json_output(result.stderr, f"{test} stderr")
    if payload.get("code") != code:
        fail(f"{test}: expected {code}, got {payload}")
    return payload


def test_successful_passes(harness):
    summaries = {}
    for pass_name in ("geometry", "lighting", "full"):
        prefix = harness.root / f"pass_{pass_name}"
        result, state, stats = harness.run_success(harness.scene, prefix, pass_name)
        ppm_path = Path(str(prefix) + f"_{pass_name}.ppm")
        if not ppm_path.exists():
            fail(f"missing {ppm_path}")
        width, height, pixels = read_ppm(ppm_path)
        if (width, height) != (192, 108):
            fail(f"wrong PPM resolution for {pass_name}")
        if len(set(pixels)) < 4:
            fail(f"{pass_name} output is unexpectedly uniform")
        if stats.get("pass") != pass_name:
            fail(f"stats pass mismatch: {stats}")
        if stats.get("width") != 192 or stats.get("height") != 108:
            fail(f"stats resolution mismatch: {stats}")
        if stats.get("render_time_ms", -1) < 0:
            fail(f"negative render time: {stats}")
        if stats.get("visible_pixels", 0) <= 0:
            fail(f"no visible pixels: {stats}")
        average = stats.get("average_luminance")
        maximum = stats.get("maximum_luminance")
        if not (0.0 <= average <= 1.0 and 0.0 <= maximum <= 1.0):
            fail(f"luminance outside [0,1]: {stats}")
        if maximum + 1.0e-9 < average:
            fail(f"maximum luminance below average: {stats}")
        if not isinstance(stats.get("average_luminance_clamped"), bool):
            fail("average clamp flag missing or not boolean")
        if not isinstance(stats.get("maximum_luminance_clamped"), bool):
            fail("maximum clamp flag missing or not boolean")
        if stats.get("light_count") != 4:
            fail(f"unexpected light count: {stats}")
        expected_shadow = "directional_2d_pcf3x3+point_cubemap_pcf3x3"
        if stats.get("shadow_method") != expected_shadow:
            fail(f"wrong shadow method: {stats}")
        if Path(str(prefix) + ".render-in-progress.json").exists():
            fail("render marker remained after success")
        for temp in prefix.parent.glob(prefix.name + "_*.tmp.*"):
            fail(f"temporary output remained after success: {temp}")
        summaries[pass_name] = pixels
    if summaries["geometry"] == summaries["lighting"]:
        fail("geometry and lighting channels are identical")
    if summaries["full"] == summaries["geometry"]:
        fail("full and geometry channels are identical")
    if summaries["full"] == summaries["lighting"]:
        fail("full and lighting channels are identical")

    isotropic = load_scene(harness.scene)
    for mesh in isotropic["meshes"]:
        mesh["material"]["anisotropy_strength"] = 0.0
    isotropic_path = harness.root / "isotropic.json"
    write_scene(isotropic_path, isotropic)
    isotropic_prefix = harness.root / "isotropic"
    harness.run_success(isotropic_path, isotropic_prefix, "full")
    _, _, isotropic_pixels = read_ppm(Path(str(isotropic_prefix) + "_full.ppm"))
    if isotropic_pixels == summaries["full"]:
        fail("anisotropy strength does not affect the final highlight response")


def test_invalid_scenes(harness):
    prefix = harness.root / "invalid"
    variants = []

    missing = load_scene(harness.scene)
    del missing["camera"]["fov_y_deg"]
    variants.append(("missing", missing, "SCENE_ERR_MISSING_FIELD"))

    bad_index = load_scene(harness.scene)
    bad_index["meshes"][0]["indices"][0] = 99999
    variants.append(("index", bad_index, "SCENE_ERR_INVALID_INDEX"))

    huge_index = load_scene(harness.scene)
    huge_index["meshes"][0]["indices"][0] = 1.0e300
    variants.append(("huge_index", huge_index, "SCENE_ERR_INVALID_INDEX"))

    bad_enum = load_scene(harness.scene)
    bad_enum["camera"]["projection"] = "orthographic"
    variants.append(("enum", bad_enum, "SCENE_ERR_INVALID_ENUM"))

    bad_tangent = load_scene(harness.scene)
    bad_tangent["meshes"][0]["vertices"]["tangents"][0] = [0.0, 0.0, 0.0]
    variants.append(("tangent", bad_tangent, "SCENE_ERR_INVALID_TANGENT"))

    too_many = load_scene(harness.scene)
    point = too_many["lights"]["point_lights"][0]
    too_many["lights"]["point_lights"] += [point, point]
    variants.append(("too_many_lights", too_many, "SCENE_ERR_TOO_MANY_LIGHTS"))

    missing_lights = load_scene(harness.scene)
    del missing_lights["lights"]["point_lights"]
    variants.append(("missing_point_lights", missing_lights,
                     "SCENE_ERR_MISSING_FIELD"))

    for name, scene, code in variants:
        path = harness.root / f"bad_{name}.json"
        write_scene(path, scene)
        result = harness.run(path, prefix)
        assert_error(name, result, 2, code)
        assert_no_render_artifacts(prefix)


def test_invalid_cli(harness):
    base_prefix = harness.root / "cli"
    result = subprocess.run(
        [
            harness.renderer,
            "--scene", str(harness.scene),
            "--output", str(base_prefix),
            "--pass", "wireframe",
        ],
        text=True,
        capture_output=True,
        timeout=30,
        env=harness.environment,
    )
    assert_error("invalid_pass", result, 5, "CLI_ERR_INVALID_PASS")
    assert_no_render_artifacts(base_prefix)

    for size in (500, 8192):
        result = subprocess.run(
            [
                harness.renderer,
                "--scene", str(harness.scene),
                "--output", str(base_prefix),
                "--shadow-size", str(size),
            ],
            text=True,
            capture_output=True,
            timeout=30,
            env=harness.environment,
        )
        assert_error(f"shadow_{size}", result, 4, "RENDER_ERR_INVALID_SHADOW_SIZE")
        assert_no_render_artifacts(base_prefix)


def test_framebuffer_failure(harness):
    prefix = harness.root / "fb_failure"
    result = harness.run(
        harness.scene,
        prefix,
        extra_env={"OFFLINE_RENDER_FORCE_FRAMEBUFFER_INCOMPLETE": "1"},
    )
    assert_error("framebuffer", result, 3, "RENDER_ERR_FRAMEBUFFER_INCOMPLETE")
    assert_no_render_artifacts(prefix)


def test_zero_lights(harness):
    prefix = harness.root / "no_lights"
    result, state, stats = harness.run_success(harness.no_lights_scene, prefix)
    if stats.get("light_count") != 0:
        fail(f"zero-light scene reports lights: {stats}")
    if stats.get("shadow_method") != "none":
        fail(f"zero-light scene reports a shadow method: {stats}")
    if stats.get("visible_pixels", 0) <= 0:
        fail("zero-light scene has no visible geometry")
    _, _, pixels = read_ppm(Path(str(prefix) + "_full.ppm"))
    if any(pixels):
        fail("zero-light final color is not black")


def test_stats_write_failure(harness):
    prefix = harness.root / "stats_failure"
    stats_path = Path(str(prefix) + "_stats.json")
    stats_path.mkdir(parents=True)
    result = harness.run(harness.scene, prefix)
    assert_error("stats_write", result, 6, "IO_ERR_STATS_WRITE")
    if Path(str(prefix) + ".render-in-progress.json").exists():
        fail("render marker remained after statistics write failure")


def test_recovery(harness):
    prefix = harness.root / "recovery"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    marker = Path(str(prefix) + ".render-in-progress.json")
    marker.write_text('{"version":1,"pid":999999}\n', encoding="utf-8")
    temp_a = Path(str(prefix) + "_full.ppm.tmp.999999.1")
    temp_b = Path(str(prefix) + "_stats.json.tmp.999999.2")
    temp_a.write_bytes(b"partial")
    temp_b.write_bytes(b"partial")
    result, state, stats = harness.run_success(harness.scene, prefix)
    if "RECOVERY_CLEANED" not in result.stderr:
        fail(f"missing RECOVERY_CLEANED report: {result.stderr!r}")
    recovery = parse_json_output(result.stderr, "recovery report")
    if recovery.get("code") != "RECOVERY_CLEANED":
        fail(f"wrong recovery report: {recovery}")
    if recovery.get("removed_temp_files", 0) < 2:
        fail(f"recovery did not report both temporary files: {recovery}")
    if marker.exists() or temp_a.exists() or temp_b.exists():
        fail("stale marker or temporary files remained after recovery")


def test_forced_termination_recovery(harness):
    prefix = harness.root / "forced_kill"
    marker = Path(str(prefix) + ".render-in-progress.json")
    env = harness.environment.copy()
    env["OFFLINE_RENDER_TEST_PAUSE_AFTER_MARKER_MS"] = "5000"
    process = subprocess.Popen(
        [
            harness.renderer,
            "--scene", str(harness.scene),
            "--output", str(prefix),
            "--width", "64",
            "--height", "64",
            "--shadow-size", "512",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    deadline = time.time() + 10
    while time.time() < deadline and not marker.exists():
        time.sleep(0.05)
    if not marker.exists():
        process.kill()
        stdout, stderr = process.communicate(timeout=10)
        fail(f"marker was not created before kill: {stdout!r} {stderr!r}")
    process.kill()
    process.communicate(timeout=10)
    if not marker.exists():
        fail("SIGKILL unexpectedly removed the render marker")
    result, _, _ = harness.run_success(harness.scene, prefix)
    if "RECOVERY_CLEANED" not in result.stderr:
        fail(f"forced-kill recovery was not reported: {result.stderr!r}")
    if marker.exists():
        fail("marker remained after forced-kill recovery")


def test_signal_handling(harness):
    prefix = harness.root / "signal"
    result = harness.run(
        harness.scene,
        prefix,
        extra_env={"OFFLINE_RENDER_TEST_RAISE_SIGINT": "1"},
    )
    if result.returncode != 130:
        fail(
            f"expected signal exit 130, got {result.returncode}; "
            f"stdout={result.stdout!r}, stderr={result.stderr!r}"
        )
    state = parse_json_output(result.stdout, "interrupted state")
    if state.get("state") != "interrupted" or state.get("code") != "RENDER_INTERRUPTED":
        fail(f"wrong interrupted state: {state}")
    if Path(str(prefix) + ".render-in-progress.json").exists():
        fail("marker remained after handled signal")
    assert_no_render_artifacts(prefix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renderer", required=True)
    parser.add_argument("--generator", required=True)
    args = parser.parse_args()

    harness = Harness(args.renderer, args.generator)
    tests = [
        ("successful_passes", test_successful_passes),
        ("invalid_scenes", test_invalid_scenes),
        ("invalid_cli", test_invalid_cli),
        ("framebuffer_failure", test_framebuffer_failure),
        ("zero_lights", test_zero_lights),
        ("stats_write_failure", test_stats_write_failure),
        ("recovery", test_recovery),
        ("forced_termination_recovery", test_forced_termination_recovery),
        ("signal_handling", test_signal_handling),
    ]
    try:
        for name, test in tests:
            test(harness)
            print(f"PASS {name}")
    except Exception as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1
    finally:
        harness.close()
    print(f"PASS all {len(tests)} integration groups")
    return 0


if __name__ == "__main__":
    sys.exit(main())
