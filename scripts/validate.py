#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent
SUMMARY_RE = re.compile(r"^success=(\d+) failed=(\d+) exit=(\d+)$")


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_ppm(path):
    data = path.read_bytes()
    pieces = data.split(b"\n", 3)
    check(len(pieces) == 4, f"{path}: missing PPM header terminator")
    check(pieces[0] == b"P6", f"{path}: expected binary P6 PPM")
    dimensions = pieces[1].split(b" ")
    check(len(dimensions) == 2, f"{path}: malformed dimensions")
    width, height = int(dimensions[0]), int(dimensions[1])
    check(pieces[2] == b"255", f"{path}: expected maxval 255")
    check(
        len(pieces[3]) == width * height * 3,
        f"{path}: RGB8 payload size mismatch",
    )
    return width, height, pieces[3]


def run_renderer(exe, config, cwd, output_dir, expected_exit):
    cwd.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    command = [
        str(exe),
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--backend",
        "auto",
    ]
    process = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )

    check(
        process.returncode == expected_exit,
        f"unexpected exit {process.returncode}, expected {expected_exit}; "
        f"stdout={process.stdout!r} stderr={process.stderr!r}",
    )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    check(len(lines) == 1, f"console must contain one summary line, got {lines!r}")
    match = SUMMARY_RE.match(lines[0])
    check(match is not None, f"malformed console summary: {lines[0]!r}")
    success, failed, summary_exit = map(int, match.groups())
    check(
        summary_exit == process.returncode,
        "console exit code disagrees with process exit code",
    )
    check(
        "Traceback" not in process.stderr and "[ERROR]" not in process.stderr,
        "detailed errors leaked to stderr",
    )
    return process, success, failed


def load_json(path):
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def assert_stats_exit(run_dir, summary_exit):
    stats = load_json(run_dir / "diff_stats.json")
    check(stats["stats_schema_version"] == 1, "stats schema version is not 1")
    check(
        stats["run"]["exit_code"] == summary_exit,
        "stats exit code disagrees with summary",
    )
    return stats


def validate_determinism(exe, work):
    first = work / "determinism_a"
    second = work / "determinism_b"
    _, success_a, failed_a = run_renderer(
        exe, ROOT / "configs/demo.json", first, first / "out", 0
    )
    _, success_b, failed_b = run_renderer(
        exe, ROOT / "configs/demo.json", second, second / "out", 0
    )
    check((success_a, failed_a) == (2, 0), "demo summary count mismatch")
    check((success_b, failed_b) == (2, 0), "demo summary count mismatch")

    first_files = sorted((first / "out").glob("*.ppm"))
    second_files = sorted((second / "out").glob("*.ppm"))
    check(len(first_files) == 2, "expected two demo PPM files")
    check(
        [path.name for path in first_files]
        == [path.name for path in second_files],
        "deterministic runs produced different filenames",
    )
    for left, right in zip(first_files, second_files):
        check(
            left.read_bytes() == right.read_bytes(),
            f"PPM bytes differ between runs: {left.name}",
        )
        width, height, payload = parse_ppm(left)
        check((width, height) == (512, 512), "demo dimensions are not 512x512")
        bright_pixels = sum(
            1
            for index in range(0, len(payload), 3)
            if max(payload[index : index + 3]) > 80
        )
        check(
            bright_pixels > width * height // 20,
            f"{left.name}: image appears empty",
        )

    log_a = (first / "render.log").read_text(encoding="utf-8")
    log_b = (second / "render.log").read_text(encoding="utf-8")
    check('group="skin_satin"' in log_a, "group name missing from log")
    check("resource framebuffers created=2" in log_a, "framebuffer reuse not shown")
    check("resource framebuffers created=2" in log_b, "framebuffer reuse not stable")
    check("resource programs created=2" in log_a, "program cache not shown")
    check("balanced=yes" in log_a, "resource release counters are unbalanced")
    check(
        "scattering_samples=24" in log_a,
        "preintegrated LUT sample count missing from log",
    )
    check("[STAT] group=\"skin_satin\"" in log_a, "baseline stats missing in log")
    stats_a = assert_stats_exit(first, 0)
    stats_b = assert_stats_exit(second, 0)
    check(
        all(group["baseline_status"] == "compared" for group in stats_a["groups"]),
        "shipped demo baseline was not compared",
    )
    check(
        all("comparison" in group for group in stats_a["groups"]),
        "shipped demo baseline comparison statistics missing",
    )
    check(
        [group["render_fingerprint"] for group in stats_a["groups"]]
        == [group["render_fingerprint"] for group in stats_b["groups"]],
        "stats fingerprints differ between deterministic runs",
    )
    return first


def validate_baseline_comparison(exe, work, first_run):
    config = load_json(ROOT / "configs/demo.json")
    for material in config["materials"]:
        material["baseline"] = str(first_run / "out" / f"{material['name']}.ppm")
    compare_config = work / "baseline_config.json"
    write_json(compare_config, config)

    run_dir = work / "baseline_compare"
    _, success, failed = run_renderer(
        exe, compare_config, run_dir, run_dir / "out", 0
    )
    check((success, failed) == (2, 0), "baseline run summary mismatch")
    stats = assert_stats_exit(run_dir, 0)
    for group in stats["groups"]:
        check(group["baseline_status"] == "compared", "baseline was not compared")
        check(group["comparison"]["mse"] == 0.0, "self-baseline MSE is not zero")
        check(
            group["comparison"]["max_absolute_error"] == 0,
            "self-baseline max absolute difference is not zero",
        )
    log_text = (run_dir / "render.log").read_text(encoding="utf-8")
    check("mse=0" in log_text, "zero MSE was not logged")
    check("max_abs=0" in log_text, "zero max absolute difference was not logged")


def validate_no_overwrite(exe, work):
    run_dir = work / "no_overwrite"
    output_dir = run_dir / "out"
    output_dir.mkdir(parents=True)
    sentinel = output_dir / "skin_satin.ppm"
    sentinel.write_bytes(b"do-not-overwrite")

    _, success, failed = run_renderer(
        exe, ROOT / "configs/demo.json", run_dir, output_dir, 0
    )
    check((success, failed) == (2, 0), "no-overwrite run summary mismatch")
    check(
        sentinel.read_bytes() == b"do-not-overwrite",
        "preexisting output was overwritten",
    )
    check(
        (output_dir / "skin_satin_001.ppm").is_file(),
        "sequence-suffixed output was not created",
    )
    parse_ppm(output_dir / "skin_satin_001.ppm")


def validate_failure_isolation(exe, work):
    run_dir = work / "failure"
    output_dir = run_dir / "out"
    _, success, failed = run_renderer(
        exe, ROOT / "configs/failure_demo.json", run_dir, output_dir, 1
    )
    check((success, failed) == (2, 2), "failure summary count mismatch")
    check(not (output_dir / "bad_roughness.ppm").exists(), "invalid group wrote output")
    check(
        not (output_dir / "bad_anisotropy.ppm").exists(),
        "invalid anisotropy group wrote output",
    )
    first_valid = output_dir / "valid_duplicate.ppm"
    second_valid = output_dir / "valid_duplicate_001.ppm"
    check(first_valid.is_file() and second_valid.is_file(), "later groups did not render")
    check(
        sha256(first_valid) != sha256(second_valid),
        "duplicate-name groups unexpectedly produced identical images",
    )
    width, height, _ = parse_ppm(first_valid)
    check((width, height) == (256, 192), "failure test dimensions are wrong")

    log_text = (run_dir / "render.log").read_text(encoding="utf-8")
    check('group="bad_roughness" code=111' in log_text, "roughness error not logged")
    check(
        'group="bad_anisotropy" code=112' in log_text,
        "anisotropy error not logged",
    )
    check(
        "baseline file does not exist; comparison skipped" in log_text,
        "missing-baseline warning not logged",
    )
    check(
        'group="valid_duplicate" rendered' in log_text,
        "valid group after the failure was not rendered",
    )

    stats = assert_stats_exit(run_dir, 1)
    statuses = [(group["name"], group["status"]) for group in stats["groups"]]
    check(
        statuses
        == [
            ("bad_roughness", "failed"),
            ("valid_duplicate", "success"),
            ("bad_anisotropy", "failed"),
            ("valid_duplicate", "success"),
        ],
        f"unexpected group statuses: {statuses}",
    )
    codes = {group["name"]: group.get("error_code") for group in stats["groups"]}
    check(codes["bad_roughness"] == 111, "roughness stats code mismatch")
    check(codes["bad_anisotropy"] == 112, "anisotropy stats code mismatch")


def validate_init_failure(exe, work):
    run_dir = work / "init_failure"
    _, success, failed = run_renderer(
        exe, ROOT / "configs/init_failure.json", run_dir, run_dir / "out", 2
    )
    check((success, failed) == (0, 0), "init failure summary count mismatch")
    log_text = (run_dir / "render.log").read_text(encoding="utf-8")
    check("config initialization failed" in log_text, "init failure not logged")
    check("field 'width'" in log_text, "init failure detail missing from log")
    check(not list((run_dir / "out").glob("*.ppm")), "init failure wrote PPM output")


def validate_default_size(exe, work):
    config = {
        "seed": 1,
        "tone_mapping": {"mode": "aces", "exposure": 0.8, "gamma": 2.2},
        "light": {"samples_x": 1, "samples_y": 1},
        "materials": [
            {
                "name": "default_size",
                "albedo": [0.7, 0.35, 0.28],
                "roughness": 0.45,
                "anisotropy": 0.3,
            }
        ],
    }
    config_path = work / "default_size.json"
    write_json(config_path, config)
    run_dir = work / "default_size"
    _, success, failed = run_renderer(
        exe, config_path, run_dir, run_dir / "out", 0
    )
    check((success, failed) == (1, 0), "default-size run summary mismatch")
    width, height, _ = parse_ppm(run_dir / "out" / "default_size.ppm")
    check((width, height) == (512, 512), "default dimensions are not 512x512")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exe",
        type=Path,
        default=ROOT / "build" / "skin_renderer",
        help="path to skin_renderer executable",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep temporary validation directories",
    )
    args = parser.parse_args()
    exe = args.exe.resolve()
    check(exe.is_file(), f"renderer executable not found: {exe}")

    work = Path(tempfile.mkdtemp(prefix="skin-renderer-validation-"))
    try:
        first_run = validate_determinism(exe, work)
        print("PASS deterministic PPM output and resource reuse")
        validate_baseline_comparison(exe, work, first_run)
        print("PASS versioned baseline comparison statistics")
        validate_no_overwrite(exe, work)
        print("PASS collision-safe PPM naming")
        validate_failure_isolation(exe, work)
        print("PASS invalid-group isolation and recovery")
        validate_init_failure(exe, work)
        print("PASS initialization failure exit code 2")
        validate_default_size(exe, work)
        print("PASS default 512x512 dimensions")
        print("validation: ALL PASSED")
        return 0
    finally:
        if args.keep:
            print(f"validation artifacts: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"validation: FAILED: {error}", file=sys.stderr)
        sys.exit(1)

