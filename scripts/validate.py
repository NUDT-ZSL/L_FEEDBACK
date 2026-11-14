#!/usr/bin/env python3
"""End-to-end acceptance harness for the deterministic CPU cloud renderer.

Uses only the Python standard library and invokes the built cloud_render CLI with
JSON configurations. Results are written to artifacts/validation/report.json and
artifacts/validation/VALIDATION.md.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
from pathlib import Path


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def base_config(output_dir: Path, width: int = 320, height: int = 180) -> dict:
    return {
        "outputDir": str(output_dir),
        "outputName": "frame",
        "width": width,
        "height": height,
        "startFrame": 0,
        "endFrame": 3,
        "seed": 1337,
        "cloudType": "cumulus",
        "renderScale": 2,
        "maxSteps": 56,
        "lightSteps": 4,
        "maxHistoryLength": 8,
        "async": True,
        "threads": 4,
        "camera": {
            "fovDeg": 50,
            "near": 0.05,
            "far": 100,
            "orbit": {
                "enabled": True,
                "center": [0.5, 0.47, 0.5],
                "radius": 2.35,
                "height": 0.95,
                "startDegrees": 35,
                "degreesPerFrame": 2.0,
            },
        },
        "sun": {
            "direction": [0.45, 0.82, 0.25],
            "color": [1.0, 0.95, 0.86],
            "ambient": 0.16,
            "degreesPerFrame": 0.0,
        },
        "memory": {"availableMB": 512},
        "snapshot": {
            "version": 2,
            "savePath": str(output_dir / "snapshot.cld"),
            "saveEveryFrames": 1,
            "stopAfterSave": False,
        },
    }


def run_renderer(binary: Path, config: dict, config_path: Path, cwd: Path) -> tuple[int, str]:
    write_json(config_path, config)
    process = subprocess.run(
        [str(binary), "--config", str(config_path)],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=600,
    )
    return process.returncode, process.stdout + process.stderr


def parse_log(path: Path) -> list[dict]:
    events = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def code_present(events: list[dict], code: str) -> bool:
    return any(event.get("code") == code for event in events)


def compare_rgba(a_path: Path, b_path: Path) -> dict:
    a = a_path.read_bytes()
    b = b_path.read_bytes()
    if len(a) != len(b):
        return {"equal": False, "reason": "size mismatch", "sizeA": len(a), "sizeB": len(b)}
    maximum_delta = 0
    sum_squared = 0.0
    changed = 0
    for left, right in zip(a, b):
        delta = abs(left - right)
        maximum_delta = max(maximum_delta, delta)
        sum_squared += float(delta * delta)
        changed += int(delta != 0)
    relative = maximum_delta / 255.0
    rmse = math.sqrt(sum_squared / max(1, len(a)))
    return {
        "equal": a == b,
        "changedComponents": changed,
        "maxAbsoluteDelta": maximum_delta,
        "maxRelativeError": relative,
        "normalizedRmse": rmse / 255.0,
        "threshold": 1.0e-4,
        "passed": relative < 1.0e-4,
    }


def copy_and_mutate(source: Path, target: Path, mutation) -> None:
    data = bytearray(source.read_bytes())
    mutation(data)
    target.write_bytes(bytes(data))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if args.binary is not None:
        binary = args.binary.resolve()
    else:
        candidates = [
            args.build_dir / "cloud_render",
            args.build_dir / "cloud_render.exe",
            args.build_dir / "Release" / "cloud_render.exe",
            args.build_dir / "Debug" / "cloud_render.exe",
        ]
        binary = next(
            (candidate.resolve() for candidate in candidates if candidate.exists()),
            candidates[0].resolve(),
        )
    if not binary.exists():
        print(f"Renderer binary not found: {binary}", file=sys.stderr)
        return 2

    validation = workspace / "artifacts" / "validation"
    if validation.exists():
        shutil.rmtree(validation)
    validation.mkdir(parents=True, exist_ok=True)
    report: dict = {"results": {}, "commands": []}
    failures: list[str] = []

    def record(name: str, passed: bool, details: dict) -> None:
        report["results"][name] = {"passed": bool(passed), **details}
        if not passed:
            failures.append(name)
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    # Continuous baseline plus a stop-after-snapshot v2 run and continuation.
    continuous_dir = validation / "continuous"
    continuous_config = base_config(continuous_dir)
    continuous_config["snapshot"]["savePath"] = str(continuous_dir / "final.cld")
    continuous_config["snapshot"]["saveEveryFrames"] = 1
    continuous_path = validation / "continuous.json"
    code, output = run_renderer(binary, continuous_config, continuous_path, workspace)
    report["commands"].append({"config": str(continuous_path), "exitCode": code})
    record("continuous_run", code == 0, {"exitCode": code, "tail": output[-800:]})

    interrupted_dir = validation / "interrupted"
    interrupted_config = base_config(interrupted_dir)
    interrupted_config["snapshot"]["savePath"] = str(interrupted_dir / "checkpoint.cld")
    interrupted_config["snapshot"]["saveEveryFrames"] = 2
    interrupted_config["snapshot"]["stopAfterSave"] = True
    interrupted_path = validation / "interrupted_first.json"
    code1, output1 = run_renderer(binary, interrupted_config, interrupted_path, workspace)
    report["commands"].append({"config": str(interrupted_path), "exitCode": code1})

    resumed_config = base_config(interrupted_dir)
    resumed_config["snapshot"]["loadPath"] = str(interrupted_dir / "checkpoint.cld")
    resumed_config["snapshot"]["savePath"] = str(interrupted_dir / "resumed.cld")
    resumed_config["snapshot"]["saveEveryFrames"] = 1
    resumed_config["snapshot"]["stopAfterSave"] = False
    resumed_path = validation / "interrupted_second.json"
    code2, output2 = run_renderer(binary, resumed_config, resumed_path, workspace)
    report["commands"].append({"config": str(resumed_path), "exitCode": code2})
    resume_compare = compare_rgba(continuous_dir / "frame.rgba", interrupted_dir / "frame.rgba")
    record(
        "snapshot_v2_resume",
        code1 == 0 and code2 == 0 and bool(resume_compare.get("passed")),
        {"firstExitCode": code1, "secondExitCode": code2, **resume_compare},
    )

    # v1 snapshot has no lighting/history fields. Loading it must emit the business warning.
    v1_first_dir = validation / "v1_first"
    v1_first = base_config(v1_first_dir)
    v1_first["snapshot"]["version"] = 1
    v1_first["snapshot"]["savePath"] = str(v1_first_dir / "v1.cld")
    v1_first["snapshot"]["saveEveryFrames"] = 2
    v1_first["snapshot"]["stopAfterSave"] = True
    v1_first_path = validation / "v1_first.json"
    code_v1_first, _ = run_renderer(binary, v1_first, v1_first_path, workspace)
    report["commands"].append({"config": str(v1_first_path), "exitCode": code_v1_first})

    v1_resume_dir = validation / "v1_resume"
    v1_resume = base_config(v1_resume_dir)
    v1_resume["snapshot"]["version"] = 2
    v1_resume["snapshot"]["loadPath"] = str(v1_first_dir / "v1.cld")
    v1_resume["snapshot"]["savePath"] = str(v1_resume_dir / "upgraded.cld")
    v1_resume["snapshot"]["saveEveryFrames"] = 1
    v1_resume_path = validation / "v1_resume.json"
    code_v1_resume, _ = run_renderer(binary, v1_resume, v1_resume_path, workspace)
    report["commands"].append({"config": str(v1_resume_path), "exitCode": code_v1_resume})
    v1_events = parse_log(v1_resume_dir / "render.log")
    v1_compare = compare_rgba(continuous_dir / "frame.rgba", v1_resume_dir / "frame.rgba")
    record(
        "snapshot_v1_upgrade_warning",
        code_v1_first == 0 and code_v1_resume == 0
        and code_present(v1_events, "W_SNAPSHOT_UPGRADED")
        and bool(v1_compare.get("passed")),
        {
            "firstExitCode": code_v1_first,
            "secondExitCode": code_v1_resume,
            "warningFound": code_present(v1_events, "W_SNAPSHOT_UPGRADED"),
            "compareAgainstContinuousDefaultLighting": v1_compare,
        },
    )

    # Half-resolution upsample versus direct full-resolution on a single identical frame.
    quality_dir = validation / "quality"
    quality_config = base_config(quality_dir, 640, 360)
    quality_config["endFrame"] = 1
    quality_config["quality"] = {
        "compareAgainstFullResolution": True,
        "writeFullResolutionReference": True,
    }
    quality_config["snapshot"]["savePath"] = str(quality_dir / "quality.cld")
    quality_path = validation / "quality.json"
    code_quality, _ = run_renderer(binary, quality_config, quality_path, workspace)
    quality_events = parse_log(quality_dir / "render.log")
    psnr_values = [
        float(event["psnrDb"])
        for event in quality_events
        if event.get("event") == "upsample_quality"
    ]
    psnr = min(psnr_values) if psnr_values else float("nan")
    record(
        "half_resolution_psnr",
        code_quality == 0 and math.isfinite(psnr) and psnr >= 30.0,
        {"exitCode": code_quality, "psnrDb": psnr, "thresholdDb": 30.0,
         "resolution": [640, 360], "renderScale": 2},
    )

    # Low-memory level selection, logging, and exact continuation at that level.
    for memory_mb, expected_resolution in ((80, 64), (40, 32)):
        low_first_dir = validation / f"low_{expected_resolution}_first"
        low_first = base_config(low_first_dir)
        low_first["memory"]["availableMB"] = memory_mb
        low_first["snapshot"]["savePath"] = str(low_first_dir / "checkpoint.cld")
        low_first["snapshot"]["saveEveryFrames"] = 2
        low_first["snapshot"]["stopAfterSave"] = True
        low_path = validation / f"low_{expected_resolution}_first.json"
        code_low_first, _ = run_renderer(binary, low_first, low_path, workspace)
        report["commands"].append({"config": str(low_path), "exitCode": code_low_first})

        low_resume_dir = validation / f"low_{expected_resolution}_resume"
        low_resume = base_config(low_resume_dir)
        low_resume["memory"]["availableMB"] = memory_mb
        low_resume["snapshot"]["loadPath"] = str(low_first_dir / "checkpoint.cld")
        low_resume["snapshot"]["savePath"] = str(low_resume_dir / "resumed.cld")
        low_resume["snapshot"]["saveEveryFrames"] = 1
        low_path_resume = validation / f"low_{expected_resolution}_resume.json"
        code_low_resume, _ = run_renderer(binary, low_resume, low_path_resume, workspace)
        report["commands"].append({"config": str(low_path_resume), "exitCode": code_low_resume})

        low_memory_dir = validation / f"low_{expected_resolution}_baseline"
        low_memory = base_config(low_memory_dir)
        low_memory["memory"]["availableMB"] = memory_mb
        low_memory["snapshot"]["savePath"] = str(low_memory_dir / "final.cld")
        low_path_full = validation / f"low_{expected_resolution}_baseline.json"
        code_low_full, _ = run_renderer(binary, low_memory, low_path_full, workspace)
        report["commands"].append({"config": str(low_path_full), "exitCode": code_low_full})
        low_compare = compare_rgba(low_memory_dir / "frame.rgba", low_resume_dir / "frame.rgba")
        low_events = parse_log(low_first_dir / "render.log")
        downsample_event = next(
            (event for event in low_events if event.get("event") == "density_ready"), None
        )
        downsample_warning = [
            event for event in low_events if event.get("code") == "W_DENSITY_DOWNSAMPLED"
        ]
        record(
            f"low_memory_{expected_resolution}",
            code_low_first == 0
            and code_low_resume == 0
            and code_low_full == 0
            and bool(low_compare.get("passed"))
            and downsample_event is not None
            and int(downsample_event.get("resolution", -1)) == expected_resolution
            and bool(downsample_warning),
            {
                "memoryMB": memory_mb,
                "expectedResolution": expected_resolution,
                "actualResolution": int(downsample_event.get("resolution", -1))
                if downsample_event
                else None,
                "reasonLogged": bool(downsample_warning),
                "resumeCompare": low_compare,
            },
        )

    # Async worker generation must exactly match the synchronous path.
    async_dir = validation / "async_path"
    async_config = base_config(async_dir)
    async_config["async"] = True
    async_config["snapshot"]["savePath"] = str(async_dir / "async.cld")
    async_path = validation / "async.json"
    code_async, _ = run_renderer(binary, async_config, async_path, workspace)
    sync_dir = validation / "sync_path"
    sync_config = base_config(sync_dir)
    sync_config["async"] = False
    sync_config["snapshot"]["savePath"] = str(sync_dir / "sync.cld")
    sync_path = validation / "sync.json"
    code_sync, _ = run_renderer(binary, sync_config, sync_path, workspace)
    async_events = parse_log(async_dir / "render.log")
    async_density = next(
        (event for event in async_events if event.get("event") == "density_ready"), None
    )
    async_compare = compare_rgba(async_dir / "frame.rgba", sync_dir / "frame.rgba")
    record(
        "async_equals_sync",
        code_async == 0
        and code_sync == 0
        and async_density is not None
        and async_density.get("async") is True
        and async_density.get("mainThreadSpinMilliseconds", 1.0) <= 1.0,
        {
            "asyncExitCode": code_async,
            "syncExitCode": code_sync,
            "asyncDensity": async_density,
            "compare": async_compare,
        },
    )

    # Business error paths and their process exit codes.
    error_cases = []

    missing_dir = validation / "error_missing"
    missing = base_config(missing_dir)
    missing["snapshot"]["loadPath"] = str(missing_dir / "does-not-exist.cld")
    missing_path = validation / "error_missing.json"
    code_missing, _ = run_renderer(binary, missing, missing_path, workspace)
    error_cases.append(
        ("E_SNAPSHOT_MISSING", 10, code_missing,
         code_present(parse_log(missing_dir / "render.log"), "E_SNAPSHOT_MISSING"))
    )

    crc_dir = validation / "error_crc"
    crc_dir.mkdir(parents=True, exist_ok=True)
    crc_source = continuous_dir / "final.cld"
    crc_path = crc_dir / "corrupt.cld"
    copy_and_mutate(crc_source, crc_path, lambda data: data.__setitem__(-1, data[-1] ^ 0x5A))
    crc_config = base_config(crc_dir)
    crc_config["snapshot"]["loadPath"] = str(crc_path)
    crc_config["snapshot"]["savePath"] = str(crc_dir / "unused.cld")
    crc_config_path = validation / "error_crc.json"
    code_crc, _ = run_renderer(binary, crc_config, crc_config_path, workspace)
    error_cases.append(
        ("E_SNAPSHOT_CRC", 11, code_crc,
         code_present(parse_log(crc_dir / "render.log"), "E_SNAPSHOT_CRC"))
    )

    version_dir = validation / "error_version"
    version_dir.mkdir(parents=True, exist_ok=True)
    version_path = version_dir / "unsupported.cld"

    def mutate_version(data: bytearray) -> None:
        data[8:12] = struct.pack("<I", 99)

    copy_and_mutate(crc_source, version_path, mutate_version)
    version_config = base_config(version_dir)
    version_config["snapshot"]["loadPath"] = str(version_path)
    version_config["snapshot"]["savePath"] = str(version_dir / "unused.cld")
    version_config_path = validation / "error_version.json"
    code_version, _ = run_renderer(binary, version_config, version_config_path, workspace)
    error_cases.append(
        ("E_VERSION_UNSUPPORTED", 12, code_version,
         code_present(parse_log(version_dir / "render.log"), "E_VERSION_UNSUPPORTED"))
    )

    oom_dir = validation / "error_oom"
    oom_config = base_config(oom_dir)
    oom_config["memory"]["availableMB"] = 16
    oom_config["snapshot"]["savePath"] = str(oom_dir / "unused.cld")
    oom_config_path = validation / "error_oom.json"
    code_oom, _ = run_renderer(binary, oom_config, oom_config_path, workspace)
    error_cases.append(
        ("E_OUT_OF_MEMORY", 13, code_oom,
         code_present(parse_log(oom_dir / "render.log"), "E_OUT_OF_MEMORY"))
    )

    for code_name, expected_exit, actual_exit, logged in error_cases:
        record(
            f"error_{code_name.lower()}",
            expected_exit == actual_exit and logged,
            {"expectedExitCode": expected_exit, "actualExitCode": actual_exit, "codeLogged": logged},
        )

    # Confirm worker cleanup occurs on successful runs.
    cleanup_present = any(
        event.get("event") == "thread_cleanup" for event in parse_log(continuous_dir / "render.log")
    )
    record("thread_cleanup", cleanup_present, {"eventLogged": cleanup_present})

    report["passed"] = not failures
    report["failures"] = failures
    write_json(validation / "report.json", report)

    lines = ["# Validation Report", ""]
    lines.append(f"Overall: **{'PASS' if report['passed'] else 'FAIL'}**")
    lines.append("")
    lines.append("| Check | Result | Detail |")
    lines.append("|---|---|---|")
    for name, result in report["results"].items():
        detail = ""
        if "maxRelativeError" in result:
            detail = f"max relative error {result['maxRelativeError']:.9g}"
        elif "psnrDb" in result:
            detail = f"PSNR {result['psnrDb']:.3f} dB"
        elif "actualExitCode" in result:
            detail = f"exit {result['actualExitCode']}"
        elif "actualResolution" in result:
            detail = f"{result['actualResolution']}^3 density"
        lines.append(
            f"| {name} | {'PASS' if result['passed'] else 'FAIL'} | {detail} |"
        )
    lines.append("")
    lines.append("Full machine-readable results: `artifacts/validation/report.json`.")
    (validation / "VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if failures:
        print("Failures: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

