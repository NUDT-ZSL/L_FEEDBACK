#!/usr/bin/env python3
"""End-to-end validation for the deferred snapshot renderer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


class ValidationFailure(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_lines(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def event(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("event") == name]
    if not matches:
        raise ValidationFailure(f"missing JSON event: {name}")
    return matches[-1]


def header(path: Path) -> dict[str, Any]:
    data = path.read_bytes()[:160]
    if len(data) < 160:
        raise ValidationFailure(f"snapshot header is truncated: {path}")
    magic = data[:8]
    format_version = struct.unpack_from("<I", data, 8)[0]
    file_size, light_version = struct.unpack_from("<QQ", data, 16)
    width, height, point_count, directional_count = struct.unpack_from("<IIII", data, 32)
    return {
        "magic": magic.decode("ascii"),
        "format_version": format_version,
        "file_size": file_size,
        "light_version": light_version,
        "width": width,
        "height": height,
        "point_lights": point_count,
        "directional_lights": directional_count,
        "actual_file_size": path.stat().st_size,
    }


class Harness:
    def __init__(self, binary: Path, scene: Path, work: Path) -> None:
        self.binary = binary
        self.scene = scene
        self.work = work
        self.commands: list[dict[str, Any]] = []

    def run(self, *arguments: str, env: dict[str, str] | None = None) -> dict[str, Any]:
        command = [str(self.binary), *arguments]
        completed = subprocess.run(
            command,
            cwd=self.work.parents[1],
            text=True,
            capture_output=True,
            timeout=300,
            env=env,
        )
        stdout_records = json_lines(completed.stdout)
        stderr_records = json_lines(completed.stderr)
        self.commands.append(
            {
                "arguments": list(arguments),
                "returncode": completed.returncode,
                "stdout": stdout_records,
                "stderr": stderr_records,
            }
        )
        return {
            "returncode": completed.returncode,
            "stdout": stdout_records,
            "stderr": stderr_records,
            "stderr_text": completed.stderr,
            "stdout_text": completed.stdout,
        }

    def require_success(self, result: dict[str, Any], label: str) -> None:
        if result["returncode"] != 0:
            raise ValidationFailure(
                f"{label} failed with exit code {result['returncode']}:\n"
                f"stdout:\n{result['stdout_text']}\n"
                f"stderr:\n{result['stderr_text']}"
            )


def versioned(base: Path, version: int) -> Path:
    return base.parent / f"{base.stem}.v{version}{base.suffix or '.dgsnap'}"


def make_corrupt_fixtures(source: Path, directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    checksum = directory / "checksum.dgsnap"
    truncated = directory / "truncated.dgsnap"
    version = directory / "version.dgsnap"
    shutil.copyfile(source, checksum)
    shutil.copyfile(source, truncated)
    shutil.copyfile(source, version)

    data = bytearray(checksum.read_bytes())
    data[50000] ^= 0x5A
    checksum.write_bytes(data)
    truncated.write_bytes(truncated.read_bytes()[:1000])
    data = bytearray(version.read_bytes())
    data[8:12] = (2).to_bytes(4, "little")
    version.write_bytes(data)
    missing = directory / "missing.dgsnap"
    return {
        "checksum": checksum,
        "truncated": truncated,
        "version": version,
        "missing": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/deferred_renderer"))
    parser.add_argument("--asan-binary", type=Path, default=Path("build-asan/deferred_renderer"))
    parser.add_argument("--skip-asan", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    binary = (root / args.binary).resolve()
    asan_binary = (root / args.asan_binary).resolve()
    scene = root / "sample_scene.json"
    work = root / "artifacts" / "validation_run"
    validation_dir = root / "validation"

    if not binary.exists():
        print(f"renderer binary not found: {binary}", file=sys.stderr)
        return 2
    if not scene.exists():
        print(f"scene file not found: {scene}", file=sys.stderr)
        return 2
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    harness = Harness(binary, scene, work)
    live = work / "live.png"
    restore = work / "restore.png"
    snapshot_base = work / "snapshots" / "lighting"
    checks: dict[str, Any] = {}
    snapshot_headers: dict[str, Any] = {}

    save = harness.run(
        "--snapshot-save",
        "--scene",
        str(scene),
        "--snapshot",
        str(snapshot_base),
        "--output",
        str(live),
        "--frames",
        "3",
        "--warmup",
        "1",
    )
    harness.require_success(save, "snapshot-save")
    v1 = versioned(snapshot_base, 1)
    if not v1.exists():
        raise ValidationFailure("v1 snapshot was not written")
    v1_hash = sha256(v1)
    snapshot_headers[v1.name] = header(v1)
    checks["snapshot_saved"] = {
        "path": str(v1),
        "sha256": v1_hash,
        "header": snapshot_headers[v1.name],
    }

    load = harness.run(
        "--snapshot-load",
        "--scene",
        str(scene),
        "--snapshot",
        str(v1),
        "--output",
        str(restore),
        "--compare",
        str(live),
        "--frames",
        "3",
        "--warmup",
        "1",
    )
    harness.require_success(load, "snapshot-load")
    comparison = event(load["stdout"], "image_comparison")
    summary = event(load["stdout"], "summary")
    if summary["mode"] != "snapshot_restore":
        raise ValidationFailure("snapshot restore did not select snapshot_restore mode")
    if comparison["max_channel_difference"] > 1:
        raise ValidationFailure("snapshot restore image differs by more than 1/255")
    checks["restore_pixel_match"] = {
        "max_channel_difference": comparison["max_channel_difference"],
        "different_pixels": comparison["different_pixels"],
        "passed": comparison["passed"],
    }

    added = harness.run(
        "--snapshot-load",
        "--scene",
        str(scene),
        "--snapshot",
        str(v1),
        "--add-light",
        "p64:point:0,3.2,0:1.0,0.35,0.15:4.0:6.0",
        "--output",
        str(work / "added_light.png"),
        "--frames",
        "1",
    )
    harness.require_success(added, "add-light")
    v2 = versioned(snapshot_base, 2)
    if not v2.exists():
        raise ValidationFailure("add-light did not create v2")
    if sha256(v1) != v1_hash:
        raise ValidationFailure("v1 snapshot was overwritten by add-light")
    snapshot_headers[v2.name] = header(v2)
    add_event = event(added["stdout"], "snapshot_saved")
    if snapshot_headers[v2.name]["light_version"] != 2 or add_event["point_lights"] != 65:
        raise ValidationFailure("add-light version/count is incorrect")
    checks["add_light"] = {
        "parent": str(v1),
        "new_snapshot": str(v2),
        "parent_hash_unchanged": True,
        "header": snapshot_headers[v2.name],
    }
    v2_hash = sha256(v2)

    removed = harness.run(
        "--snapshot-load",
        "--scene",
        str(scene),
        "--snapshot",
        str(v2),
        "--remove-light",
        "p64",
        "--output",
        str(work / "removed_light.png"),
        "--compare",
        str(live),
        "--frames",
        "1",
    )
    harness.require_success(removed, "remove-light")
    v3 = versioned(snapshot_base, 3)
    if not v3.exists():
        raise ValidationFailure("remove-light did not create v3")
    if sha256(v2) != v2_hash:
        raise ValidationFailure("v2 snapshot was overwritten by remove-light")
    snapshot_headers[v3.name] = header(v3)
    remove_event = event(removed["stdout"], "snapshot_saved")
    removal_comparison = event(removed["stdout"], "image_comparison")
    if snapshot_headers[v3.name]["light_version"] != 3 or remove_event["point_lights"] != 64:
        raise ValidationFailure("remove-light version/count is incorrect")
    if removal_comparison["max_channel_difference"] > 1:
        raise ValidationFailure("removing the added light did not restore the original image")
    checks["remove_light"] = {
        "parent": str(v2),
        "new_snapshot": str(v3),
        "parent_hash_unchanged": True,
        "header": snapshot_headers[v3.name],
        "restored_image_max_channel_difference": removal_comparison["max_channel_difference"],
    }

    help_result = harness.run("--help")
    harness.require_success(help_result, "help")
    invalid_result = harness.run("--render", "--add-light", "p64:point:0,1,0:1,1,1:1:2")
    if invalid_result["returncode"] != 2:
        raise ValidationFailure("invalid CLI arguments did not return exit code 2")
    arguments_event = event(invalid_result["stderr"], "arguments")
    if (
        arguments_event["code"] != "ERR_ARGUMENTS"
        or "Usage:" not in invalid_result["stderr_text"]
    ):
        raise ValidationFailure("invalid CLI arguments did not print the usage/error code")
    checks["cli"] = {
        "help_exit_code": help_result["returncode"],
        "invalid_exit_code": invalid_result["returncode"],
        "invalid_code": arguments_event["code"],
    }

    corrupt = make_corrupt_fixtures(v1, work / "corrupt")
    expected_codes = {
        "checksum": "ERR_SNAPSHOT_CORRUPT",
        "truncated": "ERR_SNAPSHOT_CORRUPT",
        "version": "ERR_SNAPSHOT_VERSION",
        "missing": "ERR_SNAPSHOT_NOT_FOUND",
    }
    corruption_checks: dict[str, Any] = {}
    for kind, path in corrupt.items():
        fallback = harness.run(
            "--snapshot-load",
            "--scene",
            str(scene),
            "--snapshot",
            str(path),
            "--output",
            str(work / f"fallback_{kind}.png"),
            "--compare",
            str(live),
            "--frames",
            "1",
        )
        harness.require_success(fallback, f"corruption fallback ({kind})")
        error_event = event(fallback["stderr"], "snapshot_load")
        fallback_summary = event(fallback["stdout"], "summary")
        fallback_comparison = event(fallback["stdout"], "image_comparison")
        if error_event["code"] != expected_codes[kind]:
            raise ValidationFailure(
                f"{kind}: expected {expected_codes[kind]}, got {error_event['code']}"
            )
        if "format_version" not in error_event or "light_version" not in error_event:
            raise ValidationFailure(f"{kind}: JSON error lacks version fields")
        if fallback_summary["mode"] != "realtime_fallback":
            raise ValidationFailure(f"{kind}: did not fall back to the real-time path")
        if fallback_comparison["max_channel_difference"] > 1:
            raise ValidationFailure(f"{kind}: fallback image differs from real-time image")
        corruption_checks[kind] = {
            "code": error_event["code"],
            "format_version": error_event["format_version"],
            "light_version": error_event["light_version"],
            "fallback": error_event["fallback"],
            "image_max_channel_difference": fallback_comparison["max_channel_difference"],
        }
    checks["corruption_fallback"] = corruption_checks

    perf_live = harness.run(
        "--render",
        "--scene",
        str(scene),
        "--output",
        str(work / "perf_live.png"),
        "--frames",
        "7",
        "--warmup",
        "2",
    )
    harness.require_success(perf_live, "real-time performance run")
    perf_snapshot = harness.run(
        "--snapshot-load",
        "--scene",
        str(scene),
        "--snapshot",
        str(v1),
        "--output",
        str(work / "perf_snapshot.png"),
        "--compare",
        str(work / "perf_live.png"),
        "--frames",
        "7",
        "--warmup",
        "2",
    )
    harness.require_success(perf_snapshot, "snapshot performance run")
    live_summary = event(perf_live["stdout"], "summary")
    snapshot_summary = event(perf_snapshot["stdout"], "summary")
    perf_comparison = event(perf_snapshot["stdout"], "image_comparison")
    if live_summary["total_lights"] != snapshot_summary["total_lights"]:
        raise ValidationFailure("performance runs used different light counts")
    cpu_ratio = snapshot_summary["cpu_ms_median"] / live_summary["cpu_ms_median"]
    gpu_ratio = snapshot_summary["gpu_ms_median"] / live_summary["gpu_ms_median"]
    if cpu_ratio >= 0.8 or gpu_ratio >= 0.8:
        raise ValidationFailure(
            f"snapshot performance does not meet <80%: CPU {cpu_ratio:.4f}, GPU {gpu_ratio:.4f}"
        )
    if perf_comparison["max_channel_difference"] > 1:
        raise ValidationFailure("performance snapshot image does not match real-time image")
    checks["performance"] = {
        "realtime": {
            "cpu_ms_median": live_summary["cpu_ms_median"],
            "gpu_ms_median": live_summary["gpu_ms_median"],
        },
        "snapshot_restore": {
            "cpu_ms_median": snapshot_summary["cpu_ms_median"],
            "gpu_ms_median": snapshot_summary["gpu_ms_median"],
        },
        "cpu_ratio": cpu_ratio,
        "gpu_ratio": gpu_ratio,
        "threshold": 0.8,
        "image_max_channel_difference": perf_comparison["max_channel_difference"],
        "lights": snapshot_summary["total_lights"],
    }

    asan_check: dict[str, Any] | None = None
    if not args.skip_asan and asan_binary.exists():
        environment = os.environ.copy()
        environment["ASAN_OPTIONS"] = "detect_leaks=1:abort_on_error=0"
        environment["LSAN_OPTIONS"] = f"suppressions={root / 'tools' / 'lsan.suppressions'}"
        asan_outputs: dict[str, Any] = {}
        asan_cases = [
            (
                "snapshot_save",
                [
                    "--snapshot-save",
                    "--snapshot",
                    str(work / "asan_snapshots" / "lighting"),
                    "--output",
                    str(work / "asan_live.png"),
                    "--frames",
                    "1",
                ],
            ),
            (
                "snapshot_restore",
                [
                    "--snapshot-load",
                    "--snapshot",
                    str(v1),
                    "--output",
                    str(work / "asan_restore.png"),
                    "--frames",
                    "1",
                ],
            ),
            (
                "corrupt_fallback",
                [
                    "--snapshot-load",
                    "--snapshot",
                    str(corrupt["checksum"]),
                    "--output",
                    str(work / "asan_fallback.png"),
                    "--compare",
                    str(live),
                    "--frames",
                    "1",
                ],
            ),
        ]
        for label, command in asan_cases:
            result = harness.run(*command, env=environment)
            if result["returncode"] != 0:
                raise ValidationFailure(f"ASan {label} exited with {result['returncode']}")
            sanitizer_text = result["stderr_text"]
            if "ERROR: AddressSanitizer" in sanitizer_text or "ERROR: LeakSanitizer" in sanitizer_text:
                raise ValidationFailure(f"ASan {label} reported an error")
            asan_outputs[label] = {
                "returncode": result["returncode"],
                "sanitizer_errors": 0,
                "image_comparison": (
                    event(result["stdout"], "image_comparison")
                    if "--compare" in command
                    else None
                ),
            }
        asan_check = {
            "binary": str(asan_binary),
            "suppressions": "tools/lsan.suppressions (third-party Mesa/fontconfig caches only)",
            "cases": asan_outputs,
        }
    checks["address_sanitizer"] = asan_check

    report = {
        "scene": str(scene),
        "scene_sha256": sha256(scene),
        "snapshot_headers": snapshot_headers,
        "checks": checks,
        "commands": harness.commands,
    }
    report_path = validation_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    markdown = [
        "# Validation report",
        "",
        f"- Scene: `{scene}`",
        f"- Live image: `{live}`",
        f"- Snapshot restore difference: {checks['restore_pixel_match']['max_channel_difference']}/255",
        f"- Add light: v1 -> v2, 65 point lights, old hash preserved",
        f"- Remove light: v2 -> v3, back to 64 point lights, image difference "
        f"{checks['remove_light']['restored_image_max_channel_difference']}/255",
        f"- Corruption codes: "
        f"{', '.join(item['code'] for item in checks['corruption_fallback'].values())}",
        f"- Performance: CPU ratio {cpu_ratio:.4f}, GPU ratio {gpu_ratio:.4f} "
        f"({checks['performance']['lights']} lights)",
        f"- AddressSanitizer: "
        f"{'run, no application/sanitizer errors' if asan_check else 'skipped'}",
        "",
        f"Machine-readable details: `{report_path}`",
    ]
    markdown_path = validation_dir / "report.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "event": "validation",
                "passed": True,
                "report": str(report_path),
                "restore_max_channel_difference": checks["restore_pixel_match"][
                    "max_channel_difference"
                ],
                "add_light_version": snapshot_headers[v2.name]["light_version"],
                "remove_light_version": snapshot_headers[v3.name]["light_version"],
                "corruption_codes": {
                    key: value["code"] for key, value in checks["corruption_fallback"].items()
                },
                "cpu_ratio": cpu_ratio,
                "gpu_ratio": gpu_ratio,
                "asan_ok": bool(asan_check),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValidationFailure as exception:
        print(f"VALIDATION FAILED: {exception}", file=sys.stderr)
        sys.exit(1)
