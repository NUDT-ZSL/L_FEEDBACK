#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cinemix.processor import process_project, write_report
from cinemix.wavio import riff_info, wav_sha256
from scripts.generate_samples import build_reference, generate


def _ffmpeg_measure(ffmpeg: Path, wav: Path) -> dict[str, float]:
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-nostats",
        "-i",
        str(wav),
        "-filter_complex",
        "ebur128=peak=true",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"FFmpeg measurement failed ({completed.returncode}): {completed.stderr[-1000:]}"
        )
    summary = completed.stderr.rsplit("Summary:", 1)[-1]
    integrated_match = re.search(r"\bI:\s+(-?\d+(?:\.\d+)?)\s+LUFS", summary)
    peak_match = re.search(r"\bPeak:\s+(-?\d+(?:\.\d+)?)\s+dBFS", summary)
    if not integrated_match or not peak_match:
        raise RuntimeError("Could not parse FFmpeg ebur128 summary")
    return {
        "integrated_lufs": float(integrated_match.group(1)),
        "true_peak_dbtp": float(peak_match.group(1)),
    }


def _run_cli(config: Path, *, extra: list[str] | None = None) -> subprocess.CompletedProcess:
    command = [
        sys.executable,
        "-m",
        "cinemix",
        "process",
        str(config),
    ]
    if extra:
        command.extend(extra)
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _stderr_json(completed: subprocess.CompletedProcess) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in completed.stderr.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return values


def _error_case(
    project_dir: Path,
    relative_config: str,
    expected_code: str,
    expected_exit: int,
) -> dict[str, Any]:
    completed = _run_cli(project_dir / relative_config)
    issues = _stderr_json(completed)
    codes = [issue.get("code") for issue in issues]
    passed = completed.returncode == expected_exit and expected_code in codes
    return {
        "config": relative_config,
        "expected_code": expected_code,
        "expected_exit": expected_exit,
        "actual_exit": completed.returncode,
        "stderr_codes": codes,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate samples and verify Cinemix acceptance criteria"
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=ROOT / "generated" / "acceptance",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "artifacts" / "acceptance_report.json",
    )
    parser.add_argument("--ffmpeg", type=Path, default=None)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    generate(project_dir, 71)
    reference = build_reference(project_dir, 71)
    checks: dict[str, Any] = {}
    all_passed = True

    ffmpeg = args.ffmpeg
    if ffmpeg is None:
        discovered = shutil.which("ffmpeg")
        ffmpeg = Path(discovered) if discovered else None

    for name in ("broadcast", "hot", "stress"):
        report = json.loads(
            (project_dir / f"reference_{name}_report.json").read_text(
                encoding="utf-8"
            )
        )
        expected = reference["projects"][name]
        output = Path(report["output"])
        info = riff_info(output)
        loudness_error = abs(
            report["measurement"]["integrated_lufs"] - report["target"]["integrated_lufs"]
        )
        peak_ok = (
            report["measurement"]["true_peak_dbtp"]
            <= report["target"]["true_peak_dbtp"] + 1e-9
        )
        boundary_ok = (
            report["maximum_crossfade_boundary_error_ms"] <= 1.0
        )
        format_ok = (
            info["format_tag"] == 1
            and info["bit_depth"] == 24
            and info["sample_rate"] == report["format"]["sample_rate"]
            and info["channels"] == report["format"]["channels"]
        )
        reference_ok = (
            abs(
                report["measurement"]["integrated_lufs"]
                - expected["expected_integrated_lufs"]
            )
            <= 1e-5
            and abs(
                report["measurement"]["true_peak_dbtp"]
                - expected["expected_true_peak_dbtp"]
            )
            <= 1e-5
        )
        checks[name] = {
            "target_lufs": report["target"]["integrated_lufs"],
            "measured_lufs": report["measurement"]["integrated_lufs"],
            "loudness_error_lu": round(loudness_error, 6),
            "within_0_5_lu": loudness_error <= 0.5,
            "true_peak_dbtp": report["measurement"]["true_peak_dbtp"],
            "true_peak_limit_dbtp": report["target"]["true_peak_dbtp"],
            "true_peak_within_limit": peak_ok,
            "max_crossfade_boundary_error_ms": report[
                "maximum_crossfade_boundary_error_ms"
            ],
            "crossfade_within_1ms": boundary_ok,
            "output_format_ok": format_ok,
            "reference_values_match": reference_ok,
            "limiter_max_reduction_db": report["limiter"]["max_reduction_db"],
            "adaptive_trim_db": (
                report["tracks"][0]["adaptive_trim_applied_db"]
                if report["tracks"]
                else 0.0
            ),
        }
        if ffmpeg is not None:
            independent = _ffmpeg_measure(ffmpeg, output)
            checks[name]["ffmpeg"] = independent
            checks[name]["ffmpeg_loudness_delta_lu"] = round(
                abs(independent["integrated_lufs"] - report["measurement"]["integrated_lufs"]),
                6,
            )
            checks[name]["ffmpeg_loudness_agrees"] = (
                checks[name]["ffmpeg_loudness_delta_lu"] <= 0.2
            )
        else:
            checks[name]["ffmpeg"] = None
        checks[name]["passed"] = all(
            (
                checks[name]["within_0_5_lu"],
                checks[name]["true_peak_within_limit"],
                checks[name]["crossfade_within_1ms"],
                checks[name]["output_format_ok"],
                checks[name]["reference_values_match"],
                checks[name].get("ffmpeg_loudness_agrees", True),
            )
        )
        all_passed = all_passed and checks[name]["passed"]

    repeat_output = project_dir / "out" / "broadcast_repeat.wav"
    process_project(
        project_dir / "project.json",
        output_override=repeat_output,
    )
    reference_hash = reference["projects"]["broadcast"]["output_sha256"]
    repeat_hash = wav_sha256(repeat_output)
    determinism = {
        "reference_sha256": reference_hash,
        "repeat_sha256": repeat_hash,
        "bit_identical": reference_hash == repeat_hash,
    }
    all_passed = all_passed and determinism["bit_identical"]

    error_checks = [
        _error_case(
            project_dir,
            "errors/missing_track.json",
            "E_INPUT_MISSING",
            20,
        ),
        _error_case(
            project_dir,
            "errors/invalid_field.json",
            "E_CONFIG_FIELD",
            12,
        ),
        _error_case(
            project_dir,
            "errors/sample_rate_mismatch.json",
            "E_INPUT_SAMPLE_RATE",
            22,
        ),
        _error_case(
            project_dir,
            "errors/crossfade_conflict.json",
            "E_CROSSFADE_CONFLICT",
            30,
        ),
    ]
    clamped = _run_cli(project_dir / "errors" / "gain_clamped.json")
    clamped_issues = _stderr_json(clamped)
    clamped_codes = [issue.get("code") for issue in clamped_issues]
    clamped_check = {
        "config": "errors/gain_clamped.json",
        "expected_warning": "W_GAIN_CLAMPED",
        "actual_exit": clamped.returncode,
        "stderr_codes": clamped_codes,
        "passed": clamped.returncode == 0 and "W_GAIN_CLAMPED" in clamped_codes,
    }
    all_passed = all_passed and all(item["passed"] for item in error_checks)
    all_passed = all_passed and clamped_check["passed"]

    artifact = {
        "ok": all_passed,
        "generated_project": str(project_dir),
        "reference": str(project_dir / "reference.json"),
        "ffmpeg": str(ffmpeg) if ffmpeg else None,
        "projects": checks,
        "determinism": determinism,
        "error_injection": error_checks,
        "recoverable_warning": clamped_check,
    }
    write_report(args.artifact.resolve(), artifact)
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
