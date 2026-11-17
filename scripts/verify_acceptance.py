#!/usr/bin/env python3
"""Run deterministic acceptance checks and write a JSON report."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialogue_mixer.audio_io import decode_wav_float32
from dialogue_mixer.engine import EXIT_FATAL, EXIT_PARTIAL, EXIT_SUCCESS, run_project
from dialogue_mixer.samplegen import generate_all


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _db(value: float) -> float | None:
    if value <= 0.0:
        return None
    return float(20.0 * np.log10(value))


def _sample_hashes(sample_root: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for path in sorted(sample_root.rglob("*")):
        if not path.is_file():
            continue
        if "output" in path.parts:
            continue
        hashes[str(path.relative_to(sample_root))] = _hash(path)
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/acceptance_report.json"),
    )
    args = parser.parse_args()
    sample_root = args.samples.expanduser().resolve()
    report_path = args.report.expanduser().resolve()

    projects = generate_all(sample_root)
    sample_hashes_before = _sample_hashes(sample_root)
    valid = run_project(projects["valid"])
    invalid = run_project(projects["invalid"])
    fatal = run_project(projects["fatal"])

    valid_log = _load_json(valid.log_path)
    valid_index = _load_json(valid.index_path)
    valid_audio, valid_rate = decode_wav_float32(valid.audio_path)
    invalid_log = _load_json(invalid.log_path)
    invalid_index = _load_json(invalid.index_path)
    fatal_log = _load_json(fatal.log_path)

    valid_peak = float(np.max(np.abs(valid_audio)))
    valid_ceiling = float(valid_log["normalization"]["peak_ceiling_linear"])
    alignment = float(valid_index["alignment"]["max_alignment_error_samples"])
    dialogue_track = next(
        track for track in valid_index["tracks"] if track["id"] == "dialogue"
    )
    first_dialogue = next(
        segment
        for segment in dialogue_track["segments"]
        if segment["dialogue_id"] == "line_zh_01"
    )

    checks: List[Dict[str, Any]] = []

    def check(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "actual": actual,
                "expected": expected,
            }
        )

    check("valid exit code", valid.exit_code == EXIT_SUCCESS, valid.exit_code, 0)
    check("valid status", valid.status == "success", valid.status, "success")
    check("WAV sample rate", valid_rate == 48000, valid_rate, 48000)
    check(
        "WAV sample count",
        int(valid_audio.size) == int(valid_index["total_samples"]),
        int(valid_audio.size),
        int(valid_index["total_samples"]),
    )
    check(
        "peak <= 0 dBFS",
        valid_peak <= 1.0,
        _db(valid_peak),
        "<= 0.0 dBFS",
    )
    check(
        "peak <= configured ceiling",
        valid_peak <= valid_ceiling + 1e-7,
        _db(valid_peak),
        "<= {0:.6f} dBFS".format(_db(valid_ceiling) or 0.0),
    )
    check(
        "normalization factor applied",
        valid_log["normalization"]["applied"]
        and valid_log["normalization"]["normalization_factor"] < 1.0,
        valid_log["normalization"]["normalization_factor"],
        "< 1.0",
    )
    check(
        "index alignment <= 1 sample",
        alignment <= 1.0,
        alignment,
        "<= 1",
    )
    check(
        "first stretched segment interval",
        first_dialogue["start_sample"] == 48000
        and first_dialogue["duration_samples"] == 57600
        and first_dialogue["source_kind"] == "stretched",
        {
            "start_sample": first_dialogue["start_sample"],
            "duration_samples": first_dialogue["duration_samples"],
            "source_kind": first_dialogue["source_kind"],
        },
        {
            "start_sample": 48000,
            "duration_samples": 57600,
            "source_kind": "stretched",
        },
    )
    check(
        "ducking and overlap counts",
        valid_log["counts"]["duck_events"] == 8
        and valid_log["counts"]["overlap_events"] == 1,
        {
            "duck_events": valid_log["counts"]["duck_events"],
            "overlap_events": valid_log["counts"]["overlap_events"],
        },
        {"duck_events": 8, "overlap_events": 1},
    )
    check(
        "equal-power crossfade metadata",
        valid_log["crossfades"]["transitions"][0]["curve"] == "equal_power"
        and valid_log["crossfades"]["independent_of_ducking"],
        {
            "curve": valid_log["crossfades"]["transitions"][0]["curve"],
            "independent_of_ducking": valid_log["crossfades"][
                "independent_of_ducking"
            ],
        },
        {"curve": "equal_power", "independent_of_ducking": True},
    )

    expected_partial_codes = {
        "MISSING_AUDIO_FILE",
        "MISSING_DUCK_TARGET",
        "UNMERGEABLE_OVERLAP",
    }
    actual_partial_codes = set(invalid_log["business_error_codes"])
    check(
        "partial exit code",
        invalid.exit_code == EXIT_PARTIAL,
        invalid.exit_code,
        EXIT_PARTIAL,
    )
    check(
        "partial business codes",
        expected_partial_codes.issubset(actual_partial_codes),
        sorted(actual_partial_codes),
        sorted(expected_partial_codes),
    )
    check(
        "partial outputs retained",
        invalid.audio_path is not None
        and invalid.audio_path.is_file()
        and invalid.index_path is not None
        and invalid.index_path.is_file(),
        {
            "audio_exists": bool(invalid.audio_path and invalid.audio_path.is_file()),
            "index_exists": bool(invalid.index_path and invalid.index_path.is_file()),
        },
        {"audio_exists": True, "index_exists": True},
    )
    check(
        "partial index alignment <= 1 sample",
        float(invalid_index["alignment"]["max_alignment_error_samples"]) <= 1.0,
        invalid_index["alignment"]["max_alignment_error_samples"],
        "<= 1",
    )
    check(
        "fatal exit code",
        fatal.exit_code == EXIT_FATAL,
        fatal.exit_code,
        EXIT_FATAL,
    )
    check(
        "invalid sample rate code",
        "INVALID_SAMPLE_RATE" in fatal_log["business_error_codes"],
        fatal_log["business_error_codes"],
        ["INVALID_SAMPLE_RATE"],
    )
    check(
        "fatal run emits no mix",
        not (fatal.log_path.parent / "should_not_exist.wav").exists(),
        (fatal.log_path.parent / "should_not_exist.wav").exists(),
        False,
    )

    first_audio_hash = _hash(valid.audio_path)
    first_index_hash = _hash(valid.index_path)
    rerun = run_project(projects["valid"])
    check(
        "deterministic rerun",
        rerun.exit_code == EXIT_SUCCESS
        and _hash(rerun.audio_path) == first_audio_hash
        and _hash(rerun.index_path) == first_index_hash,
        {
            "audio_sha256": _hash(rerun.audio_path),
            "index_sha256": _hash(rerun.index_path),
        },
        {
            "audio_sha256": first_audio_hash,
            "index_sha256": first_index_hash,
        },
    )
    generate_all(sample_root)
    check(
        "deterministic sample generation",
        _sample_hashes(sample_root) == sample_hashes_before,
        {
            "files_checked": len(sample_hashes_before),
            "hashes_unchanged": _sample_hashes(sample_root) == sample_hashes_before,
        },
        {"files_checked": len(sample_hashes_before), "hashes_unchanged": True},
    )

    try:
        ffmpeg_version = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        ).stdout.splitlines()[0]
    except (OSError, IndexError):
        ffmpeg_version = "unavailable"

    passed = all(item["passed"] for item in checks)
    report = {
        "schema_version": 1,
        "tool": "dialogue-mixer-acceptance",
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "ffmpeg": ffmpeg_version,
        },
        "samples": {
            "root": str(sample_root),
            "valid_project": str(projects["valid"]),
            "invalid_project": str(projects["invalid"]),
            "fatal_project": str(projects["fatal"]),
        },
        "summary": {
            "passed": passed,
            "checks_passed": sum(1 for item in checks if item["passed"]),
            "checks_total": len(checks),
        },
        "checks": checks,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for item in checks:
        print(
            "{0}: {1} (actual={2!r})".format(
                "PASS" if item["passed"] else "FAIL",
                item["check"],
                item["actual"],
            )
        )
    print("report={0}".format(report_path))
    print(
        "summary={0}/{1} passed".format(
            report["summary"]["checks_passed"],
            report["summary"]["checks_total"],
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

