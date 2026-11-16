#!/usr/bin/env python3
"""Validate produced WAV/report pairs against the acceptance thresholds."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dialogue_localizer.loudness import SAMPLE_RATE, integrated_loudness


def _linear_to_db(value: float) -> float:
    return 20.0 * math.log10(value) if value > 0.0 else -math.inf


def verify(
    output_path: Path,
    report_path: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    info = sf.info(str(output_path))
    audio, sample_rate = sf.read(
        str(output_path), dtype="float64", always_2d=True
    )

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, **details: Any) -> None:
        checks.append(
            {"name": name, "passed": bool(passed), **details}
        )

    check(
        "wav_sample_rate",
        sample_rate == 48_000,
        actual=sample_rate,
        expected=48_000,
    )
    check(
        "wav_channels",
        audio.shape[1] == 2,
        actual=audio.shape[1],
        expected=2,
    )
    check(
        "wav_pcm16",
        info.subtype == "PCM_16",
        actual=info.subtype,
        expected="PCM_16",
    )
    video_duration = float(report["video"]["duration_sec"])
    expected_frames = int(round(video_duration * SAMPLE_RATE))
    check(
        "wav_video_duration",
        audio.shape[0] == expected_frames,
        actual_frames=audio.shape[0],
        expected_frames=expected_frames,
    )
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        check(
            "video_duration_matches_manifest",
            abs(float(manifest["video"]["duration"]) - video_duration) < 1e-9,
            manifest_duration=float(manifest["video"]["duration"]),
            report_duration=video_duration,
        )

    for segment in report["segments"]:
        segment_id = segment["id"]
        start = float(segment["actual_start_sec"])
        end = float(segment["actual_end_sec"])
        check(
            f"{segment_id}.timeline_bounds",
            0.0 <= start <= end <= video_duration + 1e-9,
            start_sec=start,
            end_sec=end,
            video_duration_sec=video_duration,
        )
        check(
            f"{segment_id}.crossfade_limit",
            float(segment["crossfade_ms"]) <= 10.0 + 1e-9,
            crossfade_ms=float(segment["crossfade_ms"]),
        )
        if end <= start:
            continue
        expected_start = int(round(start * SAMPLE_RATE))
        expected_end = int(round(end * SAMPLE_RATE))
        excerpt = audio[expected_start:expected_end]
        measured_loudness = integrated_loudness(excerpt)
        measured_peak = _linear_to_db(
            float(np.max(np.abs(excerpt)))
        )
        reported_loudness = segment["integrated_loudness_lufs"]
        reported_peak = segment["peak_dbfs"]
        check(
            f"{segment_id}.reported_loudness_matches_wav",
            reported_loudness is not None
            and abs(float(reported_loudness) - measured_loudness) <= 0.01,
            reported=reported_loudness,
            measured=measured_loudness,
        )
        check(
            f"{segment_id}.reported_peak_matches_wav",
            reported_peak is not None
            and abs(float(reported_peak) - measured_peak) <= 0.01,
            reported=reported_peak,
            measured=measured_peak,
        )
        check(
            f"{segment_id}.loudness_tolerance",
            reported_loudness is not None
            and abs(float(reported_loudness) + 23.0) <= 1.0,
            lufs=reported_loudness,
            target_lufs=-23.0,
            tolerance_lu=1.0,
        )
        check(
            f"{segment_id}.peak_limit",
            reported_peak is not None and float(reported_peak) <= -1.0,
            peak_dbfs=reported_peak,
            ceiling_dbfs=-1.0,
        )

    passed = all(item["passed"] for item in checks)
    return {
        "status": "ok" if passed else "error",
        "code": "ACCEPTANCE_PASSED" if passed else "ACCEPTANCE_FAILED",
        "output": str(output_path.resolve()),
        "report": str(report_path.resolve()),
        "checks": checks,
        "passed_checks": sum(item["passed"] for item in checks),
        "failed_checks": sum(not item["passed"] for item in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--manifest")
    args = parser.parse_args()
    result = verify(
        Path(args.output),
        Path(args.report),
        Path(args.manifest) if args.manifest else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

