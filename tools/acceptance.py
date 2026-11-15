#!/usr/bin/env python3
"""Run deterministic end-to-end acceptance checks and print a JSON report."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from audiorepair import loudness, wavio
from audiorepair.batch import load_batch, run_batch
from audiorepair.errors import (
    INPUT_MISSING,
    STRETCH_UNREACHABLE,
    TIMELINE_INVALID,
    TIMELINE_OUT_OF_RANGE,
    RepairError,
)
from audiorepair.pipeline import run_job
from tools.generate_samples import generate


def _check(condition: bool, name: str, detail: Any = None) -> Dict[str, Any]:
    result = {"name": name, "passed": bool(condition)}
    if detail is not None:
        result["detail"] = detail
    return result


def _error_code(callable_obj) -> str:
    try:
        callable_obj()
    except RepairError as exc:
        return exc.code
    return "NO_ERROR"


def run_acceptance() -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    metrics: Dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="audiorepair-acceptance-") as temporary:
        root = Path(temporary)
        generated = generate(str(root / "samples"))
        for name in ("dialogue_48k_mono", "dialogue_44k_stereo"):
            output = root / (name + ".repaired.wav")
            report_path = root / (name + ".report.json")
            run_job(
                generated[name]["wav"],
                generated[name]["timeline"],
                str(output),
                str(report_path),
            )
            audio = wavio.read_wav(str(output))
            source = wavio.read_wav(generated[name]["wav"])
            report = json.loads(report_path.read_text(encoding="utf-8"))
            measured_lufs = loudness.integrated_loudness(
                audio.samples, audio.sample_rate
            )
            peak = loudness.peak_dbfs(audio.samples)
            dc = loudness.dc_offset(audio.samples)
            boundaries = [
                boundary
                for segment in report["segments"]
                for boundary in segment["boundaries"]
            ]
            max_error = max(abs(item["error_ms"]) for item in boundaries)
            metrics[name] = {
                "sample_rate": audio.sample_rate,
                "channels": audio.channels,
                "integrated_loudness_lufs": round(measured_lufs, 6),
                "peak_dbfs": round(peak, 6),
                "dc_offset": round(dc, 9),
                "max_boundary_error_ms": round(max_error, 6),
                "output_frames": audio.frames,
                "output_sha256": report["output"]["sha256"],
            }
            checks.append(
                _check(
                    audio.sample_rate == source.sample_rate
                    and audio.channels == source.channels,
                    name + ": output format preserved",
                )
            )
            checks.append(
                _check(abs(measured_lufs + 23.0) <= 0.5,
                       name + ": integrated loudness")
            )
            checks.append(_check(peak <= -1.0, name + ": peak ceiling"))
            checks.append(_check(abs(dc) < 0.001, name + ": DC offset"))
            checks.append(
                _check(max_error <= 20.0, name + ": boundary error")
            )
            checks.append(
                _check(
                    abs(
                        report["verification"]["integrated_loudness_lufs"]
                        - measured_lufs
                    )
                    <= 1e-5,
                    name + ": report remeasurement",
                )
            )
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg:
                probe = subprocess.run(
                    [
                        ffmpeg,
                        "-nostats",
                        "-hide_banner",
                        "-i",
                        str(output),
                        "-filter_complex",
                        "ebur128=peak=true",
                        "-f",
                        "null",
                        "-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=30,
                    check=False,
                )
                match = re.search(
                    r"Integrated loudness:\s*I:\s+"
                    r"(-?\d+(?:\.\d+)?)\s+LUFS",
                    probe.stderr,
                )
                ffmpeg_lufs = float(match.group(1)) if match else None
                metrics[name]["ffmpeg_ebur128_lufs"] = ffmpeg_lufs
                checks.append(
                    _check(
                        ffmpeg_lufs is not None
                        and abs(ffmpeg_lufs - measured_lufs) <= 0.1,
                        name + ": FFmpeg ebur128 cross-check",
                    )
                )

        one_dir = root / "jobs-one"
        four_dir = root / "jobs-four"
        one_jobs = load_batch(generated["batch"], str(one_dir))
        four_jobs = load_batch(generated["batch"], str(four_dir))
        before_threads = {
            thread.ident for thread in threading.enumerate()
            if thread.name.startswith("audiorepair")
        }
        run_batch(one_jobs, 1, str(one_dir / "summary.json"))
        run_batch(four_jobs, 4, str(four_dir / "summary.json"))
        after_threads = {
            thread.ident for thread in threading.enumerate()
            if thread.name.startswith("audiorepair")
        }
        deterministic = True
        for job_id in ("mono48", "stereo44"):
            deterministic &= (
                (one_dir / (job_id + ".wav")).read_bytes()
                == (four_dir / (job_id + ".wav")).read_bytes()
            )
            deterministic &= (
                (one_dir / (job_id + ".report.json")).read_bytes()
                == (four_dir / (job_id + ".report.json")).read_bytes()
            )
        checks.append(
            _check(deterministic, "batch: jobs=1 and jobs=4 byte identical")
        )
        checks.append(
            _check(
                not (after_threads - before_threads),
                "batch: thread pool fully reclaimed",
            )
        )

        source = generated["dialogue_48k_mono"]["wav"]
        timeline_source = generated["dialogue_48k_mono"]["timeline"]
        timeline = json.loads(Path(timeline_source).read_text(encoding="utf-8"))
        invalid = root / "invalid.json"
        invalid.write_text("{invalid", encoding="utf-8")
        out_of_range = json.loads(json.dumps(timeline))
        out_of_range["segments"][0]["start_sec"] = -0.25
        out_of_range_path = root / "out-of-range.json"
        out_of_range_path.write_text(json.dumps(out_of_range), encoding="utf-8")
        unreachable = json.loads(json.dumps(timeline))
        unreachable["segments"][0]["stretch_ratio"] = 1.75
        unreachable_path = root / "unreachable.json"
        unreachable_path.write_text(json.dumps(unreachable), encoding="utf-8")
        output = root / "error-output.wav"
        report = root / "error-report.json"
        codes = {
            "missing": _error_code(
                lambda: run_job(
                    str(root / "missing.wav"),
                    timeline_source,
                    str(output),
                    str(report),
                )
            ),
            "invalid_json": _error_code(
                lambda: run_job(
                    source, str(invalid), str(output), str(report)
                )
            ),
            "out_of_range": _error_code(
                lambda: run_job(
                    source, str(out_of_range_path), str(output), str(report)
                )
            ),
            "stretch_unreachable": _error_code(
                lambda: run_job(
                    source, str(unreachable_path), str(output), str(report)
                )
            ),
        }
        metrics["error_codes"] = codes
        expected_codes = {
            "missing": INPUT_MISSING,
            "invalid_json": TIMELINE_INVALID,
            "out_of_range": TIMELINE_OUT_OF_RANGE,
            "stretch_unreachable": STRETCH_UNREACHABLE,
        }
        checks.append(
            _check(codes == expected_codes, "errors: business codes")
        )
        checks.append(
            _check(
                not output.exists() and not report.exists(),
                "errors: no partial output",
            )
        )

    return {
        "status": "passed" if all(item["passed"] for item in checks)
        else "failed",
        "check_count": len(checks),
        "passed": sum(1 for item in checks if item["passed"]),
        "checks": checks,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="optional path for the JSON result")
    args = parser.parse_args()
    result = run_acceptance()
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

