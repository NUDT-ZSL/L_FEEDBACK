#!/usr/bin/env python3
"""Run the bundled deterministic acceptance scenario and external checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cinema_master.samplegen import generate_sample  # noqa: E402


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def run_cli(input_dir: str, output_dir: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "cinema_master", input_dir, output_dir],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env=env,
    )


def external_measurements(wav_path: str) -> Dict[str, float]:
    process = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            wav_path,
            "-filter_complex",
            "ebur128=peak=true",
            "-f",
            "null",
            "-",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError("FFmpeg ebur128 verification failed: " + process.stderr[-2000:])
    integrated = re.findall(r"I:\s+(-?[0-9.]+)\s+LUFS", process.stderr)
    peaks = re.findall(r"Peak:\s+(-?[0-9.]+)\s+dBFS", process.stderr)
    if not integrated or not peaks:
        raise RuntimeError("Unable to parse FFmpeg ebur128 output")
    return {
        "integrated_lufs": float(integrated[-1]),
        "true_peak_dbfs": float(peaks[-1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.join(ROOT, "test_tmp", "acceptance"))
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    protected = {os.path.abspath(os.sep), os.path.abspath(os.path.expanduser("~")), ROOT}
    if root in protected or ROOT.startswith(root + os.sep):
        raise SystemExit("Refusing to clean an unsafe acceptance root: " + root)
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root)
    input_dir = os.path.join(root, "input")
    output_dir = os.path.join(root, "output")
    generate_sample(input_dir)

    first = run_cli(input_dir, output_dir)
    if first.returncode != 0:
        sys.stderr.write(first.stderr)
        return 1
    cli_summary = json.loads(first.stdout)
    report_path = os.path.join(output_dir, "report.json")
    report = read_json(report_path)
    wav_path = report["output"]["wav"]["path"]
    mp4_path = report["output"]["container"]["path"]
    first_wav_hash = sha256(wav_path)
    first_report_hash = sha256(report_path)

    second = run_cli(input_dir, output_dir)
    if second.returncode != 0:
        sys.stderr.write(second.stderr)
        return 1
    second_report = read_json(report_path)
    second_wav_hash = sha256(wav_path)
    second_report_hash = sha256(report_path)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name,channels,channel_layout,sample_rate,duration:"
            "format=duration:format_tags",
            "-of",
            "json",
            mp4_path,
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    probe_payload = json.loads(probe.stdout)
    stream = probe_payload["streams"][0]
    external = external_measurements(wav_path)
    duration_error = abs(float(stream["duration"]) - float(report["output"]["wav"]["duration_sec"]))
    loudness_error = abs(
        float(report["loudness"]["program_lufs"])
        - float(report["loudness"]["target_lufs"])
    )

    error_input = os.path.join(root, "error_input")
    error_output = os.path.join(root, "error_output")
    os.makedirs(error_input)
    with open(os.path.join(error_input, "config.json"), "w", encoding="utf-8") as handle:
        handle.write("{invalid json")
    error_process = run_cli(error_input, error_output)
    error_report = read_json(os.path.join(error_output, "report.json"))

    limiter_input = os.path.join(root, "limiter_input")
    limiter_output = os.path.join(root, "limiter_output")
    shutil.copytree(input_dir, limiter_input)
    limiter_config_path = os.path.join(limiter_input, "config.json")
    limiter_config = read_json(limiter_config_path)
    limiter_config["target_lufs"] = -16.0
    limiter_config["output"]["basename"] = "limiter_master"
    with open(limiter_config_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(limiter_config, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    limiter_process = run_cli(limiter_input, limiter_output)
    limiter_report = read_json(os.path.join(limiter_output, "report.json"))
    limiter_peak = float(limiter_report["true_peak"]["measured_dbtp"])
    limiter_lufs = float(limiter_report["loudness"]["program_lufs"])

    checks = {
        "cli_exit_zero": first.returncode == 0,
        "program_loudness_error_le_0_5_lu": loudness_error <= 0.5,
        "external_loudness_error_le_0_5_lu": abs(
            external["integrated_lufs"] - float(report["loudness"]["target_lufs"])
        )
        <= 0.5,
        "true_peak_le_limit": float(report["true_peak"]["measured_dbtp"])
        <= float(report["true_peak"]["limit_dbtp"]),
        "external_true_peak_le_limit": external["true_peak_dbfs"]
        <= float(report["true_peak"]["limit_dbtp"]) + 0.01,
        "packaged_duration_matches": duration_error <= 0.001,
        "packaged_layout_is_51": stream.get("channel_layout") == "5.1",
        "loudness_metadata_present": "loudness_lufs="
        in str(probe_payload.get("format", {}).get("tags", {}).get("comment", "")),
        "wav_hash_stable": first_wav_hash == second_wav_hash,
        "report_hash_stable": first_report_hash == second_report_hash,
        "error_exit_nonzero": error_process.returncode != 0,
        "error_code_reported": error_report.get("error", {}).get("code")
        == "E_CONFIG_INVALID_JSON",
        "temporary_dirs_cleaned": not any(
            name.startswith(".cinema-master-") for name in os.listdir(output_dir)
        ),
        "forced_limiter_exit_zero": limiter_process.returncode == 0,
        "forced_limiter_triggered": bool(
            limiter_report["limiter"]["final"]["triggered"]
        ),
        "forced_limiter_peak_pass": limiter_peak <= -1.0,
        "forced_limiter_loudness_pass": abs(limiter_lufs - (-16.0)) <= 0.5,
    }
    result = {
        "status": "ok" if all(checks.values()) else "failed",
        "checks": checks,
        "report": report_path,
        "wav": wav_path,
        "mp4": mp4_path,
        "program_lufs": report["loudness"]["program_lufs"],
        "external_integrated_lufs": external["integrated_lufs"],
        "true_peak_dbtp": report["true_peak"]["measured_dbtp"],
        "external_true_peak_dbfs": external["true_peak_dbfs"],
        "mp4_duration_sec": float(stream["duration"]),
        "wav_duration_sec": report["output"]["wav"]["duration_sec"],
        "resampling_records": [
            {
                "file": track["file"],
                "sample_rate_before": track["sample_rate_before"],
                "sample_rate_after": track["sample_rate_after"],
            }
            for track in second_report["tracks"]
        ],
        "dialogue_segments": second_report["dialogue_activity"]["segments"],
        "wav_sha256": first_wav_hash,
        "report_sha256": first_report_hash,
        "forced_limiter": {
            "program_lufs": limiter_lufs,
            "target_lufs": -16.0,
            "true_peak_dbtp": limiter_peak,
            "iterations": limiter_report["limiter"]["iteration_count"],
            "max_reduction_db": limiter_report["limiter"]["final"][
                "max_reduction_db"
            ],
        },
        "cli_summary": cli_summary,
    }
    result_path = os.path.join(root, "acceptance_result.json")
    with open(result_path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

