#!/usr/bin/env python3
"""End-to-end acceptance checks for the Windows automix command-line tool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from generate_samples import generate, measure_bs1770, read_wav


class CheckFailure(RuntimeError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def run_tool(executable: Path, config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(executable), "--config", str(config)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def limiter(threshold: float = -3.0, lookahead: float = 10.0) -> dict[str, Any]:
    return {
        "enabled": True,
        "threshold_dbfs": threshold,
        "lookahead_ms": lookahead,
        "release_ms": 80.0,
    }


def drc(threshold: float, ratio: float, attack: float, release: float,
        knee: float) -> dict[str, Any]:
    return {
        "enabled": True,
        "threshold_db": threshold,
        "ratio": ratio,
        "attack_ms": attack,
        "release_ms": release,
        "knee_db": knee,
    }


def assert_error(process: subprocess.CompletedProcess[str], code: int,
                 error_name: str, wav: Path, report: Path,
                 log_dir: Path | None = None) -> None:
    check(process.returncode == code,
          f"expected exit {code}, got {process.returncode}; stderr={process.stderr}")
    check(error_name in process.stderr,
          f"expected {error_name} on stderr; stderr={process.stderr}")
    check(not wav.exists(), f"partial WAV was produced: {wav}")
    check(not report.exists(), f"report was produced for a failed run: {report}")
    if log_dir is not None:
        check(log_dir.exists(), f"expected retained logs in {log_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    executable = args.exe.resolve()
    work = args.work_dir.resolve()
    check(executable.exists(), f"executable not found: {executable}")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    samples = work / "samples"
    reference = generate(samples)
    metrics: dict[str, Any] = {"reference": reference}

    full_config = {
        "sample_rate": 48000,
        "output": "out/cabin_mix.wav",
        "report": "out/cabin_mix_report.json",
        "log_dir": "logs/full",
        "master_gain_db": 12.0,
        "drc": drc(-8.0, 2.0, 20.0, 200.0, 6.0),
        "limiter": limiter(-3.0, 10.0),
        "tracks": [
            {
                "name": "navigation",
                "input": "samples/nav_44k1_16bit_mono.wav",
                "gain_db": -2.0,
                "drc": drc(-18.0, 3.5, 8.0, 120.0, 4.0),
            },
            {
                "name": "media",
                "input": "samples/media_48k_24bit_stereo.wav",
                "gain_db": -3.0,
                "drc": drc(-20.0, 2.5, 15.0, 150.0, 6.0),
            },
            {
                "name": "phone",
                "input": "samples/phone_48k_float32_mono.wav",
                "gain_db": 0.0,
                "drc": drc(-22.0, 3.0, 10.0, 180.0, 5.0),
            },
            {
                "name": "system",
                "input": "samples/system_44k1_16bit_stereo.wav",
                "gain_db": -4.0,
                "drc": drc(-20.0, 4.0, 6.0, 100.0, 3.0),
            },
        ],
    }
    full_config_path = work / "full.json"
    json_write(full_config_path, full_config)
    full_process = run_tool(executable, full_config_path)
    check(full_process.returncode == 0,
          f"full render failed: {full_process.stderr}")
    full_wav = work / "out/cabin_mix.wav"
    full_report_path = work / "out/cabin_mix_report.json"
    check(full_wav.exists(), "full render did not create a WAV")
    check(full_report_path.exists(), "full render did not create a report")

    rate, channels, samples_out = read_wav(full_wav)
    check(rate == 48000, f"output rate is {rate}, expected 48000")
    check(channels == 2, f"output channels are {channels}, expected 2")
    check(abs(len(samples_out) / channels / rate - 5.0) < 1e-6,
          "output duration should equal the longest 5-second track")
    peak = max(abs(value) for value in samples_out)
    threshold_linear = 10 ** (-3.0 / 20.0)
    check(peak <= threshold_linear + 1e-7,
          f"limiter overshoot: peak={peak}, threshold={threshold_linear}")

    report = json.loads(full_report_path.read_text(encoding="utf-8"))
    check(report["limiter"]["lookahead_samples"] == 480,
          "lookahead_samples does not match 10 ms at 48 kHz")
    check(report["limiter"]["triggered_samples"] > 0,
          "full acceptance mix did not trigger the limiter")
    check(report["limiter"]["max_attenuation_db"] > 0.0,
          "limiter attenuation report is not positive")
    check(len(report["tracks"]) == 4, "report does not contain all four tracks")
    for track in report["tracks"]:
        check(track["source_sample_rate"] in (44100, 48000),
              "unexpected source sample rate in report")
        check(track["output_frames"] > 0, "empty output track")
        check("max_reduction_db" in track["drc"],
              "track DRC gain statistics are missing")
        check("read" in track["timing_ms"] and "resample" in track["timing_ms"],
              "track timing statistics are missing")

    independent = measure_bs1770(rate, channels, samples_out)
    check(report["loudness"]["integrated_lufs"] is not None and
          independent["integrated_lufs"] is not None,
          "integrated loudness was not produced")
    lufs_error = abs(report["loudness"]["integrated_lufs"] -
                     independent["integrated_lufs"])
    tp_error = abs(report["loudness"]["true_peak_dbtp"] -
                   independent["true_peak_dbtp"])
    check(lufs_error <= 0.5, f"integrated loudness mismatch: {lufs_error} LU")
    check(tp_error <= 0.5, f"true peak mismatch: {tp_error} dB")
    metrics["full_mix"] = {
        "integrated_lufs": report["loudness"]["integrated_lufs"],
        "true_peak_dbtp": report["loudness"]["true_peak_dbtp"],
        "sample_peak_dbfs": 20.0 * math.log10(peak),
        "limiter_triggered_samples": report["limiter"]["triggered_samples"],
        "limiter_max_attenuation_db": report["limiter"]["max_attenuation_db"],
        "loudness_error_lu": lufs_error,
        "true_peak_error_db": tp_error,
    }

    first_digest = hashlib.sha256(full_wav.read_bytes()).hexdigest()
    repeat_process = run_tool(executable, full_config_path)
    check(repeat_process.returncode == 0,
          f"repeat render failed: {repeat_process.stderr}")
    second_digest = hashlib.sha256(full_wav.read_bytes()).hexdigest()
    check(first_digest == second_digest,
          "repeated runs did not produce byte-identical WAV output")
    metrics["deterministic_sha256"] = first_digest

    log_path = work / "logs/full/automix.log"
    check(log_path.exists(), "full render did not produce a log")
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    check("track navigation" in log_text and "resample_ms=" in log_text,
          "log is missing per-track summaries or timings")
    check("output committed" in log_text, "log is missing commit summary")

    reference_config = {
        "output": "out/reference_render.wav",
        "report": "out/reference_render.json",
        "log_dir": "logs/reference",
        "limiter": limiter(-1.0, 10.0),
        "tracks": [{
            "name": "reference",
            "input": "samples/reference_48k_24bit_stereo.wav",
            "gain_db": 0.0,
        }],
    }
    reference_config_path = work / "reference.json"
    json_write(reference_config_path, reference_config)
    reference_process = run_tool(executable, reference_config_path)
    check(reference_process.returncode == 0,
          f"reference render failed: {reference_process.stderr}")
    reference_report = json.loads(
        (work / "out/reference_render.json").read_text(encoding="utf-8")
    )
    reference_lufs_error = abs(
        reference_report["loudness"]["integrated_lufs"] -
        reference["integrated_lufs"]
    )
    reference_tp_error = abs(
        reference_report["loudness"]["true_peak_dbtp"] -
        reference["true_peak_dbtp"]
    )
    check(reference_lufs_error <= 0.5,
          f"reference LUFS error exceeds 0.5: {reference_lufs_error}")
    check(reference_tp_error <= 0.5,
          f"reference true-peak error exceeds 0.5: {reference_tp_error}")
    metrics["reference_error"] = {
        "lufs_lu": reference_lufs_error,
        "true_peak_db": reference_tp_error,
    }

    mono_config = {
        "output": "out/equal_power.wav",
        "report": "out/equal_power.json",
        "log_dir": "logs/equal_power",
        "limiter": {"enabled": False, "threshold_dbfs": -1, "lookahead_ms": 10},
        "tracks": [{
            "name": "mono",
            "input": "samples/nav_44k1_16bit_mono.wav",
            "gain_db": 0.0,
        }],
    }
    mono_config_path = work / "equal_power.json"
    json_write(mono_config_path, mono_config)
    mono_process = run_tool(executable, mono_config_path)
    check(mono_process.returncode == 0,
          f"mono equal-power render failed: {mono_process.stderr}")
    mono_rate, mono_channels, mono_samples = read_wav(work / "out/equal_power.wav")
    check(mono_rate == 48000 and mono_channels == 2,
          "mono output is not 48 kHz stereo")
    check(all(mono_samples[index] == mono_samples[index + 1]
              for index in range(0, len(mono_samples), 2)),
          "mono track was not distributed identically to left and right")
    metrics["equal_power_peak_dbfs"] = 20.0 * math.log10(
        max(abs(value) for value in mono_samples)
    )

    syntax_wav = work / "out/syntax.wav"
    syntax_report = work / "out/syntax_report.json"
    syntax_path = work / "bad_syntax.json"
    syntax_path.write_text(
        '{"output":"out/syntax.wav","report":"out/syntax_report.json",'
        '"tracks":[}', encoding="utf-8"
    )
    syntax_process = run_tool(executable, syntax_path)
    assert_error(syntax_process, 2, "E_CFG_SYNTAX", syntax_wav, syntax_report,
                 work / "logs")

    missing_output_path = work / "missing_field.json"
    missing_output_path.write_text(
        json.dumps({"tracks": [{
            "name": "x", "input": "samples/nav_44k1_16bit_mono.wav",
            "gain_db": 0
        }]}),
        encoding="utf-8",
    )
    missing_process = run_tool(executable, missing_output_path)
    check(missing_process.returncode == 2,
          f"missing field exit code was {missing_process.returncode}")
    check("E_CFG_MISSING" in missing_process.stderr,
          "missing field did not report E_CFG_MISSING")

    missing_file_wav = work / "out/missing_file.wav"
    missing_file_report = work / "out/missing_file.json"
    missing_file_config = work / "missing_file.json"
    json_write(missing_file_config, {
        "output": "out/missing_file.wav",
        "report": "out/missing_file.json",
        "log_dir": "logs/missing_file",
        "tracks": [{
            "name": "missing", "input": "samples/not_there.wav", "gain_db": 0
        }],
    })
    missing_file_process = run_tool(executable, missing_file_config)
    assert_error(missing_file_process, 3, "E_FILE_MISSING", missing_file_wav,
                 missing_file_report, work / "logs/missing_file")

    unsupported_wav = work / "out/unsupported.wav"
    unsupported_report = work / "out/unsupported.json"
    unsupported_config = work / "unsupported.json"
    json_write(unsupported_config, {
        "output": "out/unsupported.wav",
        "report": "out/unsupported.json",
        "log_dir": "logs/unsupported",
        "tracks": [{
            "name": "unsupported", "input": "samples/unsupported_8bit.wav",
            "gain_db": 0
        }],
    })
    unsupported_process = run_tool(executable, unsupported_config)
    assert_error(unsupported_process, 3, "E_FILE_FORMAT", unsupported_wav,
                 unsupported_report, work / "logs/unsupported")

    corrupt_wav = work / "out/corrupt.wav"
    corrupt_report = work / "out/corrupt.json"
    corrupt_config = work / "corrupt.json"
    json_write(corrupt_config, {
        "output": "out/corrupt.wav",
        "report": "out/corrupt.json",
        "log_dir": "logs/corrupt",
        "tracks": [{
            "name": "corrupt", "input": "samples/corrupt_truncated.wav",
            "gain_db": 0
        }],
    })
    corrupt_process = run_tool(executable, corrupt_config)
    assert_error(corrupt_process, 3, "E_FILE_CORRUPT", corrupt_wav,
                 corrupt_report, work / "logs/corrupt")

    not_a_directory = work / "not_a_directory"
    not_a_directory.write_text("not a directory", encoding="utf-8")
    fallback_config = {
        "output": "out/log_fallback.wav",
        "report": "out/log_fallback.json",
        "log_dir": "not_a_directory",
        "limiter": limiter(-1.0, 10.0),
        "tracks": [{
            "name": "reference",
            "input": "samples/reference_48k_24bit_stereo.wav",
            "gain_db": 0.0,
        }],
    }
    fallback_path = work / "log_fallback.json"
    json_write(fallback_path, fallback_config)
    fallback_process = run_tool(executable, fallback_path)
    check(fallback_process.returncode == 0,
          f"log fallback affected rendering: {fallback_process.stderr}")
    check("warning: logging disabled" in fallback_process.stderr,
          "unwritable log path did not emit a stderr warning")
    check((work / "out/log_fallback.wav").exists(),
          "log fallback did not produce output WAV")
    check((work / "out/log_fallback.json").exists(),
          "log fallback did not produce report")

    metrics["errors"] = {
        "E_CFG_SYNTAX": 2,
        "E_CFG_MISSING": 2,
        "E_FILE_MISSING": 3,
        "E_FILE_FORMAT": 3,
        "E_FILE_CORRUPT": 3,
    }
    metrics["passed"] = True
    summary_path = work / "acceptance_summary.json"
    json_write(summary_path, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"PASS acceptance summary: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckFailure as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
