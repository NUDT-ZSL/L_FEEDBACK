#!/usr/bin/env python3
"""Run reproducible acceptance measurements and write an actual-results report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import scipy
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bgm_mixer import __version__
from bgm_mixer.dsp import (
    fade_envelope,
    loop_period_ratio,
    resample_segment,
    resampled_loop_frames,
    scheduled_period_start,
)
from bgm_mixer.synthesis import synthesize_signal

PYTHON = sys.executable


def write_synth(
    path: Path,
    sample_rate: int,
    duration: float,
    frequencies: Any,
    amplitude: float,
    channels: int = 1,
) -> Path:
    samples = synthesize_signal(
        sample_rate,
        duration,
        frequencies,
        amplitudes=amplitude,
        channels=channels,
    )
    sf.write(path, samples, sample_rate, format="WAV", subtype="PCM_24")
    return path


def write_long_synth(
    path: Path,
    sample_rate: int,
    duration: float,
    frequency: float,
    amplitude: float,
    chunk_seconds: float = 10.0,
) -> Path:
    """Write a long deterministic fixture without a large temporary array."""

    total_frames = round(duration * sample_rate)
    chunk_frames = round(chunk_seconds * sample_rate)
    with sf.SoundFile(
        path,
        mode="w",
        samplerate=sample_rate,
        channels=1,
        format="WAV",
        subtype="PCM_24",
    ) as handle:
        for start in range(0, total_frames, chunk_frames):
            count = min(chunk_frames, total_frames - start)
            samples = synthesize_signal(
                sample_rate,
                count / sample_rate,
                frequency,
                amplitudes=amplitude,
                channels=1,
                start_seconds=start / sample_rate,
            )
            handle.write(samples)
    return path


def write_config(path: Path, payload: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def run_cli(*arguments: str, timeout: float = 300.0) -> Tuple[int, Dict[str, Any], str]:
    completed = subprocess.run(
        [PYTHON, str(ROOT / "mixer.py"), *arguments],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(
            f"expected one JSON line, got {len(lines)}; stderr={completed.stderr!r}"
        )
    return completed.returncode, json.loads(lines[0]), completed.stderr


def spectral_amplitude(samples: np.ndarray, sample_rate: int, frequency: float) -> float:
    index = np.arange(samples.size, dtype=np.float64)
    carrier = np.exp(-2j * math.pi * frequency * index / sample_rate)
    return float(2.0 * abs(np.dot(samples, carrier)) / samples.size)


def measure_resampling() -> Dict[str, Any]:
    cases = [
        (44100, 48000, 15000.0, 18900.0),
        (48000, 44100, 19000.0, 15100.0),
        (44100, 48000, 20000.0, 16100.0),
        (48000, 44100, 20000.0, 16100.0),
    ]
    measurements: List[Dict[str, Any]] = []
    worst_rejection = -math.inf
    worst_length_error = 0.0
    for source_rate, target_rate, frequency, alias_frequency in cases:
        source = synthesize_signal(
            source_rate,
            3.0,
            frequency,
            amplitudes=0.5,
            channels=1,
        )
        expected_frames = resampled_loop_frames(
            source.shape[0], source_rate, target_rate
        )
        output = resample_segment(
            source, source_rate, target_rate, expected_frames
        )[:, 0]
        trim = int(0.1 * target_rate)
        output = output[trim:-trim]
        wanted = spectral_amplitude(output, target_rate, frequency)
        alias = spectral_amplitude(output, target_rate, alias_frequency)
        rejection_db = 20.0 * math.log10((alias + 1e-30) / wanted)
        theory_frames = source.shape[0] * target_rate / source_rate
        length_error = abs(expected_frames - theory_frames)
        worst_rejection = max(worst_rejection, rejection_db)
        worst_length_error = max(worst_length_error, length_error)
        measurements.append(
            {
                "source_rate": source_rate,
                "target_rate": target_rate,
                "tone_hz": frequency,
                "alias_probe_hz": alias_frequency,
                "alias_rejection_db": round(rejection_db, 3),
                "passband_gain_db": round(20.0 * math.log10(wanted / 0.5), 3),
                "frame_count_error": round(length_error, 6),
            }
        )
    return {
        "cases": measurements,
        "worst_alias_rejection_db": round(worst_rejection, 3),
        "limit_db": -60.0,
        "max_frame_count_error": round(worst_length_error, 6),
        "passed": worst_rejection < -60.0 and worst_length_error <= 1.0,
    }


def measure_loop_quality(directory: Path) -> Dict[str, Any]:
    source_rate = 44100
    target_rate = 48000
    loop_start = 0.123456
    loop_end = 1.234567
    source = write_synth(
        directory / "loop-quality.wav",
        source_rate,
        2.0,
        20.0,
        amplitude=0.4,
    )
    config = write_config(
        directory / "loop-quality.json",
        {
            "duration_seconds": 4.0,
            "loop": {"start_seconds": loop_start, "end_seconds": loop_end},
            "fade": {"in_ms": 10, "out_ms": 10, "curve": "linear"},
            "tracks": [{"volume": 0.9}],
        },
    )
    output = directory / "loop-quality-out.wav"
    code, summary, stderr = run_cli(
        "--input",
        str(source),
        "--target-rate",
        str(target_rate),
        "--config",
        str(config),
        "--output",
        str(output),
    )
    if code != 0:
        raise RuntimeError(f"loop quality mix failed: {summary} {stderr}")

    audio, rate = sf.read(str(output), dtype="float64", always_2d=True)
    track = summary["tracks"][0]
    source_frames = track["loop_end_sample"] - track["loop_start_sample"]
    numerator, denominator = loop_period_ratio(
        source_frames, source_rate, target_rate
    )
    starts: List[int] = []
    index = 0
    while True:
        start = scheduled_period_start(index, numerator, denominator)
        if start >= audio.shape[0]:
            break
        starts.append(start)
        index += 1

    seam_max = 0.0
    for start in starts[1:]:
        seam_max = max(seam_max, abs(audio[start, 0] - audio[start - 1, 0]))

    fade_in = track["fade_in_frames"]
    fade_out = track["fade_out_frames"]
    fade_output_step = 0.0
    for start in starts:
        first = audio[start : start + fade_in, 0]
        last_start = start + track["loop_output_frames"] - fade_out
        last = audio[last_start : last_start + fade_out, 0]
        if first.size > 1:
            fade_output_step = max(fade_output_step, float(np.max(np.abs(np.diff(first)))))
        if last.size > 1:
            fade_output_step = max(fade_output_step, float(np.max(np.abs(np.diff(last)))))

    envelope_step = 0.0
    for curve in ("linear", "equal_power"):
        envelope = fade_envelope(target_rate, 480, 480, curve)
        envelope_step = max(envelope_step, float(np.max(np.abs(np.diff(envelope)))))

    configured_duration = loop_end - loop_start
    duration_error_ms = abs(track["loop_duration_seconds"] - configured_duration) * 1000.0
    return {
        "source_loop_samples": [track["loop_start_sample"], track["loop_end_sample"]],
        "resampled_loop_frames": track["loop_output_frames"],
        "fractional_period_frames": round(numerator / denominator, 9),
        "loop_count": summary["loop_count"],
        "loop_duration_error_ms": round(duration_error_ms, 6),
        "seam_max_abs_sample_delta": round(seam_max, 9),
        "fade_envelope_max_step_10ms_48k": round(envelope_step, 9),
        "fade_region_output_max_step_20hz_tone": round(fade_output_step, 9),
        "passed": (
            duration_error_ms <= 1.0
            and seam_max <= 0.02
            and envelope_step <= 0.005
            and fade_output_step <= 0.005
        ),
    }


def measure_multi_track_and_determinism(directory: Path) -> Dict[str, Any]:
    left = write_synth(
        directory / "mix-left.wav", 44100, 2.0, 1000.0, amplitude=0.8
    )
    right = write_synth(
        directory / "mix-right.wav", 32000, 2.0, 1000.0, amplitude=0.8
    )
    config = write_config(
        directory / "mix.json",
        {
            "duration_seconds": 2.5,
            "fade_ms": 0,
            "tracks": [
                {"path": str(left), "volume": 1.0, "offset_seconds": 0.0},
                {"path": str(right), "volume": 1.0, "offset_seconds": 0.25},
            ],
        },
    )
    output = directory / "mix-out.wav"
    arguments = (
        "--input",
        str(left),
        "--target-rate",
        "48000",
        "--config",
        str(config),
        "--output",
        str(output),
    )
    code1, summary1, stderr1 = run_cli(*arguments)
    if code1 != 0:
        raise RuntimeError(f"first mix failed: {summary1} {stderr1}")
    bytes1 = output.read_bytes()
    hash1 = hashlib.sha256(bytes1).hexdigest()
    code2, summary2, stderr2 = run_cli(*arguments)
    if code2 != 0:
        raise RuntimeError(f"second mix failed: {summary2} {stderr2}")
    bytes2 = output.read_bytes()
    hash2 = hashlib.sha256(bytes2).hexdigest()
    audio, _ = sf.read(str(output), dtype="float64", always_2d=True)
    return {
        "tracks": 2,
        "target_rate": 48000,
        "output_frames": summary1["output_frames"],
        "clip_count": summary1["clip_count"],
        "clipped_frames": summary1["clipped_frames"],
        "second_track_offset_frames": summary1["tracks"][1]["offset_frames"],
        "output_peak": round(float(np.max(np.abs(audio))), 9),
        "first_sha256": hash1,
        "second_sha256": hash2,
        "byte_identical": bytes1 == bytes2,
        "summary_identical": summary1 == summary2,
        "passed": (
            summary1["clip_count"] > 0
            and float(np.max(np.abs(audio))) <= 1.0 + 1e-7
            and bytes1 == bytes2
            and summary1 == summary2
        ),
    }


def measure_error_paths(directory: Path) -> Dict[str, Any]:
    good = write_synth(
        directory / "error-good.wav", 44100, 1.0, 100.0, amplitude=0.4
    )
    low_rate = write_synth(
        directory / "error-low-rate.wav", 4000, 0.5, 100.0, amplitude=0.4
    )
    good_config = write_config(
        directory / "error-good.json",
        {"duration_seconds": 1.0, "tracks": [{"volume": 1.0}]},
    )
    bad_volume = write_config(
        directory / "error-volume.json",
        {"duration_seconds": 1.0, "tracks": [{"volume": 1.25}]},
    )
    bad_offset = write_config(
        directory / "error-offset.json",
        {
            "duration_seconds": 1.0,
            "tracks": [{"volume": 1.0, "offset_seconds": -0.1}],
        },
    )
    cases = [
        ("source_rate_below_8000", str(low_rate), str(good_config), 48000, "E_INVALID_INPUT"),
        ("volume_above_one", str(good), str(bad_volume), 48000, "E_INVALID_INPUT"),
        ("negative_offset", str(good), str(bad_offset), 48000, "E_INVALID_INPUT"),
        (
            "missing_output_directory",
            str(good),
            str(good_config),
            48000,
            "E_IO",
        ),
    ]
    results: List[Dict[str, Any]] = []
    all_passed = True
    for index, (name, source, config, rate, expected) in enumerate(cases):
        destination = (
            directory / "missing" / f"error-{index}.wav"
            if name == "missing_output_directory"
            else directory / f"error-{index}.wav"
        )
        code, summary, _ = run_cli(
            "--input",
            source,
            "--target-rate",
            str(rate),
            "--config",
            config,
            "--output",
            str(destination),
        )
        leftovers = list(destination.parent.glob(f".{destination.name}.*.tmp"))
        passed = (
            code != 0
            and summary.get("error_code") == expected
            and not destination.exists()
            and not leftovers
        )
        all_passed = all_passed and passed
        results.append(
            {
                "case": name,
                "expected_error_code": expected,
                "actual_error_code": summary.get("error_code"),
                "exit_code": code,
                "no_partial_output": not destination.exists() and not leftovers,
                "passed": passed,
            }
        )
    return {"cases": results, "all_passed": all_passed}


def measure_long_playback(directory: Path) -> Dict[str, Any]:
    source_rate = 44100
    target_rate = 48000
    duration_seconds = 31 * 60
    loop_start = 0.123456
    loop_end = 1.2346
    source = write_synth(
        directory / "long-loop.wav",
        source_rate,
        2.0,
        10.0,
        amplitude=0.4,
    )
    config = write_config(
        directory / "long.json",
        {
            "duration_seconds": duration_seconds,
            "loop": {"start_seconds": loop_start, "end_seconds": loop_end},
            "fade": {"in_ms": 50, "out_ms": 50, "curve": "linear"},
            "tracks": [{"volume": 1.0}],
        },
    )
    output = directory / "long-out.wav"
    started = time.perf_counter()
    code, summary, stderr = run_cli(
        "--input",
        str(source),
        "--target-rate",
        str(target_rate),
        "--config",
        str(config),
        "--output",
        str(output),
        timeout=180.0,
    )
    wall_seconds = time.perf_counter() - started
    if code != 0:
        raise RuntimeError(f"long playback failed: {summary} {stderr}")

    track = summary["tracks"][0]
    source_frames = track["loop_end_sample"] - track["loop_start_sample"]
    numerator, denominator = loop_period_ratio(
        source_frames, source_rate, target_rate
    )
    period = numerator / denominator
    minute_checks: List[Dict[str, Any]] = []
    max_phase_error = 0.0
    seam_max = 0.0
    with sf.SoundFile(str(output), mode="r") as handle:
        for minute in (10, 20, 30):
            timeline = minute * 60 * target_rate
            loop_index = int(timeline / period)
            while scheduled_period_start(loop_index, numerator, denominator) > timeline:
                loop_index -= 1
            while (
                scheduled_period_start(loop_index + 1, numerator, denominator)
                <= timeline
            ):
                loop_index += 1
            actual_start = scheduled_period_start(
                loop_index, numerator, denominator
            )
            ideal_local = timeline - loop_index * period
            actual_local = timeline - actual_start
            phase_error = abs(actual_local - ideal_local)
            max_phase_error = max(max_phase_error, phase_error)

            center = max(1, actual_start)
            handle.seek(center - 1)
            samples = handle.read(3, dtype="float64", always_2d=True)[:, 0]
            seam = abs(samples[1] - samples[0]) if samples.size >= 2 else 0.0
            seam_max = max(seam_max, seam)
            minute_checks.append(
                {
                    "minute": minute,
                    "loop_index": loop_index,
                    "loop_start_frame": actual_start,
                    "ideal_local_position": round(ideal_local, 6),
                    "actual_local_position": int(actual_local),
                    "phase_error_samples": round(phase_error, 6),
                    "seam_abs_sample_delta": round(seam, 9),
                }
            )

    return {
        "duration_seconds": duration_seconds,
        "target_rate": target_rate,
        "output_frames": summary["output_frames"],
        "loop_count": summary["loop_count"],
        "fractional_period_frames": round(period, 9),
        "wall_seconds": round(wall_seconds, 3),
        "output_size_bytes": output.stat().st_size,
        "ten_minute_checks": minute_checks,
        "max_phase_error_samples": round(max_phase_error, 6),
        "seam_max_abs_sample_delta": round(seam_max, 9),
        "passed": bool(
            summary["output_frames"] == round(duration_seconds * target_rate)
            and max_phase_error <= 1.0
            and seam_max <= 0.01
        ),
    }


def measure_ten_minute_performance(directory: Path) -> Dict[str, Any]:
    target_rate = 48000
    duration_seconds = 600.0
    source_a = write_long_synth(
        directory / "perf-a.wav",
        44100,
        duration_seconds,
        220.0,
        amplitude=0.25,
    )
    source_b = write_long_synth(
        directory / "perf-b.wav",
        48000,
        duration_seconds,
        330.0,
        amplitude=0.25,
    )
    config = write_config(
        directory / "perf.json",
        {
            "duration_seconds": duration_seconds,
            "fade": {"in_ms": 10, "out_ms": 10, "curve": "equal_power"},
            "tracks": [
                {"path": str(source_a), "volume": 0.7, "offset_seconds": 0.0},
                {"path": str(source_b), "volume": 0.6, "offset_seconds": 0.25},
            ],
        },
    )
    output = directory / "perf-out.wav"
    started = time.perf_counter()
    code, summary, stderr = run_cli(
        "--input",
        str(source_a),
        "--target-rate",
        str(target_rate),
        "--config",
        str(config),
        "--output",
        str(output),
        timeout=120.0,
    )
    wall_seconds = time.perf_counter() - started
    if code != 0:
        raise RuntimeError(f"performance mix failed: {summary} {stderr}")
    info = sf.info(str(output))
    return {
        "duration_seconds": duration_seconds,
        "tracks": 2,
        "channels": info.channels,
        "target_rate": target_rate,
        "output_frames": info.frames,
        "wall_seconds": round(wall_seconds, 3),
        "realtime_factor": round(duration_seconds / wall_seconds, 3),
        "output_size_bytes": output.stat().st_size,
        "limit_seconds": 60.0,
        "passed": wall_seconds < 60.0 and info.frames == round(duration_seconds * target_rate),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default=str(ROOT / "artifacts" / "acceptance_report.json"),
        help="path for the JSON report",
    )
    parser.add_argument(
        "--skip-long",
        action="store_true",
        help="skip the 31-minute and 10-minute playback measurements",
    )
    args = parser.parse_args(argv)

    report: Dict[str, Any] = {
        "tool": "bgm_mixer",
        "tool_version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "soundfile": sf.__version__,
            "cpu_count": os.cpu_count(),
        },
    }
    with tempfile.TemporaryDirectory(prefix="bgm-mixer-acceptance-") as temp_name:
        directory = Path(temp_name)
        report["resampling"] = measure_resampling()
        report["loop_quality"] = measure_loop_quality(directory)
        report["multi_track_and_determinism"] = measure_multi_track_and_determinism(
            directory
        )
        report["error_paths"] = measure_error_paths(directory)
        if not args.skip_long:
            report["long_playback"] = measure_long_playback(directory)
            report["ten_minute_performance"] = measure_ten_minute_performance(
                directory
            )

    checks = [
        report["resampling"]["passed"],
        report["loop_quality"]["passed"],
        report["multi_track_and_determinism"]["passed"],
        report["error_paths"]["all_passed"],
    ]
    if "long_playback" in report:
        checks.append(report["long_playback"]["passed"])
        checks.append(report["ten_minute_performance"]["passed"])
    report["all_passed"] = all(checks)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "ok" if report["all_passed"] else "failed",
                "report": str(report_path.resolve()),
                "all_passed": report["all_passed"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
