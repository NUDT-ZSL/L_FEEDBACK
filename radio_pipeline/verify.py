from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .audio import loudness_lufs_approx, pcm_square_sum, sha256_file
from .errors import VerificationFailure
from .ffmpeg_io import (
    decode_audio_int16,
    embedded_metadata,
    probe_audio,
    stream_summary,
)
from .stats import build_report, load_processing_log


VERIFY_SCHEMA = "radio-pipeline-verification-v1"


def verify_run(
    run_dir: Path,
    report_path: Path | None = None,
) -> tuple[dict[str, Any], int]:
    run_dir = run_dir.resolve()
    log_path = run_dir / "logs" / "processing_log.json"
    checks: list[dict[str, Any]] = []

    try:
        log = load_processing_log(log_path)
        rebuilt = build_report(log)
    except VerificationFailure as exc:
        return _failure_result(str(exc)), 3

    report_file = report_path.resolve() if report_path else run_dir / "report.json"
    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _failure_result(f"report file not found: {report_file}"), 3
    except json.JSONDecodeError as exc:
        return _failure_result(f"report is not valid JSON: {exc}"), 3
    _check(
        checks,
        "report_matches_log_rebuild",
        report == rebuilt,
        "report.json equals a fresh report rebuilt from processing_log.json",
    )
    _check(
        checks,
        "log_statistics_consistent",
        rebuilt["validation"]["timeline_monotonic"]
        and rebuilt["validation"]["frame_square_sum_matches_output"],
        "frame, packet, PLC, buffer, and loudness counters agree with raw logs",
    )

    config = log.get("config", {}) if isinstance(log.get("config"), dict) else {}
    output_config = config.get("output", {}) if isinstance(config, dict) else {}
    report_output = report.get("output", {})
    expected_samples = int(report["source"]["total_samples"])
    expected_duration_ms = int(report["source"]["duration_ms"])
    expected_metadata = report_output.get("embedded_metadata")
    loudness_tolerance = float(
        output_config.get("loudness_tolerance_lufs", 1.0)
    )
    duration_tolerance = int(
        output_config.get("duration_tolerance_samples", 1)
    )

    wav_path = run_dir / str(report_output.get("wav_file"))
    ogg_path = run_dir / str(report_output.get("ogg_file"))
    pcm_path = run_dir / "processed.pcm"
    _check(checks, "wav_exists", wav_path.is_file(), str(wav_path))
    _check(checks, "ogg_exists", ogg_path.is_file(), str(ogg_path))
    _check(checks, "pcm_exists", pcm_path.is_file(), str(pcm_path))
    if not (wav_path.is_file() and ogg_path.is_file() and pcm_path.is_file()):
        return _finish(checks, {}, 3)

    hash_checks = (
        ("wav_hash", wav_path, report_output.get("wav_sha256")),
        ("ogg_hash", ogg_path, report_output.get("ogg_sha256")),
        ("pcm_hash", pcm_path, report_output.get("pcm_sha256")),
    )
    for name, path, expected_hash in hash_checks:
        _check(
            checks,
            name,
            sha256_file(path) == expected_hash,
            f"SHA-256 of {path.name} matches the processing log",
        )

    metrics: dict[str, Any] = {
        "expected_samples": expected_samples,
        "expected_duration_ms": expected_duration_ms,
        "loudness_tolerance_lufs": loudness_tolerance,
        "duration_tolerance_samples": duration_tolerance,
    }

    try:
        wav_info = sf.info(str(wav_path))
        wav_samples, wav_rate = sf.read(
            str(wav_path), dtype="int16", always_2d=False
        )
    except (RuntimeError, OSError, ValueError) as exc:
        return _failure_result(f"soundfile could not read WAV: {exc}"), 3
    if wav_samples.ndim == 2:
        wav_samples = wav_samples[:, 0]
    wav_samples = np.asarray(wav_samples, dtype="<i2")
    pcm_bytes = pcm_path.read_bytes()
    reference = np.frombuffer(pcm_bytes, dtype="<i2")
    _check(
        checks,
        "wav_is_pcm_s16le",
        wav_info.subtype == "PCM_16",
        f"soundfile subtype={wav_info.subtype}",
    )
    _check(
        checks,
        "wav_soundfile_rate",
        int(wav_rate) == 48000,
        f"soundfile sample rate={wav_rate}",
    )
    _check(
        checks,
        "wav_sample_count",
        abs(int(wav_samples.size) - expected_samples) <= duration_tolerance,
        f"decoded samples={wav_samples.size}, expected={expected_samples}",
    )
    _check(
        checks,
        "wav_matches_processed_pcm",
        wav_samples.size == reference.size and np.array_equal(wav_samples, reference),
        "WAV contains exactly the processed PCM samples",
    )

    wav_probe = probe_audio(wav_path)
    wav_summary = stream_summary(wav_probe)
    metrics["wav"] = wav_summary
    _check(
        checks,
        "wav_codec",
        wav_summary["codec_name"] == "pcm_s16le",
        f"ffprobe codec={wav_summary['codec_name']}",
    )
    _check(
        checks,
        "wav_stream_sample_count",
        wav_summary["duration_ts"] is not None
        and abs(int(wav_summary["duration_ts"]) - expected_samples)
        <= duration_tolerance,
        f"ffprobe duration_ts={wav_summary['duration_ts']}",
    )
    wav_meta = embedded_metadata(wav_probe)
    _check(
        checks,
        "wav_embedded_metadata",
        wav_meta == expected_metadata,
        "WAV comment carries sample rate, channels, and duration metadata",
    )
    _verify_metadata_values(checks, "wav_metadata_values", wav_meta, expected_samples)

    ogg_samples = decode_audio_int16(ogg_path)
    ogg_probe = probe_audio(ogg_path)
    ogg_summary = stream_summary(ogg_probe)
    metrics["ogg"] = ogg_summary
    metrics["ogg_decoded_samples"] = int(ogg_samples.size)
    _check(
        checks,
        "ogg_codec",
        ogg_summary["codec_name"] == "opus",
        f"ffprobe codec={ogg_summary['codec_name']}",
    )
    _check(
        checks,
        "ogg_sample_rate",
        ogg_summary["sample_rate"] == 48000,
        f"ffprobe sample_rate={ogg_summary['sample_rate']}",
    )
    _check(
        checks,
        "ogg_channels",
        ogg_summary["channels"] == 1,
        f"ffprobe channels={ogg_summary['channels']}",
    )
    _check(
        checks,
        "ogg_decoded_sample_count",
        abs(int(ogg_samples.size) - expected_samples) <= duration_tolerance,
        (
            f"decoded samples={ogg_samples.size}, expected={expected_samples}; "
            f"container duration_ts={ogg_summary['duration_ts']} includes Opus pre-skip"
        ),
    )
    ogg_meta = embedded_metadata(ogg_probe)
    _check(
        checks,
        "ogg_embedded_metadata",
        ogg_meta == expected_metadata,
        "Ogg comments carry sample rate, channels, and duration metadata",
    )
    _verify_metadata_values(checks, "ogg_metadata_values", ogg_meta, expected_samples)

    wav_loudness = loudness_lufs_approx(pcm_square_sum(wav_samples), int(wav_samples.size))
    ogg_loudness = loudness_lufs_approx(pcm_square_sum(ogg_samples), int(ogg_samples.size))
    report_loudness = float(report_output["loudness_lufs_approx"])
    metrics["wav_loudness_lufs_approx"] = wav_loudness
    metrics["ogg_loudness_lufs_approx"] = ogg_loudness
    metrics["report_loudness_lufs_approx"] = report_loudness
    _check(
        checks,
        "wav_loudness",
        abs(wav_loudness - report_loudness) <= loudness_tolerance,
        (
            f"WAV={wav_loudness} LUFS-approx, report={report_loudness}, "
            f"tolerance={loudness_tolerance}"
        ),
    )
    _check(
        checks,
        "ogg_loudness",
        abs(ogg_loudness - report_loudness) <= loudness_tolerance,
        (
            f"Ogg={ogg_loudness} LUFS-approx, report={report_loudness}, "
            f"tolerance={loudness_tolerance}"
        ),
    )
    _check(
        checks,
        "cross_format_loudness_consistency",
        abs(wav_loudness - ogg_loudness) <= loudness_tolerance,
        (
            f"WAV={wav_loudness}, Ogg={ogg_loudness}, "
            f"tolerance={loudness_tolerance}"
        ),
    )

    wav_duration_error_samples = abs(int(wav_samples.size) - expected_samples)
    ogg_duration_error_samples = abs(int(ogg_samples.size) - expected_samples)
    metrics["wav_duration_error_samples"] = wav_duration_error_samples
    metrics["ogg_duration_error_samples"] = ogg_duration_error_samples
    _check(
        checks,
        "duration_error_within_one_sample",
        wav_duration_error_samples <= duration_tolerance
        and ogg_duration_error_samples <= duration_tolerance,
        (
            f"WAV error={wav_duration_error_samples} samples, "
            f"Ogg error={ogg_duration_error_samples} samples"
        ),
    )
    return _finish(checks, metrics, 0)


def _verify_metadata_values(
    checks: list[dict[str, Any]],
    name: str,
    metadata: dict[str, Any] | None,
    expected_samples: int,
) -> None:
    okay = (
        isinstance(metadata, dict)
        and metadata.get("sample_rate") == 48000
        and metadata.get("channels") == 1
        and isinstance(metadata.get("duration_ms"), int)
        and isinstance(metadata.get("total_samples"), int)
        and metadata.get("total_samples") == expected_samples
    )
    _check(
        checks,
        name,
        okay,
        "embedded metadata values match 48000 Hz / mono / full timeline",
    )


def _check(
    checks: list[dict[str, Any]],
    name: str,
    condition: bool,
    detail: str,
) -> None:
    checks.append({"name": name, "ok": bool(condition), "detail": detail})


def _finish(
    checks: list[dict[str, Any]], metrics: dict[str, Any], requested_code: int
) -> tuple[dict[str, Any], int]:
    failures = [check for check in checks if not check["ok"]]
    code = 3 if failures else requested_code
    result = {
        "schema": VERIFY_SCHEMA,
        "ok": code == 0,
        "return_code": code,
        "checks": checks,
        "metrics": metrics,
        "failures": failures,
    }
    return result, code


def _failure_result(message: str) -> dict[str, Any]:
    return {
        "schema": VERIFY_SCHEMA,
        "ok": False,
        "return_code": 3,
        "checks": [{"name": "precondition", "ok": False, "detail": message}],
        "metrics": {},
        "failures": [{"name": "precondition", "ok": False, "detail": message}],
    }
