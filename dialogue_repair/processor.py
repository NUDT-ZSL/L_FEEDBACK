"""Deterministic repair execution and report construction."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .atomic import commit_outputs
from .errors import RepairFailure, make_error
from .jsonutil import fixed_json
from .manifest import Manifest, OperationSpec, SAMPLE_RATE, parse_manifest
from .wavio import WavData, WavFormatError, encode_wav, read_wav


@dataclass(frozen=True)
class ProcessResult:
    output_path: Path
    report_path: Path
    sample_count: int
    wav_sha256: str
    report_sha256: str
    report: dict[str, Any]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _input_error(path: str | Path) -> tuple[WavData | None, dict[str, Any] | None]:
    try:
        audio = read_wav(path)
    except FileNotFoundError:
        return None, make_error(
            "input_missing",
            f"input WAV not found: {path}",
            details={"input": str(path)},
        )
    except (OSError, WavFormatError) as exc:
        return None, make_error(
            "input_format_mismatch",
            f"input WAV is unreadable or not PCM: {exc}",
            details={"input": str(path)},
        )

    if audio.sample_rate != SAMPLE_RATE:
        return audio, make_error(
            "input_format_mismatch",
            "input WAV must use a 48000 Hz sample rate",
            details={
                "actual_sample_rate": audio.sample_rate,
                "required_sample_rate": SAMPLE_RATE,
            },
        )
    if audio.channels != 1:
        return audio, make_error(
            "input_format_mismatch",
            "input WAV must be mono",
            details={
                "actual_channels": audio.channels,
                "required_channels": 1,
            },
        )
    return audio, None


def _path_conflicts(
    input_path: Path,
    manifest_path: Path,
    output_path: Path,
    report_path: Path,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    canonical = {
        "input": os.path.normcase(os.path.realpath(os.fspath(input_path))),
        "manifest": os.path.normcase(os.path.realpath(os.fspath(manifest_path))),
        "output": os.path.normcase(os.path.realpath(os.fspath(output_path))),
        "report": os.path.normcase(os.path.realpath(os.fspath(report_path))),
    }
    if canonical["output"] == canonical["report"]:
        errors.append(
            make_error(
                "output_path_conflict",
                "output WAV and report must use different paths",
                details={
                    "output": str(output_path),
                    "report": str(report_path),
                },
            )
        )
    for target_name in ("output", "report"):
        for source_name in ("input", "manifest"):
            if canonical[target_name] == canonical[source_name]:
                errors.append(
                    make_error(
                        "output_path_conflict",
                        f"{target_name} path would overwrite the {source_name} file",
                        details={
                            "target": canonical[target_name],
                            "source": canonical[source_name],
                        },
                    )
                )
    return errors


def _levels(segment: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(segment, dtype=np.float64)
    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))
    peak_dbfs = 20.0 * math.log10(peak) if peak > 0.0 else None
    rms_dbfs = 20.0 * math.log10(rms) if rms > 0.0 else None
    return {
        "peak_linear": peak,
        "rms_linear": rms,
        "peak_dbfs": peak_dbfs,
        "rms_dbfs": rms_dbfs,
    }


def _fade_weights(length: int) -> np.ndarray:
    if length < 2:
        raise ValueError("a crossfade needs at least two samples")
    position = np.arange(length, dtype=np.float64) / float(length - 1)
    return 0.5 - 0.5 * np.cos(math.pi * position)


def _replace_with_crossfade(
    work: np.ndarray,
    start: int,
    end: int,
    replacement: np.ndarray,
    fade_samples: int,
) -> None:
    old = work[start:end].copy()
    work[start:end] = replacement
    if fade_samples == 0:
        return
    weights = _fade_weights(fade_samples)
    work[start : start + fade_samples] = (
        old[:fade_samples] * (1.0 - weights)
        + replacement[:fade_samples] * weights
    )
    reverse_weights = weights[::-1]
    work[end - fade_samples : end] = (
        replacement[-fade_samples:] * reverse_weights
        + old[-fade_samples:] * (1.0 - reverse_weights)
    )


def _interval(start: int, end: int) -> dict[str, int]:
    return {"start_sample": start, "end_sample": end}


def _operation_report(
    spec: OperationSpec,
    *,
    before: np.ndarray,
    after: np.ndarray,
) -> dict[str, Any]:
    assert spec.start is not None and spec.end is not None
    fade = spec.fade_samples
    fade_report = {
        "requested_ms": spec.fade_ms,
        "actual_ms": fade * 1000.0 / float(SAMPLE_RATE),
        "length_samples": fade,
        "curve": "raised_cosine",
        "left_interval": _interval(spec.start, spec.start + fade),
        "right_interval": _interval(spec.end - fade, spec.end),
        "boundary_values_continuous": fade > 0,
    }
    item: dict[str, Any] = {
        "operation_index": spec.position,
        "operation_type": spec.operation_type,
        "target_interval": _interval(spec.start, spec.end),
        "actual_interval": _interval(spec.start, spec.end),
        "fade_length_samples": fade,
        "fade": fade_report,
        "levels": {
            "before": _levels(before),
            "after": _levels(after),
        },
    }
    if spec.operation_type == "replace":
        item["replace"] = {
            "frequency_hz": spec.frequency_hz,
            "phase_radians": spec.phase_radians,
            "amplitude": spec.amplitude,
            "sample_indexing": "absolute_output_sample",
        }
    elif spec.operation_type == "splice":
        assert spec.reference_audio is not None
        assert spec.reference_start is not None and spec.reference_end is not None
        item["splice"] = {
            "reference_file": spec.reference_file,
            "reference_interval": _interval(spec.reference_start, spec.reference_end),
            "reference_sample_count": (
                spec.reference_end - spec.reference_start
            ),
            "reference_sample_rate": spec.reference_audio.sample_rate,
            "reference_channels": spec.reference_audio.channels,
            "reference_bit_depth": spec.reference_audio.bit_depth,
            "reference_file_sha256": spec.reference_audio.file_sha256,
        }
    elif spec.operation_type == "gain":
        item["gain"] = {"linear_gain": spec.gain}
    return item


def _render(
    audio: WavData, manifest: Manifest, quantization_bits: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    work = np.asarray(audio.mono_samples, dtype=np.float64).copy()
    operation_reports: list[dict[str, Any]] = []
    for spec in manifest.operations:
        assert spec.start is not None and spec.end is not None
        before = work[spec.start : spec.end].copy()
        if spec.operation_type == "replace":
            assert spec.frequency_hz is not None
            assert spec.phase_radians is not None
            assert spec.amplitude is not None
            sample_numbers = np.arange(spec.start, spec.end, dtype=np.float64)
            replacement = spec.amplitude * np.sin(
                (2.0 * math.pi * spec.frequency_hz / float(SAMPLE_RATE))
                * sample_numbers
                + spec.phase_radians
            )
            _replace_with_crossfade(
                work,
                spec.start,
                spec.end,
                replacement,
                spec.fade_samples,
            )
        elif spec.operation_type == "splice":
            assert spec.reference_audio is not None
            assert spec.reference_start is not None and spec.reference_end is not None
            reference = spec.reference_audio.mono_samples
            replacement = np.asarray(
                reference[spec.reference_start : spec.reference_end],
                dtype=np.float64,
            ).copy()
            _replace_with_crossfade(
                work,
                spec.start,
                spec.end,
                replacement,
                spec.fade_samples,
            )
        elif spec.operation_type == "gain":
            assert spec.gain is not None
            work[spec.start : spec.end] = before * spec.gain
        else:
            raise ValueError(f"unsupported operation reached renderer: {spec.operation_type}")
        after = work[spec.start : spec.end].copy()
        operation_reports.append(_operation_report(spec, before=before, after=after))

    if work.shape != (audio.sample_count,):
        raise ValueError("internal sample-count invariant failed")
    if quantization_bits not in (16, 24):
        raise ValueError("invalid output quantization")
    return work, operation_reports


def _build_report(
    *,
    audio: WavData,
    process: np.ndarray,
    quantization_bits: int,
    operation_reports: list[dict[str, Any]],
    wav_bytes: bytes,
    pcm_bytes: bytes,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "input": {
            "sample_rate": audio.sample_rate,
            "channels": audio.channels,
            "bit_depth": audio.bit_depth,
            "sample_count": audio.sample_count,
            "file_sha256": audio.file_sha256,
        },
        "output": {
            "format": "WAV_PCM",
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "sample_count": int(process.shape[0]),
            "quantization_bits": quantization_bits,
            "file_sha256": _sha256(wav_bytes),
            "pcm_sha256": _sha256(pcm_bytes),
        },
        "processing": {
            "internal_precision": "float64",
            "fade_curve": "raised_cosine",
            "fade_boundary_samples": "original value at both target boundaries",
            "quantization_rounding": "nearest_half_away_from_zero",
            "quantization_saturation": True,
            "deterministic": True,
        },
        "operations": operation_reports,
    }


def process_files(
    input_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
) -> ProcessResult:
    input_path = Path(input_path)
    manifest_path = Path(manifest_path)
    output_path = Path(output_path)
    report_path = Path(report_path)

    errors = _path_conflicts(
        input_path, manifest_path, output_path, report_path
    )
    audio, input_error = _input_error(input_path)
    if input_error is not None:
        errors.append(input_error)

    manifest = parse_manifest(
        manifest_path,
        audio.sample_count if audio is not None else None,
        errors,
    )
    if errors:
        raise RepairFailure(errors, exit_code=2)
    if audio is None:
        raise RepairFailure(
            [
                make_error(
                    "input_missing",
                    f"input WAV not found or unreadable: {input_path}",
                )
            ],
            exit_code=2,
        )
    if manifest.quantization_bits is None:
        raise RepairFailure(
            [
                make_error(
                    "missing_field",
                    "manifest did not provide output quantization",
                    field="quantization_bits",
                )
            ],
            exit_code=2,
        )

    try:
        process, operation_reports = _render(
            audio, manifest, manifest.quantization_bits
        )
        wav_bytes, pcm_bytes = encode_wav(
            process,
            quantization_bits=manifest.quantization_bits,
            sample_rate=SAMPLE_RATE,
        )
        report = _build_report(
            audio=audio,
            process=process,
            quantization_bits=manifest.quantization_bits,
            operation_reports=operation_reports,
            wav_bytes=wav_bytes,
            pcm_bytes=pcm_bytes,
        )
        report_bytes = (fixed_json(report, indent=2) + "\n").encode("utf-8")
    except RepairFailure:
        raise
    except Exception as exc:
        raise RepairFailure(
            [
                make_error(
                    "processing_failed",
                    f"repair processing failed: {exc}",
                )
            ],
            exit_code=3,
        ) from exc

    try:
        commit_outputs(
            [
                (output_path, wav_bytes),
                (report_path, report_bytes),
            ]
        )
    except Exception as exc:
        raise RepairFailure(
            [
                make_error(
                    "output_write_failed",
                    f"cannot commit output files atomically: {exc}",
                )
            ],
            exit_code=3,
        ) from exc

    return ProcessResult(
        output_path=output_path,
        report_path=report_path,
        sample_count=int(process.shape[0]),
        wav_sha256=_sha256(wav_bytes),
        report_sha256=_sha256(report_bytes),
        report=report,
    )

