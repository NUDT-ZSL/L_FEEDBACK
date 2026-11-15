"""End-to-end audio repair and atomic WAV/report production."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import loudness, wavio, wsola
from .errors import (
    BOUNDARY_MISALIGNED,
    INPUT_MISSING,
    LOUDNESS_UNREACHABLE,
    PROCESSING_FAILED,
    RepairError,
)
from .timeline import SegmentSpec, Timeline, load_timeline

PEAK_LIMIT_DBFS = -1.0
# Keep a small quantisation guard below the hard -1 dBFS limit.
OUTPUT_PEAK_CEILING = 10.0 ** (-1.05 / 20.0)
DC_LIMIT = 0.001
BOUNDARY_TOLERANCE_MS = 20.0


@dataclass
class RenderedSegment:
    spec: SegmentSpec
    samples: np.ndarray
    original_loudness_lufs: float
    stretched_loudness_lufs: float
    target_gain_db: float
    span_ratios: Tuple[float, ...]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(document: Dict[str, Any]) -> bytes:
    text = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    return (text + "\n").encode("utf-8")


def _finite_or_none(value: float) -> Optional[float]:
    return None if not np.isfinite(value) else float(value)


def _apply_gain(samples: np.ndarray, gain_db: float) -> np.ndarray:
    return np.asarray(samples, dtype=np.float64) * (10.0 ** (gain_db / 20.0))


def _soft_limit(samples: np.ndarray, ceiling: float) -> np.ndarray:
    """Smoothly bound peaks without hard clipping."""
    audio = np.asarray(samples, dtype=np.float64)
    peak = loudness.sample_peak(audio)
    if peak <= ceiling:
        return audio.copy()
    threshold = ceiling * 0.80
    magnitude = np.abs(audio)
    above = magnitude > threshold
    result = audio.copy()
    if np.any(above):
        span = max(ceiling - threshold, 1e-12)
        limited = threshold + span * np.tanh((magnitude[above] - threshold) / span)
        result[above] *= limited / magnitude[above]
    return result


def _normalise_mix(
    samples: np.ndarray,
    sample_rate: int,
    target_lufs: float,
    tolerance_lu: float,
) -> Tuple[np.ndarray, float, bool]:
    """Closed-loop integrated-loudness correction with peak protection."""
    audio = np.asarray(samples, dtype=np.float64).copy()
    if audio.size:
        audio -= np.mean(audio, axis=0, keepdims=True)
    accumulated_gain_db = 0.0
    limiter_used = False
    current_lufs = loudness.integrated_loudness(audio, sample_rate)
    if not np.isfinite(current_lufs):
        raise RepairError(
            LOUDNESS_UNREACHABLE,
            "the assembled programme has no measurable integrated loudness",
        )

    for _iteration in range(8):
        error_db = target_lufs - current_lufs
        if abs(error_db) < 1e-7:
            break
        audio = _apply_gain(audio, error_db)
        accumulated_gain_db += error_db
        if loudness.sample_peak(audio) > OUTPUT_PEAK_CEILING:
            audio = _soft_limit(audio, OUTPUT_PEAK_CEILING)
            limiter_used = True
        current_lufs = loudness.integrated_loudness(audio, sample_rate)
        if not np.isfinite(current_lufs):
            raise RepairError(
                LOUDNESS_UNREACHABLE,
                "loudness became unmeasurable during normalisation",
            )
        if abs(target_lufs - current_lufs) <= min(1e-4, tolerance_lu * 0.01):
            break

    if abs(target_lufs - current_lufs) > tolerance_lu:
        raise RepairError(
            LOUDNESS_UNREACHABLE,
            "cannot reach target loudness while respecting the peak ceiling",
            {
                "target_lufs": target_lufs,
                "measured_lufs": round(current_lufs, 6),
                "tolerance_lu": tolerance_lu,
            },
        )
    if audio.size:
        audio -= np.mean(audio, axis=0, keepdims=True)
    peak = loudness.sample_peak(audio)
    if peak > OUTPUT_PEAK_CEILING:
        scale = OUTPUT_PEAK_CEILING / peak
        audio *= scale
        accumulated_gain_db += 20.0 * np.log10(scale)
    return audio, float(accumulated_gain_db), limiter_used


def _linear_crossfade_concat(
    pieces: Sequence[np.ndarray],
    crossfade_samples: Sequence[int],
) -> Tuple[np.ndarray, List[int], List[dict]]:
    if not pieces:
        raise RepairError(PROCESSING_FAILED, "timeline produced no segments")
    output = np.asarray(pieces[0], dtype=np.float64).copy()
    starts = [0]
    crossfades: List[dict] = []
    for index, piece in enumerate(pieces[1:]):
        duration = int(crossfade_samples[index])
        if duration < 0 or duration >= piece.shape[0] or duration >= output.shape[0]:
            raise RepairError(
                PROCESSING_FAILED, "invalid crossfade length reached the mixer"
            )
        start = output.shape[0] - duration
        if duration:
            ramp = np.linspace(0.0, 1.0, duration, dtype=np.float64)[:, None]
            output[start:] = (
                output[start:] * (1.0 - ramp) + piece[:duration] * ramp
            )
            remainder = piece[duration:]
            output = np.concatenate((output, remainder), axis=0)
        else:
            output = np.concatenate((output, piece), axis=0)
        starts.append(start)
        crossfades.append(
            {
                "splice_index": index,
                "start_sample": int(start),
                "end_sample": int(start + duration),
                "duration_samples": int(duration),
                "curve": "linear",
            }
        )
    return output, starts, crossfades


def _output_region_loudness(
    output: np.ndarray,
    sample_rate: int,
    start: int,
    end: int,
) -> float:
    start = max(0, min(int(start), output.shape[0]))
    end = max(start, min(int(end), output.shape[0]))
    if end <= start:
        return float("-inf")
    return loudness.integrated_loudness(output[start:end], sample_rate)


def _atomic_write_pair(
    wav_path: str,
    report_path: str,
    wav_payload: bytes,
    report_payload: bytes,
) -> None:
    """Staged write with an explicit incomplete marker."""
    wav_path = os.path.abspath(wav_path)
    report_path = os.path.abspath(report_path)
    if wav_path == report_path:
        raise RepairError(PROCESSING_FAILED, "output and report paths must differ")
    wav_dir = os.path.dirname(wav_path)
    report_dir = os.path.dirname(report_path)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    marker_path = wav_path + ".incomplete"
    marker_temp = None
    wav_temp = None
    report_temp = None
    wav_committed = False
    report_committed = False
    try:
        marker_fd, marker_temp = tempfile.mkstemp(
            prefix=".audiorepair-marker-", dir=wav_dir
        )
        with os.fdopen(marker_fd, "w", encoding="utf-8") as handle:
            handle.write('{"status":"incomplete"}\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(marker_temp, marker_path)
        marker_temp = None

        wav_fd, wav_temp = tempfile.mkstemp(
            prefix=".audiorepair-wav-", suffix=".tmp", dir=wav_dir
        )
        with os.fdopen(wav_fd, "wb") as handle:
            handle.write(wav_payload)
            handle.flush()
            os.fsync(handle.fileno())
        report_fd, report_temp = tempfile.mkstemp(
            prefix=".audiorepair-report-", suffix=".tmp", dir=report_dir
        )
        with os.fdopen(report_fd, "wb") as handle:
            handle.write(report_payload)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(wav_temp, wav_path)
        wav_temp = None
        wav_committed = True
        os.replace(report_temp, report_path)
        report_temp = None
        report_committed = True
        try:
            os.unlink(marker_path)
        except FileNotFoundError:
            pass
    except Exception:
        if wav_committed:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
        if report_committed:
            try:
                os.unlink(report_path)
            except OSError:
                pass
        raise
    finally:
        for temp_path in (marker_temp, wav_temp, report_temp):
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass


def process_audio(
    audio: wavio.WavAudio,
    timeline: Timeline,
) -> Tuple[bytes, Dict[str, Any]]:
    """Process already-decoded audio and return WAV bytes plus report."""
    sample_rate = audio.sample_rate
    channels = audio.channels
    if sample_rate not in loudness._COEFFICIENTS:
        raise RepairError(
            "AUDIO_UNSUPPORTED_FORMAT",
            "no BS.1770 coefficients for sample rate {}".format(sample_rate),
        )

    rendered_segments: List[RenderedSegment] = []
    for spec in timeline.segments:
        source = audio.samples[spec.start_sample : spec.end_sample]
        original_lufs = loudness.integrated_loudness(source, sample_rate)
        if not np.isfinite(original_lufs):
            raise RepairError(
                LOUDNESS_UNREACHABLE,
                "segment {} is silent and cannot be normalised".format(spec.index),
                {"segment": spec.index},
            )
        stretched, span_ratios = wsola.stretch_segment(source, spec, sample_rate)
        stretched_lufs = loudness.integrated_loudness(stretched, sample_rate)
        if not np.isfinite(stretched_lufs):
            raise RepairError(
                LOUDNESS_UNREACHABLE,
                "segment {} became silent after stretching".format(spec.index),
                {"segment": spec.index},
            )
        target_gain_db = spec.target_lufs - stretched_lufs
        normalised = _apply_gain(stretched, target_gain_db)
        rendered_segments.append(
            RenderedSegment(
                spec=spec,
                samples=normalised,
                original_loudness_lufs=original_lufs,
                stretched_loudness_lufs=stretched_lufs,
                target_gain_db=target_gain_db,
                span_ratios=tuple(span_ratios),
            )
        )

    crossfade_samples = [
        int(round(duration_ms * sample_rate / 1000.0))
        for duration_ms in timeline.crossfade_ms
    ]
    mixed, output_starts, crossfades = _linear_crossfade_concat(
        [segment.samples for segment in rendered_segments],
        crossfade_samples,
    )
    mixed, global_gain_db, limiter_used = _normalise_mix(
        mixed,
        sample_rate,
        timeline.target_lufs,
        timeline.tolerance_lu,
    )

    wav_payload = wavio.encode_wav(mixed, sample_rate, channels)
    decoded, decoded_rate, decoded_channels = wavio.decode_wav_bytes(wav_payload)
    if decoded_rate != sample_rate or decoded_channels != channels:
        raise RepairError(
            PROCESSING_FAILED, "quantised output format changed unexpectedly"
        )

    output_lufs = loudness.integrated_loudness(decoded, sample_rate)
    output_peak = loudness.sample_peak(decoded)
    output_peak_dbfs = (
        float("-inf") if output_peak <= 0.0 else float(20.0 * np.log10(output_peak))
    )
    output_dc = loudness.dc_offset(decoded)
    max_channel_dc = float(
        np.max(np.abs(np.mean(decoded, axis=0))) if decoded.size else 0.0
    )
    if (
        not np.isfinite(output_lufs)
        or abs(output_lufs - timeline.target_lufs) > timeline.tolerance_lu
    ):
        raise RepairError(
            LOUDNESS_UNREACHABLE,
            "quantised output misses the loudness tolerance",
            {
                "measured_lufs": _finite_or_none(output_lufs),
                "target_lufs": timeline.target_lufs,
                "tolerance_lu": timeline.tolerance_lu,
            },
        )
    if output_peak_dbfs > PEAK_LIMIT_DBFS:
        raise RepairError(
            PROCESSING_FAILED,
            "quantised output exceeds the peak ceiling",
            {"peak_dbfs": round(output_peak_dbfs, 6)},
        )
    if abs(output_dc) >= DC_LIMIT or max_channel_dc >= DC_LIMIT:
        raise RepairError(
            PROCESSING_FAILED,
            "output DC offset exceeds {:.6f}".format(DC_LIMIT),
            {
                "dc_offset": round(output_dc, 9),
                "max_channel_dc_offset": round(max_channel_dc, 9),
            },
        )

    segment_entries: List[Dict[str, Any]] = []
    all_boundaries: List[dict] = []
    for index, segment in enumerate(rendered_segments):
        spec = segment.spec
        start = output_starts[index]
        xfade_in = crossfade_samples[index - 1] if index > 0 else 0
        xfade_out = (
            crossfade_samples[index]
            if index < len(crossfade_samples)
            else 0
        )
        region_start = start + xfade_in
        region_end = start + spec.output_length - xfade_out
        normalised_lufs = _output_region_loudness(
            decoded, sample_rate, region_start, region_end
        )
        expected_global = [
            start + anchor for anchor in spec.expected_anchors
        ]
        measured = wsola.detect_boundaries(
            decoded, sample_rate, expected_global
        )
        boundaries: List[dict] = []
        for boundary_index, (expected_sample, detection) in enumerate(
            zip(expected_global, measured)
        ):
            details = dict(detection)
            details["index"] = boundary_index
            details["expected_sample"] = int(expected_sample)
            details["actual_sample"] = int(detection["actual_sample"])
            details["expected_local_sample"] = int(spec.expected_anchors[
                boundary_index
            ])
            details["actual_local_sample"] = int(
                detection["actual_sample"] - start
            )
            details["tolerance_ms"] = BOUNDARY_TOLERANCE_MS
            boundaries.append(details)
            all_boundaries.append(details)

        xfade_in_info = None
        if index > 0:
            xfade_in_info = dict(crossfades[index - 1])
        xfade_out_info = None
        if index < len(crossfades):
            xfade_out_info = dict(crossfades[index])
        segment_entries.append(
            {
                "index": spec.index,
                "id": spec.segment_id,
                "original_start_sample": int(spec.start_sample),
                "original_end_sample": int(spec.end_sample),
                "original_start_sec": round(
                    spec.start_sample / float(sample_rate), 6
                ),
                "original_end_sec": round(spec.end_sample / float(sample_rate), 6),
                "output_start_sample": int(start),
                "output_length_samples": int(spec.output_length),
                "original_loudness_lufs": round(
                    segment.original_loudness_lufs, 6
                ),
                "stretched_loudness_lufs": round(
                    segment.stretched_loudness_lufs, 6
                ),
                "normalized_loudness_lufs": (
                    None
                    if not np.isfinite(normalised_lufs)
                    else round(normalised_lufs, 6)
                ),
                "target_lufs": spec.target_lufs,
                "target_gain_db": round(segment.target_gain_db, 6),
                "applied_gain_db": round(
                    segment.target_gain_db + global_gain_db, 6
                ),
                "stretch_ratio": round(spec.stretch_ratio, 9),
                "span_stretch_ratios": [
                    round(ratio, 9) for ratio in segment.span_ratios
                ],
                "reachable_range": [spec.stretch_min, spec.stretch_max],
                "crossfade_in": xfade_in_info,
                "crossfade_out": xfade_out_info,
                "boundaries": boundaries,
            }
        )

    max_boundary_error_ms = max(
        (abs(boundary["error_ms"]) for boundary in all_boundaries),
        default=0.0,
    )
    if max_boundary_error_ms > BOUNDARY_TOLERANCE_MS:
        raise RepairError(
            BOUNDARY_MISALIGNED,
            "measured boundary error {:.3f} ms exceeds tolerance".format(
                max_boundary_error_ms
            ),
            {
                "max_boundary_error_ms": round(max_boundary_error_ms, 6),
                "tolerance_ms": BOUNDARY_TOLERANCE_MS,
            },
        )

    report = {
        "version": 1,
        "status": "complete",
        "standard": "ITU-R BS.1770 / EBU R128",
        "parameters": {
            "target_lufs": timeline.target_lufs,
            "tolerance_lu": timeline.tolerance_lu,
            "boundary_tolerance_ms": BOUNDARY_TOLERANCE_MS,
            "peak_limit_dbfs": PEAK_LIMIT_DBFS,
            "dc_limit": DC_LIMIT,
            "crossfade_curve": "linear",
            "wsola_frame_ms": wsola.FRAME_SECONDS * 1000.0,
            "wsola_search_ms": wsola.SEARCH_SECONDS * 1000.0,
        },
        "input": {
            "file": os.path.basename(timeline.path) if timeline.path == "<memory>" else "",
            "timeline_file": os.path.basename(timeline.path),
            "sample_rate": sample_rate,
            "channels": channels,
            "bits_per_sample": audio.bits_per_sample,
            "frames": audio.frames,
        },
        "output": {
            "sample_rate": sample_rate,
            "channels": channels,
            "bits_per_sample": 16,
            "frames": int(decoded.shape[0]),
            "duration_sec": round(decoded.shape[0] / float(sample_rate), 6),
            "size_bytes": len(wav_payload),
            "sha256": _sha256_bytes(wav_payload),
        },
        "verification": {
            "integrated_loudness_lufs": round(output_lufs, 6),
            "loudness_error_lu": round(output_lufs - timeline.target_lufs, 6),
            "peak_dbfs": None if not np.isfinite(output_peak_dbfs)
            else round(output_peak_dbfs, 6),
            "dc_offset": round(output_dc, 9),
            "max_channel_dc_offset": round(max_channel_dc, 9),
            "max_boundary_error_ms": round(max_boundary_error_ms, 6),
            "boundary_count": len(all_boundaries),
            "stem_gain_db": round(global_gain_db, 6),
            "soft_limiter_used": limiter_used,
        },
        "crossfades": crossfades,
        "segments": segment_entries,
    }
    return wav_payload, report


def run_job(
    input_path: str,
    timeline_path: str,
    output_path: str,
    report_path: str,
) -> Dict[str, Any]:
    """Run one independent file job.  No output exists on failure."""
    if not os.path.isfile(input_path):
        raise RepairError(INPUT_MISSING, "input WAV not found: {}".format(input_path))
    if not os.path.isfile(timeline_path):
        raise RepairError(
            INPUT_MISSING, "timeline file not found: {}".format(timeline_path)
        )
    try:
        with open(input_path, "rb") as handle:
            input_bytes = handle.read()
        with open(timeline_path, "rb") as handle:
            timeline_bytes = handle.read()
    except OSError as exc:
        raise RepairError(INPUT_MISSING, "cannot read input: {}".format(exc))

    audio = wavio.parse_wav_bytes(input_bytes)
    timeline = load_timeline(timeline_path, audio.frames, audio.sample_rate)
    wav_payload, report = process_audio(audio, timeline)
    report["input"]["file"] = os.path.basename(input_path)
    report["input"]["input_sha256"] = _sha256_bytes(input_bytes)
    report["input"]["timeline_sha256"] = _sha256_bytes(timeline_bytes)
    report_payload = _json_bytes(report)
    try:
        _atomic_write_pair(
            output_path, report_path, wav_payload, report_payload
        )
    except RepairError:
        raise
    except OSError as exc:
        raise RepairError(
            PROCESSING_FAILED, "failed to commit output: {}".format(exc)
        )
    return {
        "status": "ok",
        "output": os.path.abspath(output_path),
        "report": os.path.abspath(report_path),
        "integrated_loudness_lufs": report["verification"][
            "integrated_loudness_lufs"
        ],
        "max_boundary_error_ms": report["verification"][
            "max_boundary_error_ms"
        ],
    }

