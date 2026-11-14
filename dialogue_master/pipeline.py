from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from dialogue_master import __version__
from dialogue_master.audio_io import (
    AudioError,
    FormatMismatch,
    read_matching_wav,
)
from dialogue_master.config import ClipConfig, SessionConfig
from dialogue_master.dsp import (
    amplitude_to_dbfs,
    apply_cosine_squared_fades,
    fade_region_peak,
    find_placement,
    milliseconds_to_samples,
    samples_to_milliseconds,
)
from dialogue_master.loudness import integrated_loudness


GAIN_THRESHOLD_LU = 10.0
MAX_GAIN_DB = 20.0
PEAK_LIMIT_DBFS = 0.0
FLOAT_EPSILON = 1e-12


@dataclass
class ProcessingResult:
    log: Dict[str, Any]
    master: np.ndarray
    messages: List[str]
    exit_code: int


def _rounded(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    result = round(float(value), digits)
    if result == 0.0:
        return 0.0
    return result


def _round_trip_ms(value: Any) -> Any:
    if isinstance(value, int):
        return value
    return _rounded(float(value), 6)


def _warning(code: str, message: str) -> Dict[str, str]:
    return {"code": code, "message": message}


def _error(code: str, message: str) -> Dict[str, str]:
    return {"code": code, "message": message}


def _base_entry(clip: ClipConfig, sample_rate: int) -> Dict[str, Any]:
    requested_sample = milliseconds_to_samples(clip.start_ms, sample_rate)
    return {
        "index": clip.index,
        "id": clip.clip_id,
        "input_path": clip.input_path,
        "status": "skipped",
        "warnings": [],
        "errors": [],
        "requested_start_ms": _round_trip_ms(clip.start_ms),
        "requested_start_sample": requested_sample,
        "final_start_ms": None,
        "final_start_sample": None,
        "sample_shift": None,
        "duration_samples": None,
        "duration_ms": None,
        "integrated_lufs": None,
        "target_lufs": _rounded(clip.target_lufs),
        "loudness_delta_db": None,
        "gain_requested_db": None,
        "gain_db": None,
        "gain_linear": None,
        "peak_before_gain_dbfs": None,
        "peak_after_gain_dbfs": None,
        "peak_constrained": False,
        "fade_in_samples": None,
        "fade_out_samples": None,
        "fade_in_applied_samples": None,
        "fade_out_applied_samples": None,
        "fade_region_peak_dbfs": None,
    }


def _add_warning(
    entry: Dict[str, Any],
    messages: List[str],
    clip_id: str,
    code: str,
    message: str,
) -> None:
    entry["warnings"].append(_warning(code, message))
    messages.append("{}: clip={} {}".format(code, clip_id, message))


def _prepare_clip_audio(
    audio: np.ndarray,
    clip: ClipConfig,
    config: SessionConfig,
    entry: Dict[str, Any],
    messages: List[str],
) -> np.ndarray:
    frame_count = audio.shape[0]
    entry["duration_samples"] = frame_count
    entry["duration_ms"] = samples_to_milliseconds(
        frame_count, config.sample_rate
    )

    fade_in_samples = milliseconds_to_samples(
        clip.fade_in_ms, config.sample_rate
    )
    fade_out_samples = milliseconds_to_samples(
        clip.fade_out_ms, config.sample_rate
    )
    entry["fade_in_samples"] = fade_in_samples
    entry["fade_out_samples"] = fade_out_samples
    applied_fade_in = min(fade_in_samples, frame_count)
    applied_fade_out = min(fade_out_samples, frame_count)
    entry["fade_in_applied_samples"] = applied_fade_in
    entry["fade_out_applied_samples"] = applied_fade_out
    if fade_in_samples > frame_count or fade_out_samples > frame_count:
        _add_warning(
            entry,
            messages,
            clip.clip_id,
            "W_FADE",
            "requested fade exceeds clip length; fade window truncated "
            "at the clip boundary",
        )

    peak_before = float(np.max(np.abs(audio))) if frame_count else 0.0
    entry["peak_before_gain_dbfs"] = _rounded(
        amplitude_to_dbfs(peak_before)
    )

    loudness = integrated_loudness(
        audio, config.sample_rate, config.channel_labels
    )
    entry["integrated_lufs"] = _rounded(loudness)
    if loudness is None:
        _add_warning(
            entry,
            messages,
            clip.clip_id,
            "W_SILENT",
            "integrated loudness is not measurable; gain left at 0 dB",
        )
        requested_gain_db = 0.0
    else:
        loudness_delta_db = clip.target_lufs - loudness
        entry["loudness_delta_db"] = _rounded(loudness_delta_db)
        if abs(loudness_delta_db) > GAIN_THRESHOLD_LU:
            requested_gain_db = loudness_delta_db
        else:
            requested_gain_db = 0.0

    entry["gain_requested_db"] = _rounded(requested_gain_db)
    final_gain_db = max(
        -MAX_GAIN_DB, min(MAX_GAIN_DB, requested_gain_db)
    )
    if not math.isclose(
        final_gain_db, requested_gain_db, abs_tol=FLOAT_EPSILON
    ):
        _add_warning(
            entry,
            messages,
            clip.clip_id,
            "W_GAIN_LIMIT",
            "requested gain {:.6f} dB limited to {:.6f} dB".format(
                requested_gain_db, final_gain_db
            ),
        )

    peak_after = peak_before * (10.0 ** (final_gain_db / 20.0))
    if peak_after > 1.0 + FLOAT_EPSILON and peak_before > 0.0:
        headroom_gain_db = -20.0 * math.log10(peak_before)
        constrained_gain_db = min(final_gain_db, headroom_gain_db)
        if constrained_gain_db < final_gain_db - FLOAT_EPSILON:
            _add_warning(
                entry,
                messages,
                clip.clip_id,
                "W_PEAK",
                "gain constrained from {:.6f} dB to {:.6f} dB to keep "
                "the clip at or below 0 dBFS".format(
                    final_gain_db, constrained_gain_db
                ),
            )
            final_gain_db = constrained_gain_db
            peak_after = peak_before * (10.0 ** (final_gain_db / 20.0))
            entry["peak_constrained"] = True

    gain_linear = 10.0 ** (final_gain_db / 20.0)
    entry["gain_db"] = _rounded(final_gain_db)
    entry["gain_linear"] = _rounded(gain_linear, 9)
    entry["peak_after_gain_dbfs"] = _rounded(
        amplitude_to_dbfs(peak_after)
    )

    processed = audio * gain_linear
    processed = apply_cosine_squared_fades(
        processed,
        applied_fade_in,
        applied_fade_out,
    )
    fade_peak = fade_region_peak(
        processed, applied_fade_in, applied_fade_out
    )
    entry["fade_region_peak_dbfs"] = _rounded(
        amplitude_to_dbfs(fade_peak)
    )
    if fade_peak > 1.0 + FLOAT_EPSILON:
        safety = 1.0 / fade_peak
        processed *= safety
        final_gain_db += 20.0 * math.log10(safety)
        entry["gain_db"] = _rounded(final_gain_db)
        entry["gain_linear"] = _rounded(10.0 ** (final_gain_db / 20.0), 9)
        entry["peak_after_gain_dbfs"] = _rounded(
            amplitude_to_dbfs(peak_before * (10.0 ** (final_gain_db / 20.0)))
        )
        entry["peak_constrained"] = True
        _add_warning(
            entry,
            messages,
            clip.clip_id,
            "W_PEAK",
            "fade-region safety attenuation applied to keep peak at or "
            "below 0 dBFS",
        )
    return processed


def _sanitize_audio_error(exc: Exception, clip: ClipConfig) -> str:
    """Keep decoder diagnostics but remove the host-specific absolute path."""

    return str(exc).replace(str(clip.resolved_path), clip.input_path)


def _summary(
    entries: Sequence[Dict[str, Any]],
    overall_warnings: Sequence[Dict[str, str]],
    config: SessionConfig,
    master_peak_dbfs: Optional[float],
    clipped_samples: int,
    exit_code: int,
) -> Dict[str, Any]:
    placed = sum(
        1 for entry in entries if entry["final_start_sample"] is not None
    )
    skipped = len(entries) - placed
    success = sum(1 for entry in entries if entry["status"] == "success")
    warning_status = sum(
        1 for entry in entries if entry["status"] == "warning"
    )
    code_counts: Dict[str, int] = {}
    for entry in entries:
        for warning in entry["warnings"]:
            code_counts[warning["code"]] = (
                code_counts.get(warning["code"], 0) + 1
            )
        for error in entry["errors"]:
            code_counts[error["code"]] = (
                code_counts.get(error["code"], 0) + 1
            )
    for warning in overall_warnings:
        code_counts[warning["code"]] = code_counts.get(warning["code"], 0) + 1

    if placed == 0:
        overall_status = "failed"
    elif skipped or overall_warnings or warning_status:
        overall_status = "warning"
    else:
        overall_status = "success"

    return {
        "overall_status": overall_status,
        "clips_total": len(entries),
        "placed": placed,
        "skipped": skipped,
        "success": success,
        "warning": warning_status,
        "warning_count": sum(len(entry["warnings"]) for entry in entries)
        + len(overall_warnings),
        "error_count": sum(len(entry["errors"]) for entry in entries),
        "code_counts": code_counts,
        "samples_placed": sum(
            int(entry["duration_samples"] or 0) for entry in entries
            if entry["final_start_sample"] is not None
        ),
        "timeline_samples": config.timeline_samples,
        "timeline_ms": _round_trip_ms(config.timeline_ms),
        "master_peak_dbfs": _rounded(master_peak_dbfs),
        "master_over_0_dbfs": bool(
            master_peak_dbfs is not None
            and master_peak_dbfs > PEAK_LIMIT_DBFS + FLOAT_EPSILON
        ),
        "samples_over_full_scale": clipped_samples,
        "exit_code": exit_code,
    }


def process(config: SessionConfig) -> ProcessingResult:
    """Process clips in configuration order and mix the complete master."""

    master = np.zeros(
        (config.timeline_samples, config.channels), dtype=np.float64
    )
    placed_intervals: List[Tuple[int, int]] = []
    entries: List[Dict[str, Any]] = []
    messages: List[str] = []

    for clip in config.clips:
        entry = _base_entry(clip, config.sample_rate)
        requested_sample = int(entry["requested_start_sample"])
        try:
            _, audio = read_matching_wav(
                clip.resolved_path,
                config.sample_rate,
                config.bit_depth,
                config.channels,
            )
        except FormatMismatch as exc:
            _add_warning(
                entry,
                messages,
                clip.clip_id,
                "W_FORMAT",
                str(exc) + "; clip skipped",
            )
            entries.append(entry)
            continue
        except AudioError as exc:
            message = _sanitize_audio_error(exc, clip) + "; clip skipped"
            entry["errors"].append(_error("E_AUDIO", message))
            messages.append("E_AUDIO: clip={} {}".format(clip.clip_id, message))
            entries.append(entry)
            continue

        processed = _prepare_clip_audio(
            audio, clip, config, entry, messages
        )
        frame_count = processed.shape[0]
        final_sample = find_placement(
            requested_sample,
            frame_count,
            config.timeline_samples,
            placed_intervals,
        )
        if final_sample is None:
            _add_warning(
                entry,
                messages,
                clip.clip_id,
                "W_LATE",
                "no free timeline region can fit this clip; clip skipped",
            )
            entries.append(entry)
            continue

        shift = final_sample - requested_sample
        if shift:
            _add_warning(
                entry,
                messages,
                clip.clip_id,
                "W_SHIFTED",
                "requested slot was occupied or out of bounds; placed at "
                "the nearest later free sample",
            )
        entry["final_start_sample"] = final_sample
        entry["final_start_ms"] = samples_to_milliseconds(
            final_sample, config.sample_rate
        )
        entry["sample_shift"] = shift
        entry["status"] = "warning" if entry["warnings"] else "success"
        master[final_sample:final_sample + frame_count] += processed
        placed_intervals.append(
            (final_sample, final_sample + frame_count)
        )
        placed_intervals.sort()
        entries.append(entry)

    if master.shape[0] and master.shape[1]:
        master_peak = float(np.max(np.abs(master)))
    else:
        master_peak = 0.0
    master_peak_dbfs = amplitude_to_dbfs(master_peak)
    clipped_samples = int(np.count_nonzero(np.abs(master) > 1.0))
    overall_warnings: List[Dict[str, str]] = []
    if master_peak > 1.0 + FLOAT_EPSILON:
        message = (
            "mixed master peak is {:.6f} dBFS, above 0 dBFS; "
            "the WAV will be explicitly reported as clipped"
        ).format(master_peak_dbfs if master_peak_dbfs is not None else 0.0)
        overall_warnings.append(_warning("W_PEAK", message))
        messages.append("W_PEAK: master {}".format(message))

    placed_count = sum(
        1 for entry in entries if entry["final_start_sample"] is not None
    )
    exit_code = 0 if placed_count > 0 else 2
    summary = _summary(
        entries,
        overall_warnings,
        config,
        master_peak_dbfs,
        clipped_samples,
        exit_code,
    )
    log: Dict[str, Any] = {
        "schema_version": 1,
        "processor": {
            "name": "dialogue-master",
            "version": __version__,
        },
        "config": {
            "timeline_ms": _round_trip_ms(config.timeline_ms),
            "timeline_samples": config.timeline_samples,
            "sample_rate": config.sample_rate,
            "bit_depth": config.bit_depth,
            "channels": config.channels,
            "channel_layout": config.channel_layout,
            "channel_labels": list(config.channel_labels),
        },
        "policy": {
            "loudness_standard": "ITU-R BS.1770-4",
            "gain_threshold_lu": GAIN_THRESHOLD_LU,
            "max_gain_db": MAX_GAIN_DB,
            "peak_limit_dbfs": PEAK_LIMIT_DBFS,
            "fade_shape": "cosine_squared",
            "sample_rounding": "round_half_up",
            "placement": "original_order_greedy_forward",
        },
        "clips": entries,
        "overall_warnings": overall_warnings,
        "summary": summary,
    }
    return ProcessingResult(
        log=log,
        master=master,
        messages=messages,
        exit_code=exit_code,
    )
