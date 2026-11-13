"""End-to-end deterministic render pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import os
from typing import Any

import numpy as np

from .automation import build_gain_curve
from .dsp import (
    apply_lookahead_limiter,
    channel_layout,
    detect_sync_marker,
    integrated_loudness,
    limiter_lookahead_samples,
    resample_polyphase,
)
from .project import (
    OutputConfig,
    Project,
    ProjectError,
    Track,
    header_error_to_source_error,
    validate_headers,
)
from .report import write_report
from .util import db_to_linear, linear_to_db, peak_dbfs, rms, rms_dbfs, round_half_even
from .wavio import WavError, read_wav, read_wav_header, write_wav


@dataclass
class RenderResult:
    exit_code: int
    report: dict[str, Any]
    output_path: str
    report_path: str


def _stage_stats(samples: np.ndarray) -> dict[str, Any]:
    return {
        "samples": int(samples.shape[0]),
        "channels": int(samples.shape[1]),
        "peak": float(np.max(np.abs(samples))) if samples.size else 0.0,
        "peak_dbfs": peak_dbfs(samples),
        "rms": rms(samples),
        "rms_dbfs": rms_dbfs(samples),
    }


def _align_in_output_domain(
    samples: np.ndarray, offset_output_samples: int
) -> np.ndarray:
    data = np.asarray(samples, dtype=np.float64)
    if offset_output_samples > 0:
        return data[min(offset_output_samples, data.shape[0]) :, :].copy()
    if offset_output_samples < 0:
        padding = np.zeros((-offset_output_samples, data.shape[1]), dtype=np.float64)
        return np.concatenate((padding, data), axis=0)
    return data.copy()


def _fit_timeline(samples: np.ndarray, frame_count: int) -> np.ndarray:
    if samples.shape[0] >= frame_count:
        return samples[:frame_count, :].copy()
    padding = np.zeros((frame_count - samples.shape[0], samples.shape[1]), dtype=np.float64)
    return np.concatenate((samples, padding), axis=0)


def _warning(track: str, code: str, detail: str) -> dict[str, str]:
    return {"track": track, "code": code, "detail": detail}


def _expected_marker_position(sync: dict[str, Any]) -> int:
    for key in (
        "expected_sample_position",
        "expected_position",
        "expected_marker_position",
        "marker_start_sample",
    ):
        if key in sync:
            return int(sync[key])
    return 0


def render_project(
    project: Project,
    config: OutputConfig,
    output_path: str,
    report_path: str,
) -> RenderResult:
    output_identity = os.path.normcase(os.path.realpath(output_path))
    report_identity = os.path.normcase(os.path.realpath(report_path))
    if output_identity == report_identity:
        raise ProjectError("OUTPUT_PATH_CONFLICT", "output WAV and report paths are identical")
    for track in project.tracks:
        source_identity = os.path.normcase(os.path.realpath(track.source_path))
        if source_identity in (output_identity, report_identity):
            raise ProjectError(
                "OUTPUT_PATH_CONFLICT",
                f"output/report path would overwrite source track {track.name}",
            )
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    if os.path.exists(output_path):
        os.unlink(output_path)
    header_summaries, header_errors = validate_headers(project)
    del header_summaries
    errors = [header_error_to_source_error(error) for error in header_errors]
    failed_tracks = {error["track"] for error in errors}
    warnings: list[dict[str, str]] = []
    prepared: list[dict[str, Any]] = []

    for track in project.sorted_tracks:
        if track.name in failed_tracks:
            continue
        warnings_before = len(warnings)
        try:
            source, header = read_wav(track.source_path)
            source_frames = int(source.shape[0])
            sync = track.sync or {}
            expected_position = _expected_marker_position(sync)
            default_threshold = int(sync.get("threshold_samples", 96))
            detection = detect_sync_marker(
                source,
                header.sample_rate,
                expected_position,
                sync if track.sync is not None else None,
                default_threshold,
            )
            offset_samples = int(detection["offset_samples"])
            if not detection["detected"]:
                warnings.append(
                    _warning(
                        track.name,
                        "SYNC_MARKER_NOT_FOUND",
                        f"expected_position={expected_position}; using offset 0",
                    )
                )
                detection["offset_samples"] = 0
                offset_samples = 0
            elif abs(offset_samples) > int(detection["threshold_samples"]):
                warnings.append(
                    _warning(
                        track.name,
                        "ALIGNMENT_WARNING",
                        f"offset_samples={offset_samples} threshold={detection['threshold_samples']}; "
                        "detected offset applied",
                    )
                )
            offset_output = round_half_even(
                Fraction(offset_samples * config.sample_rate, header.sample_rate)
            )
            resampled, resample_stats = resample_polyphase(
                source,
                header.sample_rate,
                config.sample_rate,
                half_width=config.resampler_half_width,
            )
            layout_data, downmix_applied = channel_layout(resampled, config.channels)
            aligned = _align_in_output_domain(layout_data, offset_output)
            prepared.append(
                {
                    "track": track,
                    "header": header,
                    "source_frames": source_frames,
                    "source_peak_dbfs": peak_dbfs(source),
                    "resampled_samples": int(resampled.shape[0]),
                    "resample_stats": resample_stats,
                    "aligned": aligned,
                    "offset_samples": offset_samples,
                    "offset_samples_output": offset_output,
                    "detection": detection,
                    "downmix_applied": downmix_applied,
                }
            )
        except (WavError, OSError, ValueError) as exc:
            errors.append(
                {
                    "track": track.name,
                    "code": "SOURCE_READ_ERROR",
                    "detail": str(exc),
                }
            )
            failed_tracks.add(track.name)
            del warnings[warnings_before:]

    errors = sorted(errors, key=lambda item: (item["track"], item["code"], item["detail"]))
    if not prepared:
        report = {"errors": errors}
        write_report(report_path, report)
        return RenderResult(1, report, output_path, report_path)

    if project.timeline_seconds is not None:
        timeline_frames = round_half_even(
            Fraction(str(project.timeline_seconds)) * config.sample_rate
        )
        timeline_source = "manifest.timeline_seconds"
    else:
        timeline_frames = max(int(item["aligned"].shape[0]) for item in prepared)
        timeline_source = "maximum_aligned_track_length"
    if timeline_frames <= 0:
        errors.append(
            {
                "track": "__timeline__",
                "code": "EMPTY_TIMELINE",
                "detail": "all aligned tracks have zero frames",
            }
        )
        report = {"errors": sorted(errors, key=lambda item: (item["track"], item["code"], item["detail"]))}
        write_report(report_path, report)
        return RenderResult(1, report, output_path, report_path)

    mix = np.zeros((timeline_frames, config.channels), dtype=np.float64)
    track_reports: list[dict[str, Any]] = []
    for item in prepared:
        track: Track = item["track"]
        header = item["header"]
        aligned = _fit_timeline(item["aligned"], timeline_frames)
        gain_linear, gain_stats = build_gain_curve(
            track.automation,
            config.sample_rate,
            timeline_frames,
            config.smooth_window_ms,
        )
        if "SMOOTH_WINDOW_CLAMPED" in gain_stats["smooth_warnings"]:
            warnings.append(
                _warning(
                    track.name,
                    "SMOOTH_WINDOW_CLAMPED",
                    f"requested_ms={config.smooth_window_ms:g}; "
                    f"effective_ms={gain_stats['smooth_window_effective_ms']:.6f}",
                )
            )
        processed = aligned * gain_linear[:, None]
        mix += processed
        detection = item["detection"]
        track_report = {
            "name": track.name,
            "source_file": os.path.relpath(track.source_path, project.root).replace(os.sep, "/"),
            "input_sample_rate": int(header.sample_rate),
            "input_sr": int(header.sample_rate),
            "output_sr": int(config.sample_rate),
            "input_bit_depth": int(header.bit_depth),
            "input_channels": int(header.channels),
            "output_channels": int(config.channels),
            "source_frames": int(item["source_frames"]),
            "resampled": bool(item["resample_stats"]["resampled"]),
            "rational_ratio": item["resample_stats"]["rational_ratio"],
            "rational_approximation": bool(item["resample_stats"]["rational_approximation"]),
            "filter_taps": int(item["resample_stats"]["filter_taps"]),
            "filter_half_width": int(item["resample_stats"]["filter_half_width"]),
            "resampled_samples": int(item["resampled_samples"]),
            "aligned_samples": int(item["aligned"].shape[0]),
            "timeline_samples": int(timeline_frames),
            "downmix_applied": bool(item["downmix_applied"]),
            "downmix_rule": "stereo_to_mono_-3db_power" if item["downmix_applied"] else None,
            "offset_samples": int(item["offset_samples"]),
            "offset_samples_output": int(item["offset_samples_output"]),
            "sync_detected": bool(detection["detected"]),
            "sync_actual_position": detection["actual_position"],
            "sync_expected_position": int(
                _expected_marker_position(track.sync or {})
            ),
            "sync_threshold_samples": int(detection["threshold_samples"]),
            "sync_correlation": float(detection["correlation"]),
            "gain_automation": {
                "node_count": len(track.automation),
                "smooth_steps": int(gain_stats["smooth_steps"]),
                "smooth_window_requested_ms": float(gain_stats["smooth_window_requested_ms"]),
                "smooth_window_effective_ms": float(gain_stats["smooth_window_effective_ms"]),
                "smooth_steps_detail": gain_stats["smooth_steps_detail"],
                "min_gain_db": float(gain_stats["min_gain_db"]),
                "max_gain_db": float(gain_stats["max_gain_db"]),
                "mean_gain_db": float(gain_stats["mean_gain_db"]),
                "rms_gain_db": float(gain_stats["rms_gain_db"]),
                "time_to_sample_method": gain_stats["time_to_sample_method"],
            },
            "smooth_steps": int(gain_stats["smooth_steps"]),
            "gain_applied": {
                "min_db": float(gain_stats["min_gain_db"]),
                "max_db": float(gain_stats["max_gain_db"]),
                "mean_db": float(gain_stats["mean_gain_db"]),
                "rms_db": float(gain_stats["rms_gain_db"]),
            },
            "gain_applied_db": float(gain_stats["mean_gain_db"]),
            "stage_stats": {
                "source_peak_dbfs": float(item["source_peak_dbfs"]),
                "pre_gain_peak_dbfs": peak_dbfs(aligned),
                "post_gain_peak_dbfs": peak_dbfs(processed),
                "post_gain_rms_dbfs": rms_dbfs(processed),
            },
        }
        track_reports.append(track_report)

    mix_stats = _stage_stats(mix)
    limit_linear = config.peak_limit_linear
    lookahead_samples = limiter_lookahead_samples(config.limiter_lookahead_ms, config.sample_rate)
    normalization_enabled = config.loudness_target is not None
    normalization_gain_db = 0.0
    normalization_stats: dict[str, Any]
    if normalization_enabled:
        current_loudness = integrated_loudness(mix, config.sample_rate)
        if current_loudness is None:
            errors.append(
                {
                    "track": "__loudness__",
                    "code": "LOUDNESS_UNMEASURABLE",
                    "detail": "mix is shorter than one 400 ms gating block",
                }
            )
            report = {"errors": sorted(errors, key=lambda item: (item["track"], item["code"], item["detail"]))}
            write_report(report_path, report)
            return RenderResult(1, report, output_path, report_path)
        delta = float(config.loudness_target - current_loudness)
        if abs(delta) < 1e-9:
            delta = 0.0
        normalization_gain_db = delta
        gain = db_to_linear(delta)
        normalized = mix * gain if gain != 1.0 else mix.copy()
        normalized_peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
        peak_conflict = normalization_gain_db > 0.0 and normalized_peak > limit_linear
        if peak_conflict:
            warnings.append(
                _warning(
                    "mix",
                    "NORMALIZATION_PEAK_CONFLICT",
                    f"normalized_peak={linear_to_db(normalized_peak):.6f} dBFS; "
                    f"peak_limit={config.peak_limit_dbfs:.6f} dBFS",
                )
            )
        limited, limiter_stats = apply_lookahead_limiter(
            normalized, limit_linear, lookahead_samples
        )
        normalization_stats = {
            "enabled": True,
            "target_lufs": float(config.loudness_target),
            "input_loudness_lufs": float(current_loudness),
            "gain_db": float(normalization_gain_db),
            "peak_conflict": bool(peak_conflict),
            "post_normalization_peak_dbfs": linear_to_db(normalized_peak),
        }
    else:
        limited, limiter_stats = apply_lookahead_limiter(mix, limit_linear, lookahead_samples)
        normalization_stats = {
            "enabled": False,
            "target_lufs": None,
            "input_loudness_lufs": None,
            "gain_db": 0.0,
            "peak_conflict": False,
            "post_normalization_peak_dbfs": None,
        }
    final_loudness = integrated_loudness(limited, config.sample_rate)
    if (
        normalization_enabled
        and final_loudness is not None
        and abs(float(config.loudness_target) - final_loudness) > 0.5
        and not normalization_stats["peak_conflict"]
    ):
        warnings.append(
            _warning(
                "mix",
                "LOUDNESS_TARGET_MISSED",
                f"target={config.loudness_target:.6f} actual={final_loudness:.6f} tolerance=0.5 LU",
            )
        )
    write_info = write_wav(output_path, limited, config.sample_rate, config.encoding)
    written_header = read_wav_header(output_path)
    warnings = sorted(warnings, key=lambda item: (item["track"], item["code"], item["detail"]))
    report: dict[str, Any] = {
        "schema_version": 1,
        "project": os.path.basename(project.root),
        "output": {
            "path": os.path.relpath(output_path, project.root).replace(os.sep, "/"),
            "sample_rate": int(config.sample_rate),
            "channels": int(written_header.channels),
            "encoding": str(config.encoding),
            "bit_depth": int(written_header.bit_depth),
            "samples": int(written_header.frame_count),
            "bytes": int(write_info["bytes"]),
            "integer_rounding": "round-half-to-even",
        },
        "timeline": {
            "samples": int(timeline_frames),
            "sample_rate": int(config.sample_rate),
            "seconds": float(Fraction(timeline_frames, config.sample_rate)),
            "source": timeline_source,
            "time_to_sample_method": "round-half-to-even",
            "mix_accumulation_order": "track_name_ascending",
        },
        "resampler": {
            "type": "rational_polyphase_fir",
            "window": "blackman-harris",
            "half_width_samples": int(config.resampler_half_width),
            "float_precision": "float64",
        },
        "tracks": track_reports,
        "stages": {
            "mix": mix_stats,
            "normalized": {
                "enabled": bool(normalization_enabled),
                "gain_db": float(normalization_gain_db),
                "peak_dbfs": normalization_stats["post_normalization_peak_dbfs"],
            },
            "limited": _stage_stats(limited),
        },
        "peak_limiter": {
            "peak_limit_dbfs": float(config.peak_limit_dbfs),
            "lookahead_ms": float(config.limiter_lookahead_ms),
            "lookahead_samples": int(lookahead_samples),
            "delay_compensation_samples": int(limiter_stats["delay_compensation_samples"]),
            "bypassed": bool(limiter_stats["bypassed"]),
            "pre_peak_dbfs": linear_to_db(float(limiter_stats["pre_peak_linear"])),
            "post_peak_dbfs": linear_to_db(float(limiter_stats["post_peak_linear"])),
            "limited_samples": int(limiter_stats["limited_samples"]),
            "clipping_events": int(limiter_stats["limited_samples"]),
            "max_attenuation_db": float(limiter_stats["max_attenuation_db"]),
        },
        "normalization_gain_db": float(normalization_gain_db),
        "integrated_loudness_lufs": final_loudness,
        "clipping_events": int(limiter_stats["limited_samples"]),
        "limited_samples": int(limiter_stats["limited_samples"]),
        "max_attenuation_db": float(limiter_stats["max_attenuation_db"]),
        "pre_limiter_peak_dbfs": linear_to_db(float(limiter_stats["pre_peak_linear"])),
        "post_limiter_peak_dbfs": linear_to_db(float(limiter_stats["post_peak_linear"])),
        "loudness_normalization": normalization_stats,
        "warnings": warnings,
        "errors": errors,
        "determinism": {
            "float_precision": "float64",
            "report_float_decimals": 6,
            "track_order": "track_name_ascending",
            "manifest_track_order_affects_mix": False,
            "timestamps_included": False,
            "randomness_used": False,
        },
    }
    write_report(report_path, report)
    exit_code = 2 if errors else 0
    return RenderResult(exit_code, report, output_path, report_path)
