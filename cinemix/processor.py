from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import FadeSpec, ProjectConfig, TrackSpec, load_config
from .dsp import (
    downmix_to_mono,
    integrated_loudness,
    interval_envelope,
    pan_track,
    resample_signal,
    sample_peak_dbfs,
    true_peak,
)
from .errors import CinemixError, Diagnostics
from .limiter import LimiterResult, apply_limiter
from .wavio import quantize_pcm24, read_pcm_wav, wav_sha256, write_pcm_wav


@dataclass
class TrackRender:
    spec: TrackSpec
    source_rate: int
    source_bit_depth: int
    source_channels: int
    source_duration: float
    output_rate: int
    resampled: bool
    audio: np.ndarray
    envelope: np.ndarray
    offset: int
    gain_db: float
    stem_lufs_before_adaptive: float = float("-inf")
    stem_lufs_after_adaptive: float = float("-inf")
    fade_intervals: tuple[tuple[str, float, float, str], ...] = ()


@dataclass
class ProcessingResult:
    output_path: Path
    report: dict[str, Any]
    diagnostics: Diagnostics


def _db_to_linear(value_db: float) -> float:
    return 10.0 ** (value_db / 20.0)


def _finite_db(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return round(float(value), 6)


def _resolve_fade(
    fade: FadeSpec | None,
    track_start: float,
    track_duration: float,
    direction: str,
    track_id: str,
) -> tuple[float, float] | None:
    if fade is None:
        return None
    if fade.duration is not None and fade.duration <= 0.0 and fade.start is None and fade.end is None:
        return None
    start = fade.start
    end = fade.end
    duration = fade.duration
    if duration is not None and duration <= 0.0:
        return None
    if start is None and end is None:
        if duration is None:
            return None
        if direction == "in":
            start = track_start
            end = track_start + duration
        else:
            end = track_start + track_duration
            start = end - duration
    elif start is None and end is not None:
        if direction == "in":
            start = track_start
        elif duration is not None:
            start = end - duration
        else:
            raise CinemixError(
                "E_CONFIG_FIELD",
                f"Track {track_id} fade_out needs start or duration when end is supplied",
                context={"track": track_id, "fade": "out"},
            )
    elif start is not None and end is None:
        if duration is not None:
            end = start + duration
        elif direction == "out":
            end = track_start + track_duration
        else:
            raise CinemixError(
                "E_CONFIG_FIELD",
                f"Track {track_id} fade_in needs end or duration when start is supplied",
                context={"track": track_id, "fade": "in"},
            )
    assert start is not None and end is not None
    if start < 0.0:
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"Track {track_id} fade starts before zero",
            context={"track": track_id, "start": start},
        )
    if end <= start:
        raise CinemixError(
            "E_CONFIG_FIELD",
            f"Track {track_id} fade end must be greater than start",
            context={"track": track_id, "start": start, "end": end},
        )
    return float(start), float(end)


def _ranges_overlap(
    left: tuple[float, float], right: tuple[float, float], epsilon: float = 1e-9
) -> bool:
    return min(left[1], right[1]) - max(left[0], right[0]) > epsilon


def _check_interval_conflicts(renders: list[TrackRender], config: ProjectConfig) -> None:
    for render in renders:
        own = [(name, start, end) for name, start, end, _ in render.fade_intervals]
        for left_index, left in enumerate(own):
            for right in own[left_index + 1 :]:
                if _ranges_overlap((left[1], left[2]), (right[1], right[2])):
                    raise CinemixError(
                        "E_CROSSFADE_CONFLICT",
                        f"Fade intervals overlap on track {render.spec.id}",
                        context={
                            "track": render.spec.id,
                            "fades": [left[0], right[0]],
                            "intervals": [[left[1], left[2]], [right[1], right[2]]],
                        },
                    )
    for index, crossfade in enumerate(config.crossfades):
        for track_id in (crossfade.from_track, crossfade.to_track):
            render = next(item for item in renders if item.spec.id == track_id)
            for name, start, end, _ in render.fade_intervals:
                if name.startswith("track_") and _ranges_overlap(
                    (crossfade.start, crossfade.end), (start, end)
                ):
                    raise CinemixError(
                        "E_CROSSFADE_CONFLICT",
                        f"Crossfade {index} overlaps the {name} of track {track_id}",
                        context={
                            "crossfade": index,
                            "track": track_id,
                            "fade": name,
                        },
                    )


def _build_envelope(
    render_spec: TrackSpec,
    frames: int,
    sample_rate: int,
    own_fades: tuple[tuple[str, float, float, str], ...],
    config: ProjectConfig,
) -> tuple[np.ndarray, tuple[tuple[str, float, float, str], ...]]:
    envelope = np.ones(frames, dtype=np.float64)
    intervals: list[tuple[str, float, float, str]] = []
    for name, start, end, curve in own_fades:
        direction = "in" if name == "fade_in" else "out"
        envelope *= interval_envelope(
            frames,
            sample_rate,
            render_spec.start,
            start,
            end,
            curve,
            direction,
        )
        intervals.append((f"track_{name}", start, end, curve))

    for crossfade_index, crossfade in enumerate(config.crossfades):
        if crossfade.from_track == render_spec.id:
            envelope *= interval_envelope(
                frames,
                sample_rate,
                render_spec.start,
                crossfade.start,
                crossfade.end,
                crossfade.curve,
                "out",
            )
            intervals.append(
                (
                    f"crossfade_{crossfade_index}_out",
                    crossfade.start,
                    crossfade.end,
                    crossfade.curve,
                )
            )
        if crossfade.to_track == render_spec.id:
            envelope *= interval_envelope(
                frames,
                sample_rate,
                render_spec.start,
                crossfade.start,
                crossfade.end,
                crossfade.curve,
                "in",
            )
            intervals.append(
                (
                    f"crossfade_{crossfade_index}_in",
                    crossfade.start,
                    crossfade.end,
                    crossfade.curve,
                )
            )
    return envelope, tuple(intervals)


def _render_tracks(
    config: ProjectConfig,
    diagnostics: Diagnostics,
    output_rate: int,
) -> list[TrackRender]:
    renders: list[TrackRender] = []
    for track in config.tracks:
        samples, info = read_pcm_wav(track.path)
        if info.frames == 0:
            raise CinemixError(
                "E_INPUT_FORMAT",
                f"Input WAV contains no audio frames: {track.path}",
                context={"track": track.id, "path": str(track.path)},
            )
        resampled = False
        if info.sample_rate != output_rate:
            explicit = track.resample_to
            allowed = explicit == output_rate or (
                explicit is None and config.resample_mismatched
            )
            if not allowed:
                if explicit is not None and explicit != output_rate:
                    detail = (
                        f"track resample_to={explicit} Hz conflicts with project "
                        f"sample_rate={output_rate} Hz"
                    )
                else:
                    detail = (
                        f"track is {info.sample_rate} Hz but project is {output_rate} Hz; "
                        "set resample_to or project.resample=true"
                    )
                raise CinemixError(
                    "E_INPUT_SAMPLE_RATE",
                    f"Sample-rate mismatch for track {track.id}: {detail}",
                    context={
                        "track": track.id,
                        "track_sample_rate": info.sample_rate,
                        "project_sample_rate": output_rate,
                    },
                )
            samples = resample_signal(samples, info.sample_rate, output_rate)
            resampled = True
            diagnostics.info(
                "I_TRACK_RESAMPLED",
                f"Track {track.id} resampled from {info.sample_rate} Hz to {output_rate} Hz",
                track=track.id,
                from_sample_rate=info.sample_rate,
                to_sample_rate=output_rate,
            )
        elif track.resample_to is not None and track.resample_to != output_rate:
            raise CinemixError(
                "E_CONFIG_FIELD",
                f"Track {track.id} resample_to conflicts with output sample rate",
                context={
                    "track": track.id,
                    "resample_to": track.resample_to,
                    "output_sample_rate": output_rate,
                },
            )

        duration = samples.shape[0] / output_rate
        fade_in = _resolve_fade(
            track.fade_in, track.start, duration, "in", track.id
        )
        fade_out = _resolve_fade(
            track.fade_out, track.start, duration, "out", track.id
        )
        own_fades: list[tuple[str, float, float, str]] = []
        if fade_in is not None:
            own_fades.append(("fade_in", fade_in[0], fade_in[1], track.fade_in.curve))
        if fade_out is not None:
            own_fades.append(
                ("fade_out", fade_out[0], fade_out[1], track.fade_out.curve)
            )
        envelope, intervals = _build_envelope(
            track,
            samples.shape[0],
            output_rate,
            tuple(own_fades),
            config,
        )
        audio = pan_track(samples, track.pan)
        renders.append(
            TrackRender(
                spec=track,
                source_rate=info.sample_rate,
                source_bit_depth=info.bit_depth,
                source_channels=info.channels,
                source_duration=info.duration,
                output_rate=output_rate,
                resampled=resampled,
                audio=audio,
                envelope=envelope,
                offset=round(track.start * output_rate),
                gain_db=track.gain_db,
                fade_intervals=intervals,
            )
        )
    _check_interval_conflicts(renders, config)
    return renders


def _apply_adaptive_gain(
    renders: list[TrackRender],
    config: ProjectConfig,
    diagnostics: Diagnostics,
) -> float:
    if not config.adaptive_gain.enabled:
        return 0.0
    cuts: list[float] = []
    for render in renders:
        stem = (
            render.audio
            * render.envelope[:, None]
            * _db_to_linear(render.gain_db)
        )
        measurement = integrated_loudness(stem, render.output_rate)
        render.stem_lufs_before_adaptive = measurement.lufs
        if math.isfinite(measurement.lufs):
            excess = measurement.lufs - config.adaptive_gain.threshold_lufs
            if excess > 0.0:
                cuts.append(float(excess))
    if not cuts:
        for render in renders:
            render.stem_lufs_after_adaptive = render.stem_lufs_before_adaptive
        return 0.0
    trim_db = -min(max(cuts), config.adaptive_gain.max_cut_db)
    for render in renders:
        render.gain_db += trim_db
        render.stem_lufs_after_adaptive = (
            render.stem_lufs_before_adaptive + trim_db
        )
    affected = [
        render.spec.id
        for render in renders
        if math.isfinite(render.stem_lufs_before_adaptive)
        and render.stem_lufs_before_adaptive > config.adaptive_gain.threshold_lufs
    ]
    diagnostics.info(
        "I_ADAPTIVE_GAIN_APPLIED",
        f"Adaptive common trim {trim_db:.3f} dB applied to preserve inter-track relativity",
        trim_db=trim_db,
        triggering_tracks=affected,
        threshold_lufs=config.adaptive_gain.threshold_lufs,
    )
    return trim_db


def _mix(renders: list[TrackRender], total_frames: int) -> np.ndarray:
    mix = np.zeros((total_frames, 2), dtype=np.float64)
    for render in renders:
        usable = min(render.audio.shape[0], total_frames - render.offset)
        if usable <= 0:
            continue
        gain = _db_to_linear(render.gain_db)
        segment = (
            render.audio[:usable, :]
            * render.envelope[:usable, None]
            * gain
        )
        mix[render.offset : render.offset + usable, :] += segment
    return mix


def _measure_iteratively(
    mix: np.ndarray,
    sample_rate: int,
    config: ProjectConfig,
    diagnostics: Diagnostics,
) -> tuple[np.ndarray, LimiterResult, dict[str, Any]]:
    target = config.target_lufs
    program_gain_db = 0.0
    best: tuple[float, np.ndarray, LimiterResult, float, float, float] | None = None
    last_loudness = float("-inf")
    last_true_peak = float("-inf")
    iterations = 0
    stagnant = 0
    for iteration in range(20):
        iterations = iteration + 1
        driven = mix * _db_to_linear(program_gain_db)
        limited = apply_limiter(
            driven,
            sample_rate,
            config.true_peak_dbtp,
            config.limiter,
        )
        loudness = integrated_loudness(limited.samples, sample_rate).lufs
        peak = true_peak(limited.samples, sample_rate).dbtp
        last_loudness = loudness
        last_true_peak = peak
        if math.isfinite(loudness):
            error = target - loudness
        else:
            error = float("inf")
        candidate = (
            abs(error),
            limited.samples,
            limited,
            loudness,
            peak,
            program_gain_db,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
        if abs(error) <= 0.05:
            break
        if not math.isfinite(error):
            break
        if iteration >= 10 and best is not None:
            improvement = abs(best[3] - last_loudness)
            stagnant = stagnant + 1 if improvement < 0.002 else 0
            if stagnant >= 2 and abs(error) > 0.1:
                break
        step = error * (0.80 if error > 0.0 else 1.0)
        step = max(-12.0, min(12.0, step))
        program_gain_db += step

    if best is None:
        raise CinemixError("E_INPUT_SILENT", "Unable to measure a non-silent mix")
    (
        best_error,
        best_samples,
        best_limiter,
        best_loudness,
        best_peak,
        best_program_gain_db,
    ) = best
    if best_error > 0.5:
        raise CinemixError(
            "E_LIMIT_UNREACHABLE",
            (
                f"Could not reach target {target:.2f} LUFS within 0.5 LU while "
                f"respecting {config.true_peak_dbtp:.2f} dBTP; best was "
                f"{best_loudness:.3f} LUFS"
            ),
            context={
                "target_lufs": target,
                "measured_lufs": _finite_db(best_loudness),
                "true_peak_dbtp": _finite_db(best_peak),
                "true_peak_limit_dbtp": config.true_peak_dbtp,
            },
        )
    if best_error > 0.2:
        diagnostics.warning(
            "W_LOUDNESS_TOLERANCE_MARGIN",
            f"Loudness converged to {best_loudness:.3f} LUFS ({best_error:.3f} LU error)",
            target_lufs=target,
            measured_lufs=best_loudness,
            error_lu=best_error,
        )
    return best_samples, best_limiter, {
        "iterations": iterations,
        "program_gain_db": round(best_program_gain_db, 6),
        "last_loudness_lufs": _finite_db(last_loudness),
        "last_true_peak_dbtp": _finite_db(last_true_peak),
        "converged_error_lu": round(best_error, 6),
    }


def _quantized_verification(
    samples: np.ndarray,
    sample_rate: int,
    config: ProjectConfig,
) -> tuple[np.ndarray, float, float]:
    working = samples
    for _ in range(3):
        quantized = quantize_pcm24(working)
        decoded = quantized.astype(np.float64) / 8388608.0
        peak = true_peak(decoded, sample_rate).dbtp
        loudness = integrated_loudness(decoded, sample_rate).lufs
        if peak <= config.true_peak_dbtp:
            return decoded, loudness, peak
        trim_db = min(-0.001, config.true_peak_dbtp - peak - 0.005)
        working = working * _db_to_linear(trim_db)
    quantized = quantize_pcm24(working)
    decoded = quantized.astype(np.float64) / 8388608.0
    return (
        decoded,
        integrated_loudness(decoded, sample_rate).lufs,
        true_peak(decoded, sample_rate).dbtp,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def process_project(
    config_path: str | Path,
    *,
    output_override: str | Path | None = None,
    diagnostics: Diagnostics | None = None,
) -> ProcessingResult:
    diagnostics = diagnostics or Diagnostics()
    config = load_config(config_path, diagnostics)
    output_path = (
        Path(output_override).resolve()
        if output_override is not None
        else config.output
    )
    if output_path is None:
        raise CinemixError(
            "E_CONFIG_FIELD",
            "No output path: set project.output or pass --output",
            context={"field": "project.output"},
        )

    first_track_samples, first_info = read_pcm_wav(config.tracks[0].path)
    del first_track_samples
    output_rate = config.sample_rate or first_info.sample_rate
    renders = _render_tracks(config, diagnostics, output_rate)
    adaptive_trim_db = _apply_adaptive_gain(renders, config, diagnostics)

    end_frame = 0
    for render in renders:
        end_frame = max(
            end_frame, render.offset + render.audio.shape[0]
        )
    if config.crossfades:
        end_frame = max(
            end_frame,
            max(round(item.end * output_rate) for item in config.crossfades),
        )
    if config.duration_seconds is not None:
        end_frame = max(
            end_frame, round(config.duration_seconds * output_rate)
        )
    end_frame = max(1, end_frame)

    mix = _mix(renders, end_frame)
    if config.output_channels == 1:
        mix = downmix_to_mono(mix)[:, None]
    mix_level = sample_peak_dbfs(mix)
    if not math.isfinite(mix_level) or mix_level < -240.0:
        raise CinemixError(
            "E_INPUT_SILENT",
            "Track mix is silent after fades and gains",
            context={"sample_peak_dbfs": _finite_db(mix_level)},
        )

    limited_samples, limiter_result, convergence = _measure_iteratively(
        mix, output_rate, config, diagnostics
    )
    final_samples, final_loudness, final_peak = _quantized_verification(
        limited_samples, output_rate, config
    )
    loudness_error = final_loudness - config.target_lufs
    if abs(loudness_error) > 0.5:
        raise CinemixError(
            "E_LIMIT_UNREACHABLE",
            (
                f"Quantized output loudness {final_loudness:.3f} LUFS missed "
                f"target {config.target_lufs:.3f} LUFS by more than 0.5 LU"
            ),
            context={
                "target_lufs": config.target_lufs,
                "measured_lufs": final_loudness,
                "true_peak_dbtp": final_peak,
            },
        )
    if final_peak > config.true_peak_dbtp:
        raise CinemixError(
            "E_LIMIT_UNREACHABLE",
            (
                f"Quantized true peak {final_peak:.3f} dBTP exceeds "
                f"{config.true_peak_dbtp:.3f} dBTP"
            ),
            context={
                "true_peak_limit_dbtp": config.true_peak_dbtp,
                "measured_true_peak_dbtp": final_peak,
            },
        )

    try:
        write_pcm_wav(output_path, final_samples, output_rate)
    except OSError as exc:
        raise CinemixError(
            "E_OUTPUT_WRITE",
            f"Cannot write output WAV {output_path}: {exc}",
            context={"path": str(output_path)},
        ) from exc

    crossfade_report = []
    for crossfade in config.crossfades:
        actual_start = round(crossfade.start * output_rate)
        actual_end = round(crossfade.end * output_rate)
        start_error_ms = abs(actual_start / output_rate - crossfade.start) * 1000.0
        end_error_ms = abs(actual_end / output_rate - crossfade.end) * 1000.0
        crossfade_report.append(
            {
                "from": crossfade.from_track,
                "to": crossfade.to_track,
                "curve": crossfade.curve,
                "configured_start_seconds": crossfade.start,
                "configured_end_seconds": crossfade.end,
                "actual_start_sample": actual_start,
                "actual_end_sample": actual_end,
                "start_error_ms": round(start_error_ms, 9),
                "end_error_ms": round(end_error_ms, 9),
            }
        )
    maximum_boundary_error_ms = max(
        (
            max(item["start_error_ms"], item["end_error_ms"])
            for item in crossfade_report
        ),
        default=0.0,
    )
    if maximum_boundary_error_ms > 1.0:
        raise CinemixError(
            "E_CROSSFADE_CONFLICT",
            "Crossfade boundary quantization exceeded 1 ms",
            context={"maximum_boundary_error_ms": maximum_boundary_error_ms},
        )

    track_report = []
    for render in renders:
        track_report.append(
            {
                "id": render.spec.id,
                "path": str(render.spec.path),
                "source_sample_rate": render.source_rate,
                "source_bit_depth": render.source_bit_depth,
                "source_channels": render.source_channels,
                "resampled": render.resampled,
                "configured_gain_db": render.spec.gain_db,
                "adaptive_trim_applied_db": round(adaptive_trim_db, 6)
                if render.spec.id
                else 0.0,
                "applied_gain_db": round(render.gain_db, 6),
                "pan": render.spec.pan,
                "start_seconds": render.spec.start,
                "stem_lufs_before_adaptive": _finite_db(
                    render.stem_lufs_before_adaptive
                ),
                "stem_lufs_after_adaptive": _finite_db(
                    render.stem_lufs_after_adaptive
                ),
            }
        )

    report: dict[str, Any] = {
        "ok": True,
        "output": str(output_path),
        "output_sha256": wav_sha256(output_path),
        "format": {
            "sample_rate": output_rate,
            "channels": config.output_channels,
            "bit_depth": 24,
            "encoding": "PCM",
        },
        "duration_seconds": round(final_samples.shape[0] / output_rate, 9),
        "target": {
            "integrated_lufs": config.target_lufs,
            "true_peak_dbtp": config.true_peak_dbtp,
        },
        "measurement": {
            "integrated_lufs": round(final_loudness, 6),
            "true_peak_dbtp": round(final_peak, 6),
            "loudness_error_lu": round(loudness_error, 6),
            "true_peak_margin_db": round(
                config.true_peak_dbtp - final_peak, 6
            ),
        },
        "tracks": track_report,
        "crossfades": crossfade_report,
        "maximum_crossfade_boundary_error_ms": round(
            maximum_boundary_error_ms, 9
        ),
        "limiter": {
            "attack_ms": config.limiter.attack_ms,
            "release_ms": config.limiter.release_ms,
            "ratio": config.limiter.ratio,
            "lookahead_ms": config.limiter.lookahead_ms,
            "compression_threshold_dbtp": config.limiter.compression_threshold_dbtp,
            "safety_margin_db": config.limiter.safety_margin_db,
            "passes": limiter_result.passes,
            "sample_ceiling_dbtp": round(
                limiter_result.sample_ceiling_dbtp, 6
            ),
            "max_reduction_db": round(limiter_result.max_reduction_db, 6),
        },
        "convergence": convergence,
        "diagnostics": diagnostics.as_list(),
    }
    return ProcessingResult(
        output_path=output_path,
        report=report,
        diagnostics=diagnostics,
    )


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    try:
        _write_json_atomic(Path(path).resolve(), report)
    except OSError as exc:
        raise CinemixError(
            "E_OUTPUT_WRITE",
            f"Cannot write report {path}: {exc}",
            context={"path": str(path)},
        ) from exc
