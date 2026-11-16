"""End-to-end mixing engine with per-track and per-segment isolation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from . import __version__
from .audio import (
    AudioProbe,
    check_toolchain,
    decode_audio,
    probe_audio,
    write_json_atomic,
    write_wav_24,
)
from .dsp import (
    ActivityResult,
    DenoiseResult,
    DuckCurve,
    LoudnessResult,
    apply_channel_matrix,
    amp_to_db,
    apply_track_gain,
    concatenate_intervals,
    denoise_dialogue,
    detect_activity,
    duck_gain_curve,
    guard_output_peak,
    integrated_loudness,
    normalize_dialogue_loudness,
    peak_dbfs,
    remove_dc,
    short_term_deviation,
)
from .errors import OutputError
from .manifest import (
    MixManifest,
    SegmentSpec,
    TrackSpec,
    load_manifest,
    safe_filename,
)


@dataclass
class TrackRuntime:
    spec: TrackSpec
    status: str = "pending"
    skip_reason: str | None = None
    error: str | None = None
    probe: AudioProbe | None = None
    conversion: dict[str, Any] = field(default_factory=dict)
    source_signal: np.ndarray | None = None
    signal: np.ndarray | None = None
    denoise: DenoiseResult | None = None
    activity: ActivityResult | None = None
    loudness: LoudnessResult | None = None
    duck_curve: DuckCurve | None = None

    @property
    def readable(self) -> bool:
        return self.probe is not None

    @property
    def processed(self) -> bool:
        return self.status == "processed" and self.signal is not None


@dataclass(frozen=True)
class MixResult:
    exit_code: int
    metadata_path: Path | None
    output_files: tuple[Path, ...]
    metadata: dict[str, Any]
    failure_message: str | None = None


def run_mix(
    manifest_path: str | Path,
    output_dir: str | Path,
) -> MixResult:
    manifest = load_manifest(manifest_path)
    ffmpeg, ffprobe = check_toolchain()
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.exists() and not output_root.is_dir():
        raise OutputError("MIX3001", f"output path is not a directory: {output_root}")

    runtimes = [
        _load_track(
            spec,
            output_channels=manifest.output_channels,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )
        for spec in manifest.tracks
    ]
    loaded = [runtime for runtime in runtimes if runtime.processed]

    if not loaded:
        output_root.mkdir(parents=True, exist_ok=True)
        metadata = _base_metadata(manifest, output_root, runtimes)
        message = "all tracks are unreadable or have no applicable channel mapping"
        metadata["summary"].update(
            {
                "status": "failed",
                "exit_code": 1,
                "failure_reason": message,
            }
        )
        metadata["warnings"].append(message)
        metadata_path = output_root / "mix_metadata.json"
        write_json_atomic(metadata_path, _sanitize(metadata))
        return MixResult(
            exit_code=1,
            metadata_path=metadata_path,
            output_files=(),
            metadata=metadata,
            failure_message=message,
        )

    total_samples = _determine_total_samples(manifest, loaded)
    for runtime in loaded:
        assert runtime.signal is not None
        runtime.signal = _fit_length(runtime.signal, total_samples)

    dialogues = [runtime for runtime in loaded if runtime.spec.role == "dialogue"]
    beds = [runtime for runtime in loaded if runtime.spec.role in {"music", "sfx"}]

    for runtime in dialogues:
        _process_dialogue(runtime)

    union_intervals = _union_active_intervals(dialogues)
    for runtime in beds:
        _process_bed(runtime, union_intervals, total_samples)

    processed = [runtime for runtime in runtimes if runtime.processed]
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = _base_metadata(manifest, output_root, runtimes)
    metadata["dialogue_timing"] = {
        "basis": "sample_index_at_48000_hz",
        "union_active_intervals_samples": _intervals_json(union_intervals),
        "union_active_intervals_seconds": _intervals_seconds(
            union_intervals, manifest.raw.get("sample_rate", 48_000)
        ),
        "ducking_alignment": "sample_exact",
    }
    metadata["audio"]["total_samples"] = total_samples
    metadata["audio"]["duration_s"] = total_samples / 48_000.0

    if not processed:
        message = "all readable tracks failed during processing"
        metadata["summary"].update(
            {"status": "failed", "exit_code": 1, "failure_reason": message}
        )
        metadata["warnings"].append(message)
        metadata_path = output_root / "mix_metadata.json"
        write_json_atomic(metadata_path, _sanitize(metadata))
        return MixResult(
            exit_code=1,
            metadata_path=metadata_path,
            output_files=(),
            metadata=metadata,
            failure_message=message,
        )

    mix = np.zeros((total_samples, manifest.output_channels), dtype=np.float64)
    for runtime in processed:
        assert runtime.signal is not None
        mix += runtime.signal[:total_samples]
    guarded_mix, mix_peak_before, mix_guard_db = guard_output_peak(mix)
    metadata["mix"] = {
        "peak_before_guard_dbfs": _finite_or_none(
            amp_to_db(mix_peak_before) if mix_peak_before > 0.0 else None
        ),
        "peak_guard_scale_db": mix_guard_db,
        "final_peak_dbfs": _finite_or_none(peak_dbfs(guarded_mix)),
        "final_integrated_lufs": _finite_or_none(
            integrated_loudness(guarded_mix)
        ),
        "active_track_count": len(processed),
    }

    plans = _segment_plans(manifest, total_samples)
    output_files: list[Path] = []
    segment_entries: list[dict[str, Any]] = []
    used_names: set[str] = set()
    for plan in plans:
        entry = _write_segment(
            plan,
            guarded_mix,
            runtimes,
            manifest,
            output_root,
            ffmpeg,
            ffprobe,
            used_names,
        )
        segment_entries.append(entry)
        output_file = entry.get("output_file")
        if output_file:
            output_files.append(output_root / output_file)

    metadata["segments"] = segment_entries
    succeeded = sum(1 for entry in segment_entries if entry["status"] in {"ok", "partial"})
    failed = len(segment_entries) - succeeded
    metadata["summary"].update(
        {
            "segments_total": len(segment_entries),
            "segments_succeeded": succeeded,
            "segments_failed": failed,
        }
    )

    metadata_path = output_root / "mix_metadata.json"
    if succeeded == 0:
        message = "all output segments failed to render"
        metadata["summary"].update(
            {"status": "failed", "exit_code": 1, "failure_reason": message}
        )
        metadata["warnings"].append(message)
        write_json_atomic(metadata_path, _sanitize(metadata))
        return MixResult(
            exit_code=1,
            metadata_path=metadata_path,
            output_files=tuple(output_files),
            metadata=metadata,
            failure_message=message,
        )

    if failed or any(runtime.status != "processed" for runtime in runtimes):
        status = "partial"
    else:
        status = "ok"
    metadata["summary"].update({"status": status, "exit_code": 0})
    write_json_atomic(metadata_path, _sanitize(metadata))
    return MixResult(
        exit_code=0,
        metadata_path=metadata_path,
        output_files=tuple(output_files),
        metadata=metadata,
    )


def _load_track(
    spec: TrackSpec,
    *,
    output_channels: int,
    ffmpeg: str,
    ffprobe: str | None,
) -> TrackRuntime:
    runtime = TrackRuntime(spec=spec)
    if not spec.path.exists():
        runtime.status = "skipped"
        runtime.skip_reason = f"file_not_found: {spec.path}"
        runtime.conversion = _conversion_metadata(
            None, spec, output_channels, applied=False
        )
        return runtime
    try:
        probe = probe_audio(spec.path, ffprobe=ffprobe)
        runtime.probe = probe
        actual_channels = probe.channels
        if spec.declared_channels is not None and spec.declared_channels != actual_channels:
            runtime.status = "skipped"
            runtime.skip_reason = (
                "declared_input_channels_mismatch: "
                f"manifest={spec.declared_channels}, file={actual_channels}"
            )
            runtime.conversion = _conversion_metadata(
                actual_channels, spec, output_channels, applied=False
            )
            return runtime

        matrix, mode, method = _select_mapping(
            actual_channels, spec, output_channels
        )
        if mode == "unavailable":
            runtime.status = "skipped"
            runtime.skip_reason = (
                "no_matching_channel_mapping: explicit matrix or method does not match "
                f"{actual_channels} input channels -> {output_channels} output channels"
            )
            runtime.conversion = _conversion_metadata(
                actual_channels,
                spec,
                output_channels,
                applied=False,
                mode="unavailable",
                method=spec.channel_map.method,
            )
            return runtime
        runtime.conversion = _conversion_metadata(
            actual_channels,
            spec,
            output_channels,
            applied=True,
            mode=mode,
            method=method,
        )
        source, decoded_probe = decode_audio(spec.path, ffmpeg=ffmpeg, ffprobe=ffprobe)
        runtime.probe = decoded_probe
        if source.shape[1] != actual_channels:
            raise OSError(
                f"decoded channel count changed from {actual_channels} to {source.shape[1]}"
            )
        source = remove_dc(source)
        if matrix is None:
            if actual_channels != output_channels:
                runtime.status = "skipped"
                runtime.skip_reason = (
                    "no_matching_channel_mapping: "
                    f"{actual_channels} input channels -> {output_channels} output channels"
                )
                runtime.conversion = _conversion_metadata(
                    actual_channels, spec, output_channels, applied=False
                )
                return runtime
            converted = source
            runtime.conversion = _conversion_metadata(
                actual_channels,
                spec,
                output_channels,
                applied=True,
                mode="identity",
                method="identity",
            )
        else:
            converted = apply_channel_matrix(source, matrix)
        runtime.source_signal = source
        runtime.signal = converted
        runtime.status = "processed"
        return runtime
    except FileNotFoundError:
        runtime.status = "skipped"
        runtime.skip_reason = f"file_not_found: {spec.path}"
    except OSError as exc:
        runtime.status = "skipped"
        runtime.skip_reason = f"unsupported_or_unreadable_audio: {exc}"
    except Exception as exc:
        runtime.status = "error"
        runtime.error = f"{type(exc).__name__}: {exc}"
    runtime.signal = None
    if not runtime.conversion:
        runtime.conversion = _conversion_metadata(
            runtime.probe.channels if runtime.probe else None,
            spec,
            output_channels,
            applied=False,
        )
    return runtime


def _select_mapping(
    actual_channels: int,
    spec: TrackSpec,
    output_channels: int,
) -> tuple[list[list[float]] | None, str, str | None]:
    declared = spec.channel_map.matrix
    configured_method = spec.channel_map.method
    if declared is None:
        if actual_channels == output_channels:
            return None, "identity", "identity"
        return None, "none", None
    if len(declared[0]) != actual_channels:
        return None, "unavailable", None
    matrix = [list(row) for row in declared]
    if actual_channels == output_channels:
        if configured_method not in (None, "identity", "explicit"):
            return None, "unavailable", None
        return matrix, "identity", configured_method or "identity"
    expected = "upmix" if actual_channels < output_channels else "downmix"
    if configured_method not in (None, expected, "explicit"):
        return None, "unavailable", None
    return matrix, expected, configured_method or expected


def _conversion_metadata(
    actual_channels: int | None,
    spec: TrackSpec,
    output_channels: int,
    *,
    applied: bool,
    mode: str | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    matrix = (
        [list(row) for row in spec.channel_map.matrix]
        if spec.channel_map.matrix is not None
        else None
    )
    return {
        "source_channels": actual_channels,
        "output_channels": output_channels,
        "mode": mode or ("identity" if actual_channels == output_channels else "unavailable"),
        "method": method or spec.channel_map.method,
        "description": spec.channel_map.description,
        "matrix": matrix,
        "applied": bool(applied),
    }


def _determine_total_samples(
    manifest: MixManifest,
    loaded: list[TrackRuntime],
) -> int:
    maximum_track = max(
        (runtime.signal.shape[0] for runtime in loaded if runtime.signal is not None),
        default=0,
    )
    if manifest.duration_s is not None:
        return max(1, int(round(manifest.duration_s * 48_000)))
    maximum_segment = max(
        (
            int(round(segment.end_s * 48_000))
            for segment in manifest.segments
        ),
        default=0,
    )
    return max(1, maximum_track, maximum_segment)


def _fit_length(signal: np.ndarray, samples: int) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float64)
    if array.shape[0] == samples:
        return array
    if array.shape[0] > samples:
        return array[:samples].copy()
    padding = np.zeros((samples - array.shape[0], array.shape[1]), dtype=np.float64)
    return np.concatenate((array, padding), axis=0)


def _process_dialogue(runtime: TrackRuntime) -> None:
    assert runtime.signal is not None
    try:
        base = apply_track_gain(runtime.signal, runtime.spec.gain_db)
        denoised = denoise_dialogue(base, 48_000, runtime.spec.noise_gate)
        activity = detect_activity(denoised.signal, 48_000, runtime.spec.ducking)
        normalized = normalize_dialogue_loudness(
            denoised.signal,
            48_000,
            runtime.spec.target_lufs,
            activity.intervals,
            runtime.spec.loudness,
        )
        runtime.signal = normalized.signal
        runtime.denoise = denoised
        runtime.activity = activity
        runtime.loudness = normalized
        runtime.status = "processed"
    except Exception as exc:
        runtime.status = "error"
        runtime.error = f"dialogue processing failed: {type(exc).__name__}: {exc}"
        runtime.signal = None


def _process_bed(
    runtime: TrackRuntime,
    active_intervals: list[tuple[int, int]],
    total_samples: int,
) -> None:
    assert runtime.signal is not None
    try:
        duck = duck_gain_curve(
            total_samples,
            active_intervals,
            depth_db=runtime.spec.ducking.depth_db,
            attack_ms=runtime.spec.ducking.attack_ms,
            release_ms=runtime.spec.ducking.release_ms,
            sample_rate=48_000,
        )
        gain = np.power(10.0, duck.gain_db / 20.0)[:, None]
        runtime.signal = runtime.signal * gain
        runtime.signal = apply_track_gain(runtime.signal, runtime.spec.gain_db)
        runtime.duck_curve = duck
        runtime.status = "processed"
    except Exception as exc:
        runtime.status = "error"
        runtime.error = f"ducking failed: {type(exc).__name__}: {exc}"
        runtime.signal = None


def _union_active_intervals(
    dialogues: Iterable[TrackRuntime],
) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    for runtime in dialogues:
        if runtime.activity is not None and runtime.status == "processed":
            intervals.extend(runtime.activity.intervals)
    if not intervals:
        return []
    intervals.sort()
    union: list[list[int]] = []
    for start, end in intervals:
        if union and start <= union[-1][1]:
            union[-1][1] = max(union[-1][1], end)
        else:
            union.append([start, end])
    return [(start, end) for start, end in union]


def _segment_plans(
    manifest: MixManifest,
    total_samples: int,
) -> list[SegmentSpec]:
    if manifest.segments:
        return list(manifest.segments)
    return [
        SegmentSpec(
            id="full",
            start_s=0.0,
            end_s=total_samples / 48_000.0,
            index=0,
        )
    ]


def _write_segment(
    segment: SegmentSpec,
    mix: np.ndarray,
    runtimes: list[TrackRuntime],
    manifest: MixManifest,
    output_root: Path,
    ffmpeg: str,
    ffprobe: str | None,
    used_names: set[str],
) -> dict[str, Any]:
    start = int(round(segment.start_s * 48_000))
    end = int(round(segment.end_s * 48_000))
    total = mix.shape[0]
    start = max(0, min(total, start))
    end = max(start, min(total, end))
    entry: dict[str, Any] = {
        "id": segment.id,
        "index": segment.index,
        "start_s": segment.start_s,
        "end_s": segment.end_s,
        "sample_range": {"start": start, "end_exclusive": end},
        "output_sample_range": None,
        "boundary_handoff_samples": [],
        "boundary_method": "none",
        "status": "pending",
        "output_file": None,
        "error": None,
        "dialogue": [],
        "tracks": [_segment_track_summary(runtime) for runtime in runtimes],
    }
    if end <= start:
        entry["status"] = "failed"
        entry["error"] = "segment resolves to zero samples"
        return entry

    # Keep nominal durations exact while handing the same averaged boundary
    # sample to both independently encoded files. This removes a one-sample
    # codec discontinuity without extending either segment.
    output_end = end
    entry["output_sample_range"] = {"start": start, "end_exclusive": output_end}

    filename_base = safe_filename(segment.id)
    filename = f"{filename_base}.wav"
    suffix = 1
    while filename in used_names:
        filename = f"{filename_base}_{suffix:03d}.wav"
        suffix += 1
    used_names.add(filename)
    output_path = output_root / filename
    piece = mix[start:output_end].copy()
    adjacent_starts = [
        int(round(other.end_s * 48_000))
        for other in manifest.segments
        if other.index != segment.index
        and int(round(other.end_s * 48_000)) == start
    ]
    adjacent_ends = [
        int(round(other.start_s * 48_000))
        for other in manifest.segments
        if other.index != segment.index
        and int(round(other.start_s * 48_000)) == end
    ]
    if adjacent_starts and start > 0:
        handoff = 0.5 * (mix[start - 1] + mix[start])
        piece[0] = handoff
        entry["boundary_handoff_samples"].append(start)
    if adjacent_ends and end < total:
        handoff = 0.5 * (mix[end - 1] + mix[end])
        piece[-1] = handoff
        entry["boundary_handoff_samples"].append(end - 1)
    if entry["boundary_handoff_samples"]:
        entry["boundary_method"] = "one_sample_averaged_handoff"
    try:
        write_wav_24(
            output_path,
            piece,
            sample_rate=48_000,
            channels=manifest.output_channels,
            channel_layout=manifest.channel_layout,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )
        entry["output_file"] = filename
        track_partial = any(runtime.status != "processed" for runtime in runtimes)
        entry["status"] = "partial" if track_partial else "ok"
    except Exception as exc:
        entry["status"] = "failed"
        entry["error"] = f"segment output failed: {type(exc).__name__}: {exc}"
        return entry

    entry["audio"] = {
        "channels": manifest.output_channels,
        "channel_layout": manifest.channel_layout,
        "sample_rate": 48_000,
        "samples": int(piece.shape[0]),
        "integrated_lufs": _finite_or_none(integrated_loudness(piece)),
        "peak_dbfs": _finite_or_none(peak_dbfs(piece)),
    }
    for runtime in runtimes:
        if runtime.spec.role != "dialogue" or not runtime.processed:
            continue
        entry["dialogue"].append(
            _dialogue_segment_metadata(runtime, start, end)
        )
    return entry


def _segment_track_summary(runtime: TrackRuntime) -> dict[str, Any]:
    return {
        "id": runtime.spec.id,
        "role": runtime.spec.role,
        "status": runtime.status,
        "skip_reason": runtime.skip_reason,
        "error": runtime.error,
        "conversion": runtime.conversion,
    }


def _dialogue_segment_metadata(
    runtime: TrackRuntime,
    start: int,
    end: int,
) -> dict[str, Any]:
    assert runtime.signal is not None
    segment_signal = runtime.signal[start:end]
    clipped_intervals = _clip_intervals(
        runtime.activity.intervals if runtime.activity is not None else [],
        start,
        end,
    )
    absolute_intervals = [(a + start, b + start) for a, b in clipped_intervals]
    active_signal = concatenate_intervals(segment_signal, clipped_intervals)
    deviation, frame_count = short_term_deviation(
        runtime.signal,
        48_000,
        runtime.spec.target_lufs,
        active_intervals=(
            runtime.activity.intervals if runtime.activity is not None else []
        ),
        segment_range=(start, end),
    )
    interval_measurements = []
    for a, b in absolute_intervals:
        interval_measurements.append(
            {
                "start_sample": a,
                "end_sample": b,
                "integrated_lufs": _finite_or_none(
                    integrated_loudness(runtime.signal[a:b])
                ),
            }
        )
    return {
        "track_id": runtime.spec.id,
        "target_lufs": runtime.spec.target_lufs,
        "segment_integrated_lufs": _finite_or_none(
            integrated_loudness(segment_signal)
        ),
        "active_integrated_lufs": _finite_or_none(
            integrated_loudness(active_signal)
        ),
        "short_term_400ms": {
            "max_abs_deviation_lu": _finite_or_none(deviation),
            "active_frame_count": frame_count,
            "within_tolerance": bool(
                np.isfinite(deviation)
                and deviation <= runtime.spec.loudness.tolerance_lu + 1e-9
            ),
        },
        "active_intervals_in_segment": _intervals_json(absolute_intervals),
        "active_interval_loudness": interval_measurements,
        "conversion": runtime.conversion,
        "static_gain_db": runtime.spec.gain_db,
    }


def _clip_intervals(
    intervals: Iterable[tuple[int, int]],
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    clipped: list[tuple[int, int]] = []
    for interval_start, interval_end in intervals:
        lo = max(start, interval_start)
        hi = min(end, interval_end)
        if hi > lo:
            clipped.append((lo - start, hi - start))
    return clipped


def _base_metadata(
    manifest: MixManifest,
    output_root: Path,
    runtimes: list[TrackRuntime],
) -> dict[str, Any]:
    processed = sum(1 for runtime in runtimes if runtime.status == "processed")
    skipped = sum(1 for runtime in runtimes if runtime.status == "skipped")
    errors = sum(1 for runtime in runtimes if runtime.status == "error")
    unreadable = sum(1 for runtime in runtimes if not runtime.readable)
    return {
        "schema_version": "1.0",
        "tool": {
            "name": "offline-dialogue-mixer",
            "version": __version__,
            "loudness_standard": "ITU-R BS.1770 (K-weighting, absolute/relative gating)",
            "sample_rate": 48_000,
            "pcm": "24-bit PCM",
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest.source_path),
        "output_directory": str(output_root),
        "audio": {
            "sample_rate": 48_000,
            "pcm_bits": 24,
            "channels": manifest.output_channels,
            "channel_layout": manifest.channel_layout,
            "channel_order": (
                ["FL", "FR"]
                if manifest.output_channels == 2
                else ["FL", "FR", "FC", "LFE", "BL", "BR"]
            ),
            "total_samples": None,
            "duration_s": None,
        },
        "parameters": {
            "target_lufs": manifest.target_lufs,
            "noise_gate": asdict(manifest.noise_gate),
            "loudness": asdict(manifest.loudness),
            "ducking": asdict(manifest.ducking),
        },
        "tracks": [_track_metadata(runtime) for runtime in runtimes],
        "segments": [],
        "summary": {
            "tracks_total": len(runtimes),
            "tracks_processed": processed,
            "tracks_skipped": skipped,
            "tracks_errors": errors,
            "tracks_unreadable": unreadable,
            "segments_total": 0,
            "segments_succeeded": 0,
            "segments_failed": 0,
            "status": "processing",
            "exit_code": 0,
            "failure_reason": None,
        },
        "warnings": [],
    }


def _track_metadata(runtime: TrackRuntime) -> dict[str, Any]:
    spec = runtime.spec
    result: dict[str, Any] = {
        "id": spec.id,
        "role": spec.role,
        "path": str(spec.path),
        "status": runtime.status,
        "skip_reason": runtime.skip_reason,
        "error": runtime.error,
        "readable": runtime.readable,
        "declared_input_channels": spec.declared_channels,
        "source": (
            {
                "channels": runtime.probe.channels,
                "sample_rate": runtime.probe.sample_rate,
                "codec_name": runtime.probe.codec_name,
                "channel_layout": runtime.probe.channel_layout,
            }
            if runtime.probe is not None
            else None
        ),
        "conversion": runtime.conversion,
        "static_gain_db": spec.gain_db,
        "parameters": {
            "target_lufs": spec.target_lufs,
            "noise_gate": asdict(spec.noise_gate),
            "loudness": asdict(spec.loudness),
            "ducking": asdict(spec.ducking),
        },
    }
    if runtime.denoise is not None:
        denoise = runtime.denoise
        result["denoise"] = {
            "method": "sliding-window noise-floor gate with SNR-margin cap",
            "window_ms": denoise.gate_window_ms,
            "hop_ms": denoise.gate_hop_ms,
            "threshold_db": denoise.gate_threshold_db,
            "estimated_noise_floor_dbfs": denoise.noise_floor_dbfs,
            "input_snr_db": _finite_or_none(denoise.input_snr_db),
            "output_snr_db": _finite_or_none(denoise.output_snr_db),
            "max_allowed_output_snr_db": _finite_or_none(
                denoise.max_allowed_snr_db
            ),
            "snr_margin_db": spec.noise_gate.snr_margin_db,
            "snr_constraint_met": denoise.snr_constraint_met,
            "attenuation_scale": denoise.attenuation_scale,
            "max_gate_reduction_db": denoise.max_gate_reduction_db,
        }
    if runtime.activity is not None:
        activity = runtime.activity
        result["activity"] = {
            "basis": "sample_index_at_48000_hz",
            "detector": "centered moving RMS with adaptive floor and min-gap merge",
            "window_ms": spec.ducking.vad_window_ms,
            "noise_floor_dbfs": activity.noise_floor_db,
            "threshold_dbfs": activity.threshold_db,
            "intervals_samples": _intervals_json(activity.intervals),
            "intervals_seconds": _intervals_seconds(activity.intervals, 48_000),
        }
    if runtime.loudness is not None:
        loudness = runtime.loudness
        result["loudness"] = {
            "target_lufs": loudness.target_lufs,
            "integrated_lufs": _finite_or_none(loudness.integrated_lufs),
            "short_term_window_ms": 400.0,
            "short_term_max_abs_deviation_lu": _finite_or_none(
                loudness.max_abs_deviation_lu
            ),
            "short_term_active_frames": loudness.active_frame_count,
            "within_1_lu": loudness.within_tolerance,
            "iterations": loudness.iterations,
            "gain_min_db": loudness.gain_min_db,
            "gain_max_db": loudness.gain_max_db,
            "peak_before_guard_dbfs": _finite_or_none(
                amp_to_db(loudness.peak_before_guard)
                if loudness.peak_before_guard > 0.0
                else None
            ),
            "peak_guard_scale_db": loudness.peak_guard_scale_db,
            "gain_envelope_state": "continuous_across_output_segments",
        }
    if runtime.duck_curve is not None:
        duck = runtime.duck_curve
        result["ducking"] = {
            "depth_db": spec.ducking.depth_db,
            "attack_ms": spec.ducking.attack_ms,
            "release_ms": spec.ducking.release_ms,
            "alignment": "sample_exact_against_detected_dialogue_intervals",
            "attack_start_samples": list(duck.attack_start_samples),
            "release_end_samples": [list(pair) for pair in duck.release_end_samples],
            "min_gain_db": float(np.min(duck.gain_db)) if duck.gain_db.size else 0.0,
            "max_gain_db": float(np.max(duck.gain_db)) if duck.gain_db.size else 0.0,
        }
    return result


def _intervals_json(intervals: Iterable[tuple[int, int]]) -> list[list[int]]:
    return [[int(start), int(end)] for start, end in intervals]


def _intervals_seconds(
    intervals: Iterable[tuple[int, int]],
    sample_rate: int,
) -> list[list[float]]:
    return [
        [round(start / float(sample_rate), 9), round(end / float(sample_rate), 9)]
        for start, end in intervals
    ]


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_or_none(float(value))
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value
