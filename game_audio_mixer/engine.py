from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import soundfile as sf

from .dsp import (
    SAMPLE_RATE,
    SUPPORTED_SAMPLE_RATES,
    amplitude_to_db,
    apply_limiter,
    compress_dynamic_range,
    db_to_amplitude,
    integrated_loudness,
    quantize_pcm24,
    resample_audio,
    short_term_loudness,
    soft_limit,
    true_peak,
)
from .model import (
    CachedAsset,
    ClipSpec,
    EventSpec,
    ParsedTimeline,
    RenderedEvent,
    TrackConfig,
    TrackState,
)


REPORT_SCHEMA = "game-audio-mixer.report/v1"
MAX_TIMELINE_MS = 86_400_000.0
MAX_OUTPUT_SECONDS = 7200.0


class MixProcessingError(Exception):
    pass


@dataclass
class MixOptions:
    target_lufs: float = -23.0
    mute_override: Optional[List[str]] = None
    solo_override: Optional[List[str]] = None
    compressor_threshold_db: float = -24.0
    compressor_ratio: float = 2.5
    compressor_knee_db: float = 6.0
    compressor_attack_ms: float = 10.0
    compressor_release_ms: float = 180.0
    limiter_threshold_db: float = -1.0
    limiter_ceiling_db: float = -0.2
    oversampling: int = 4


@dataclass
class AssetCache:
    input_dir: Path
    successful: Dict[Path, CachedAsset] = field(default_factory=dict)
    failed: Dict[Path, str] = field(default_factory=dict)
    load_attempts: int = 0
    cache_hits: int = 0

    def load(self, spec: ClipSpec) -> Tuple[Optional[CachedAsset], Optional[str]]:
        path = spec.path.resolve()
        if path in self.successful:
            self.cache_hits += 1
            asset = self.successful[path]
            if spec.id not in asset.referenced_by:
                asset.referenced_by.append(spec.id)
            return asset, None
        if path in self.failed:
            self.cache_hits += 1
            return None, self.failed[path]

        self.load_attempts += 1
        if not path.exists() or not path.is_file():
            message = f"asset file not found: {path}"
            self.failed[path] = message
            return None, message
        try:
            info = sf.info(str(path))
        except Exception as exc:
            message = f"cannot inspect/decode asset '{path}': {exc}"
            self.failed[path] = message
            return None, message
        if info.samplerate not in SUPPORTED_SAMPLE_RATES:
            message = (
                f"unsupported sample rate {info.samplerate} Hz for '{path}'; "
                f"supported: {', '.join(str(rate) for rate in SUPPORTED_SAMPLE_RATES)}"
            )
            self.failed[path] = message
            return None, message
        if info.channels not in (1, 2):
            message = (
                f"unsupported channel count {info.channels} for '{path}'; "
                "only mono and stereo are supported"
            )
            self.failed[path] = message
            return None, message
        if info.frames <= 0:
            message = f"asset has no audio frames: {path}"
            self.failed[path] = message
            return None, message
        try:
            audio, sample_rate = sf.read(
                str(path), dtype="float64", always_2d=True
            )
        except Exception as exc:
            message = f"cannot decode asset '{path}': {exc}"
            self.failed[path] = message
            return None, message

        if audio.shape[0] == 0:
            message = f"asset decoded with zero frames: {path}"
            self.failed[path] = message
            return None, message
        if not np.all(np.isfinite(audio)):
            message = f"asset contains non-finite samples: {path}"
            self.failed[path] = message
            return None, message
        try:
            resampled, method, up, down = resample_audio(audio, sample_rate)
        except Exception as exc:
            message = f"resampling failed for '{path}': {exc}"
            self.failed[path] = message
            return None, message
        if resampled.shape[1] == 1:
            resampled = np.repeat(resampled, 2, axis=1)
        asset = CachedAsset(
            path=path,
            sample_rate=sample_rate,
            channels=info.channels,
            frames=info.frames,
            audio=np.asarray(resampled, dtype=np.float64),
            resampling=method,
            ratio_up=up,
            ratio_down=down,
            referenced_by=[spec.id],
            load_count=1,
        )
        self.successful[path] = asset
        return asset, None


def _parse_override(raw_values: Optional[Sequence[str]]) -> Optional[Set[str]]:
    if raw_values is None:
        return None
    result: Set[str] = set()
    for raw in raw_values:
        for item in raw.split(","):
            track = item.strip()
            if track:
                result.add(track)
    return result


def resolve_track_states(
    tracks: Sequence[TrackConfig],
    mute_override: Optional[Sequence[str]],
    solo_override: Optional[Sequence[str]],
) -> Tuple[List[TrackState], List[str]]:
    mute_set = _parse_override(mute_override)
    solo_set = _parse_override(solo_override)
    warnings: List[str] = []
    known_ids = {track.id for track in tracks}
    for track_id in sorted((mute_set or set()) - known_ids):
        warnings.append(f"mute override references unknown track '{track_id}'")
    for track_id in sorted((solo_set or set()) - known_ids):
        warnings.append(f"solo override references unknown track '{track_id}'")

    effective_mute: Dict[str, bool] = {}
    effective_solo: Dict[str, bool] = {}
    for track in tracks:
        if mute_set is None:
            effective_mute[track.id] = track.initial_mute
        else:
            effective_mute[track.id] = track.id in mute_set
        if solo_set is None:
            effective_solo[track.id] = track.initial_solo
        else:
            effective_solo[track.id] = track.id in solo_set
    any_solo = any(effective_solo.values())

    states: List[TrackState] = []
    for track in tracks:
        solo = effective_solo[track.id]
        mute = effective_mute[track.id]
        audible = solo if any_solo else not mute
        states.append(
            TrackState(
                id=track.id,
                name=track.name,
                initial_mute=track.initial_mute,
                initial_solo=track.initial_solo,
                effective_mute=mute,
                effective_solo=solo,
                audible=audible,
                mute_source="cli" if mute_set is not None else "timeline",
                solo_source="cli" if solo_set is not None else "timeline",
                gain_db=track.gain_db,
            )
        )
    return states, warnings


def _ms_to_frame(value_ms: float) -> int:
    return int(round(value_ms * SAMPLE_RATE / 1000.0))


def _resolve_clip(
    event: EventSpec, timeline: ParsedTimeline, input_dir: Path
) -> Optional[ClipSpec]:
    spec = timeline.clip_specs.get(event.asset_id)
    if spec is not None:
        return spec
    direct = Path(event.asset_id).expanduser()
    if not direct.is_absolute():
        direct = input_dir / direct
    if direct.suffix.lower() in (".wav", ".wave", ".flac", ".aiff", ".aif", ".ogg"):
        return ClipSpec(id=event.asset_id, path=direct.resolve())
    return None


def _validate_event_time(event: EventSpec, timeline_duration_ms: Optional[float]) -> Optional[str]:
    if event.time_ms < 0.0:
        return "negative timestamp"
    if event.time_ms > MAX_TIMELINE_MS:
        return f"timestamp exceeds maximum supported timeline ({MAX_TIMELINE_MS:.0f} ms)"
    if (
        timeline_duration_ms is not None
        and event.time_ms >= timeline_duration_ms
    ):
        return (
            f"timestamp {event.time_ms:g} ms is outside timeline duration "
            f"{timeline_duration_ms:g} ms"
        )
    return None


def _event_error(event: EventSpec, reason: str) -> Dict[str, Any]:
    return {
        "scope": "event",
        "event_index": event.source_index,
        "event_id": event.id,
        "track": event.track_id,
        "reason": reason,
    }


def _render_events(
    timeline: ParsedTimeline,
    input_dir: Path,
    cache: AssetCache,
    track_lookup: Dict[str, TrackState],
    warnings: List[str],
    errors: List[Dict[str, Any]],
    event_reports: List[Dict[str, Any]],
) -> Tuple[List[RenderedEvent], int]:
    rendered: List[RenderedEvent] = []
    max_end_frame = 0
    for event in timeline.events:
        invalid_reason = _validate_event_time(event, timeline.timeline_duration_ms)
        if invalid_reason:
            errors.append(_event_error(event, invalid_reason))
            event_reports.append(
                {
                    "event_id": event.id,
                    "type": event.event_type,
                    "track": event.track_id,
                    "asset_id": event.asset_id,
                    "time_ms": event.time_ms,
                    "status": "skipped",
                    "reason": invalid_reason,
                    "start_s": None,
                    "end_s": None,
                }
            )
            continue

        spec = _resolve_clip(event, timeline, input_dir)
        if spec is None:
            reason = f"clip/asset reference '{event.asset_id}' not found in manifest"
            errors.append(_event_error(event, reason))
            event_reports.append(
                {
                    "event_id": event.id,
                    "type": event.event_type,
                    "track": event.track_id,
                    "asset_id": event.asset_id,
                    "time_ms": event.time_ms,
                    "status": "skipped",
                    "reason": reason,
                    "start_s": None,
                    "end_s": None,
                }
            )
            continue

        asset, load_error = cache.load(spec)
        if asset is None:
            reason = load_error or f"failed to load asset '{spec.path}'"
            errors.append(_event_error(event, reason))
            event_reports.append(
                {
                    "event_id": event.id,
                    "type": event.event_type,
                    "track": event.track_id,
                    "asset_id": event.asset_id,
                    "time_ms": event.time_ms,
                    "status": "skipped",
                    "reason": reason,
                    "start_s": None,
                    "end_s": None,
                }
            )
            continue

        offset_ms = event.offset_ms if event.offset_ms is not None else spec.offset_ms
        duration_ms = (
            event.duration_ms if event.duration_ms is not None else spec.duration_ms
        )
        offset_frame = max(0, _ms_to_frame(offset_ms))
        if offset_frame >= asset.audio.shape[0]:
            reason = (
                f"clip offset {offset_ms:g} ms is beyond asset duration "
                f"({asset.audio.shape[0] / SAMPLE_RATE:.3f} s)"
            )
            errors.append(_event_error(event, reason))
            event_reports.append(
                {
                    "event_id": event.id,
                    "type": event.event_type,
                    "track": event.track_id,
                    "asset_id": event.asset_id,
                    "time_ms": event.time_ms,
                    "status": "skipped",
                    "reason": reason,
                    "start_s": None,
                    "end_s": None,
                }
            )
            continue

        available = asset.audio.shape[0] - offset_frame
        requested_frames = (
            _ms_to_frame(duration_ms) if duration_ms is not None else available
        )
        if requested_frames <= 0:
            reason = "clip duration resolves to zero frames"
            errors.append(_event_error(event, reason))
            event_reports.append(
                {
                    "event_id": event.id,
                    "type": event.event_type,
                    "track": event.track_id,
                    "asset_id": event.asset_id,
                    "time_ms": event.time_ms,
                    "status": "skipped",
                    "reason": reason,
                    "start_s": None,
                    "end_s": None,
                }
            )
            continue
        frame_count = min(requested_frames, available)
        if frame_count < requested_frames:
            warnings.append(
                f"event '{event.id}' clip duration clamped from "
                f"{requested_frames / SAMPLE_RATE:.3f}s to "
                f"{frame_count / SAMPLE_RATE:.3f}s at source end"
            )

        start_frame = _ms_to_frame(event.time_ms)
        end_frame = start_frame + frame_count
        if end_frame > int(MAX_OUTPUT_SECONDS * SAMPLE_RATE):
            reason = (
                f"event end exceeds maximum output duration "
                f"({MAX_OUTPUT_SECONDS:.0f} s)"
            )
            errors.append(_event_error(event, reason))
            event_reports.append(
                {
                    "event_id": event.id,
                    "type": event.event_type,
                    "track": event.track_id,
                    "asset_id": event.asset_id,
                    "time_ms": event.time_ms,
                    "status": "skipped",
                    "reason": reason,
                    "start_s": None,
                    "end_s": None,
                }
            )
            continue

        track_state = track_lookup[event.track_id]
        combined_gain_db = track_state.gain_db + event.gain_db
        audio = asset.audio[offset_frame : offset_frame + frame_count].copy()
        audio *= db_to_amplitude(combined_gain_db)
        rendered_event = RenderedEvent(
            event=event,
            start_frame=start_frame,
            end_frame=end_frame,
            audio=audio,
            start_s=start_frame / SAMPLE_RATE,
            end_s=end_frame / SAMPLE_RATE,
            audible=track_state.audible,
        )
        rendered.append(rendered_event)
        max_end_frame = max(max_end_frame, end_frame)
        status = "mixed" if track_state.audible else "skipped"
        reason = None if track_state.audible else "track is not audible under mute/solo state"
        event_reports.append(
            {
                "event_id": event.id,
                "type": event.event_type,
                "track": event.track_id,
                "asset_id": event.asset_id,
                "time_ms": event.time_ms,
                "status": status,
                "reason": reason,
                "start_s": round(rendered_event.start_s, 6),
                "end_s": round(rendered_event.end_s, 6),
                "duration_s": round(frame_count / SAMPLE_RATE, 6),
                "gain_db": round(combined_gain_db, 6),
            }
        )
    return rendered, max_end_frame


def _build_mix(rendered: Sequence[RenderedEvent], frames: int) -> np.ndarray:
    mix = np.zeros((frames, 2), dtype=np.float64)
    for rendered_event in rendered:
        if not rendered_event.audible:
            continue
        mix[
            rendered_event.start_frame : rendered_event.end_frame
        ] += rendered_event.audio
    return mix


def _normalize_with_limiter(
    base_audio: np.ndarray,
    target_lufs: float,
    options: MixOptions,
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, np.ndarray]:
    initial_loudness = integrated_loudness(base_audio)
    if initial_loudness is None:
        silence = np.zeros_like(base_audio)
        return (
            silence,
            {
                "iterations": 0,
                "gain_db": 0.0,
                "initial_lufs": None,
                "final_lufs": None,
            },
            silence,
            silence,
        )
    gain_db = target_lufs - initial_loudness
    best: Optional[
        Tuple[float, np.ndarray, np.ndarray, np.ndarray, float, Any, int]
    ] = None
    for iteration in range(1, 17):
        gain_linear = db_to_amplitude(gain_db)
        trial = base_audio * gain_linear
        limiter_result = apply_limiter(
            trial,
            threshold_db=options.limiter_threshold_db,
            ceiling_db=options.limiter_ceiling_db,
        )
        quantized = quantize_pcm24(limiter_result.audio)
        current_loudness = integrated_loudness(quantized)
        if current_loudness is None:
            break
        distance = abs(current_loudness - target_lufs)
        if best is None or distance < best[0]:
            best = (
                distance,
                quantized,
                trial,
                limiter_result.audio,
                gain_db,
                limiter_result,
                iteration,
            )
        if distance <= 0.03:
            break
        gain_db += target_lufs - current_loudness

    if best is None:
        raise MixProcessingError("loudness normalization failed for non-silent audio")
    (
        _,
        final_audio,
        final_pre_limiter,
        final_limited,
        gain_db,
        _final_limiter,
        final_iteration,
    ) = best
    final_lufs = integrated_loudness(final_audio)
    processing = {
        "iterations": final_iteration,
        "gain_db": round(gain_db, 6),
        "initial_lufs": round(initial_loudness, 6),
        "final_lufs": None if final_lufs is None else round(final_lufs, 6),
    }
    return final_audio, processing, final_pre_limiter, final_limited


def _dominant_events(
    rendered: Sequence[RenderedEvent],
    start_frame: int,
    end_frame: int,
) -> List[Dict[str, Any]]:
    contributors: List[Dict[str, Any]] = []
    for rendered_event in rendered:
        if not rendered_event.audible:
            continue
        left = max(start_frame, rendered_event.start_frame)
        right = min(end_frame, rendered_event.end_frame)
        if right <= left:
            continue
        local_start = left - rendered_event.start_frame
        local_end = right - rendered_event.start_frame
        local_peak = float(
            np.max(np.abs(rendered_event.audio[local_start:local_end]))
        )
        if local_peak > 0.0:
            contributors.append(
                {
                    "track": rendered_event.event.track_id,
                    "event_id": rendered_event.event.id,
                    "event_type": rendered_event.event.event_type,
                    "max_peak_dbfs": round(float(amplitude_to_db(local_peak) or -240.0), 6),
                    "time_s": round(left / SAMPLE_RATE, 6),
                }
            )
    contributors.sort(key=lambda item: item["max_peak_dbfs"], reverse=True)
    return contributors


def _limiter_triggers(
    pre: np.ndarray,
    post: np.ndarray,
    reduction_mask: np.ndarray,
    rendered: Sequence[RenderedEvent],
    enabled: bool,
) -> List[Dict[str, Any]]:
    if not enabled:
        return []
    indices = np.flatnonzero(reduction_mask)
    ranges: List[Tuple[int, int]] = []
    if indices.size:
        gap_frames = int(0.05 * SAMPLE_RATE)
        range_start = int(indices[0])
        previous = int(indices[0])
        for current in indices[1:]:
            current = int(current)
            if current - previous > gap_frames:
                ranges.append((range_start, previous + 1))
                range_start = current
            previous = current
        ranges.append((range_start, previous + 1))
    else:
        _, fallback_index = true_peak(pre)
        ranges.append((fallback_index, min(pre.shape[0], fallback_index + 1)))

    triggers: List[Dict[str, Any]] = []
    for range_start, range_stop in ranges:
        region = pre[range_start:range_stop]
        magnitudes = np.max(np.abs(region), axis=1)
        local_index = int(np.argmax(magnitudes))
        peak_frame = range_start + local_index
        peak_value = float(magnitudes[local_index])
        limited_magnitude = float(np.max(np.abs(post[peak_frame])))
        reduction_db = 0.0
        if peak_value > 0.0 and limited_magnitude > 0.0:
            reduction_db = max(0.0, 20.0 * math.log10(peak_value / limited_magnitude))
        contributors = _dominant_events(rendered, range_start, range_stop)
        primary = contributors[0] if contributors else {
            "track": None,
            "event_id": None,
            "event_type": None,
        }
        trigger = {
            "track": primary["track"],
            "event_id": primary["event_id"],
            "event_type": primary["event_type"],
            "time_s": round(peak_frame / SAMPLE_RATE, 6),
            "start_s": round(range_start / SAMPLE_RATE, 6),
            "end_s": round(range_stop / SAMPLE_RATE, 6),
            "peak_dbfs": round(float(amplitude_to_db(peak_value) or -240.0), 6),
            "gain_reduction_db": round(reduction_db, 6),
            "contributors": contributors,
        }
        triggers.append(trigger)
    return triggers


def render_timeline(
    timeline: ParsedTimeline,
    input_dir: Path,
    options: MixOptions,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    warnings = list(timeline.parse_warnings)
    errors = list(timeline.parse_errors)
    event_reports: List[Dict[str, Any]] = []
    track_states, state_warnings = resolve_track_states(
        timeline.track_configs, options.mute_override, options.solo_override
    )
    warnings.extend(state_warnings)
    track_lookup = {state.id: state for state in track_states}
    cache = AssetCache(input_dir=input_dir)
    rendered, max_end_frame = _render_events(
        timeline,
        input_dir,
        cache,
        track_lookup,
        warnings,
        errors,
        event_reports,
    )
    if max_end_frame > int(MAX_OUTPUT_SECONDS * SAMPLE_RATE):
        raise MixProcessingError(
            f"computed output duration exceeds {MAX_OUTPUT_SECONDS:.0f} seconds"
        )
    raw_mix = _build_mix(rendered, max_end_frame)
    raw_loudness = integrated_loudness(raw_mix)

    pre_normalization_gain_db = 0.0
    compressor_result = None
    compressor_input = raw_mix
    if raw_loudness is not None:
        pre_normalization_gain_db = options.target_lufs - raw_loudness
        compressor_input = raw_mix * db_to_amplitude(pre_normalization_gain_db)
        compressor_result = compress_dynamic_range(
            compressor_input,
            threshold_db=options.compressor_threshold_db,
            ratio=options.compressor_ratio,
            knee_db=options.compressor_knee_db,
            attack_ms=options.compressor_attack_ms,
            release_ms=options.compressor_release_ms,
        )
    compressor_output = (
        compressor_result.audio if compressor_result is not None else compressor_input
    )

    final_audio, normalization, pre_limit, limited = _normalize_with_limiter(
        compressor_output, options.target_lufs, options
    )
    final_audio = quantize_pcm24(final_audio)
    final_loudness = integrated_loudness(final_audio)
    final_true_peak, _ = true_peak(final_audio)
    sample_peak = (
        float(np.max(np.abs(final_audio))) if final_audio.size else 0.0
    )
    pre_limit_true_peak, _ = true_peak(pre_limit)
    limiter_threshold = db_to_amplitude(options.limiter_threshold_db)
    limiter_enabled = pre_limit_true_peak > limiter_threshold
    # Reconstruct the mask if the final branch was the best normalization branch.
    if limiter_enabled:
        _, reduction_mask = soft_limit(
            pre_limit, options.limiter_threshold_db, options.limiter_ceiling_db
        )
    else:
        reduction_mask = np.zeros(final_audio.shape[0], dtype=bool)
    triggers = _limiter_triggers(
        pre_limit,
        limited,
        reduction_mask,
        rendered,
        limiter_enabled,
    )

    target_achieved = (
        final_loudness is not None
        and abs(final_loudness - options.target_lufs) <= 0.5
    )
    true_peak_ok = final_true_peak <= 1.0
    if final_loudness is not None and not target_achieved:
        warnings.append(
            "final integrated loudness missed target tolerance: "
            f"{final_loudness:.2f} LUFS vs {options.target_lufs:.2f} +/- 0.5 LU"
        )
    if not true_peak_ok:
        warnings.append(
            f"true peak exceeds 0 dBFS after processing: "
            f"{20.0 * math.log10(max(final_true_peak, 1e-12)):.3f} dBTP"
        )
    if raw_loudness is None and len(rendered) > 0:
        warnings.append("all rendered audio is silent; loudness target cannot be achieved")

    if compressor_result is not None and compressor_result.max_gain_reduction_db < 0.01:
        warnings.append("dynamic-range compressor did not engage above its threshold")

    asset_reports = [
        asset.as_dict()
        for asset in sorted(cache.successful.values(), key=lambda item: str(item.path))
    ]
    report = {
        "report_schema": REPORT_SCHEMA,
        "status": (
            "silent"
            if final_loudness is None
            else (
                "constraint_failed"
                if (
                    (final_loudness is not None and not target_achieved)
                    or not true_peak_ok
                )
                else ("ok_with_warnings" if warnings or errors else "ok")
            )
        ),
        "project": timeline.title,
        "input": {
            "input_directory": str(input_dir.resolve()),
            "timeline_file": (
                None
                if timeline.timeline_path is None
                else str(timeline.timeline_path)
            ),
            "asset_manifest": timeline.asset_manifest,
            "timeline_duration_ms": timeline.timeline_duration_ms,
        },
        "output": {
            "file": "mix.wav",
            "sample_rate": SAMPLE_RATE,
            "bit_depth": 24,
            "encoding": "PCM_24",
            "channels": 2,
            "frames": int(final_audio.shape[0]),
            "duration_s": round(final_audio.shape[0] / SAMPLE_RATE, 6),
        },
        "target": {
            "lufs": options.target_lufs,
            "tolerance_lu": 0.5,
            "achieved": target_achieved,
        },
        "loudness": {
            "integrated_lufs": (
                None if final_loudness is None else round(final_loudness, 6)
            ),
            "true_peak_dbtp": (
                None
                if final_true_peak <= 1e-12
                else round(20.0 * math.log10(final_true_peak), 6)
            ),
            "sample_peak_dbfs": (
                None
                if sample_peak <= 1e-12
                else round(20.0 * math.log10(sample_peak), 6)
            ),
            "short_term": short_term_loudness(final_audio),
            "short_term_window_s": 3.0,
            "short_term_hop_s": 1.0,
            "standard": "ITU-R BS.1770-4 / EBU R128",
            "true_peak_method": "4x polyphase oversampling (scipy.signal.resample_poly)",
        },
        "tracks": [state.as_dict() for state in track_states],
        "assets": asset_reports,
        "asset_cache": {
            "unique_files_referenced": len(cache.successful) + len(cache.failed),
            "load_attempts": cache.load_attempts,
            "successful_loads": len(cache.successful),
            "cache_hits": cache.cache_hits,
            "failed_loads": len(cache.failed),
        },
        "events": event_reports,
        "processing": {
            "pre_normalization_gain_db": round(pre_normalization_gain_db, 6),
            "compressor": {
                "enabled": True,
                "threshold_db": options.compressor_threshold_db,
                "ratio": options.compressor_ratio,
                "knee_db": options.compressor_knee_db,
                "attack_ms": options.compressor_attack_ms,
                "release_ms": options.compressor_release_ms,
                "max_gain_reduction_db": (
                    None
                    if compressor_result is None
                    else round(compressor_result.max_gain_reduction_db, 6)
                ),
                "active_fraction": (
                    None
                    if compressor_result is None
                    else round(compressor_result.active_fraction, 6)
                ),
            },
            "normalization": normalization,
            "limiter": {
                "enabled": limiter_enabled,
                "activation_condition": "pre-limiter true peak > -1.0 dBTP",
                "threshold_dbfs": options.limiter_threshold_db,
                "soft_ceiling_dbfs": options.limiter_ceiling_db,
                "true_peak_ceiling_dbfs": 0.0,
                "pre_limiter_true_peak_dbtp": (
                    None
                    if pre_limit_true_peak <= 1e-12
                    else round(20.0 * math.log10(pre_limit_true_peak), 6)
                ),
                "trigger_count": len(triggers),
            },
        },
        "limiter_triggers": triggers,
        "warnings": warnings,
        "errors": errors,
    }
    return final_audio, report


def save_outputs(
    output_dir: Path,
    audio: np.ndarray,
    report: Dict[str, Any],
) -> Tuple[Path, Path]:
    output_dir = output_dir.expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise MixProcessingError(f"cannot create output directory '{output_dir}': {exc}")
    wav_path = output_dir / "mix.wav"
    report_path = output_dir / "loudness_report.json"
    try:
        sf.write(
            str(wav_path),
            np.asarray(audio, dtype=np.float32),
            SAMPLE_RATE,
            subtype="PCM_24",
            format="WAV",
        )
    except Exception as exc:
        raise MixProcessingError(f"cannot write mix WAV '{wav_path}': {exc}")
    try:
        serialized = json.dumps(
            report, ensure_ascii=False, indent=2, allow_nan=False
        )
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(serialized + "\n", encoding="utf-8")
        os.replace(str(temporary), str(report_path))
    except (OSError, TypeError, ValueError) as exc:
        raise MixProcessingError(f"cannot write JSON report '{report_path}': {exc}")
    return wav_path, report_path


def build_and_save(
    timeline: ParsedTimeline,
    input_dir: Path,
    output_dir: Path,
    options: MixOptions,
) -> Tuple[Path, Path, Dict[str, Any], np.ndarray]:
    final_audio, report = render_timeline(timeline, input_dir, options)
    wav_path, report_path = save_outputs(output_dir, final_audio, report)
    return wav_path, report_path, report, final_audio
