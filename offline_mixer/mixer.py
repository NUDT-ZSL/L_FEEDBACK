"""End-to-end mixing pipeline and report assembly."""

from __future__ import annotations

import math
import time

import numpy as np

from . import TARGET_SR
from .audioio import load_wav, save_wav_stereo
from .ducking import apply_ducking, measure_recovery_time
from .limiter import soft_limit
from .loudness import integrated_loudness, normalize_loudness
from .resample import resample_status, resample_to_48k
from .timeline import Segment, Timeline
from .timescale import align_segment


def _load_reference(path: str | None, total_length: int) -> np.ndarray:
    if path is None:
        return np.zeros(total_length)
    signal, sample_rate = load_wav(path)
    signal = resample_to_48k(signal, sample_rate)
    if signal.size < total_length:
        signal = np.pad(signal, (0, total_length - signal.size))
    return signal[:total_length]


def _extract_source(segment: Segment, cache: dict) -> tuple[np.ndarray, int, dict]:
    if segment.file not in cache:
        raw, source_sr = load_wav(segment.file)
        resampled = resample_to_48k(raw, source_sr)
        cache[segment.file] = (resampled, source_sr)
    resampled, source_sr = cache[segment.file]
    start_sample = int(round(segment.source_start * TARGET_SR))
    end_sample = int(round(segment.source_end * TARGET_SR))
    return resampled[start_sample:end_sample].copy(), source_sr, resample_status(source_sr)


def process_segment(segment: Segment, cache: dict) -> tuple[dict, np.ndarray | None]:
    """Process one timeline segment; report is always returned, audio may be None."""
    started = time.perf_counter()
    source, source_sr, conversion = _extract_source(segment, cache)
    source_seconds = source.size / TARGET_SR
    target_seconds = segment.duration

    report = {
        "timeline_index": segment.index,
        "language": segment.language,
        "file": segment.file,
        "timeline_start": segment.start,
        "timeline_end": segment.end,
        "source_range": [segment.source_start, segment.source_end],
        "source_sample_rate": source_sr,
        "resampling": conversion,
        "original_duration_seconds": round(source_seconds, 6),
        "target_duration_seconds": round(target_seconds, 6),
        "aligned_duration_seconds": 0.0,
        "speed_ratio": None,
        "gain_adjustment_db": 0.0,
        "measured_lufs_before": None,
        "measured_lufs_after": None,
        "target_lufs": segment.target_lufs,
        "loudness_error_lu": None,
        "is_silence": False,
        "aligned": False,
        "alignment_error": None,
        "processing_seconds": 0.0,
    }

    before_lufs = integrated_loudness(source, TARGET_SR)
    report["measured_lufs_before"] = _safe_round(before_lufs)

    if source.size == 0:
        # Zero-length source/target: nothing to align or normalize, but it is
        # still classified as silence in the report.
        report["is_silence"] = True

    aligned, ratio, alignment_error = align_segment(source, target_seconds, TARGET_SR)
    report["speed_ratio"] = None if math.isnan(ratio) else round(ratio, 4)
    if alignment_error is not None:
        report["alignment_error"] = alignment_error
        report["processing_seconds"] = round(time.perf_counter() - started, 6)
        return report, None

    normalized, gain_db, realized_lufs, is_silence = normalize_loudness(
        aligned, segment.target_lufs, TARGET_SR)
    report.update({
        "aligned": True,
        "aligned_duration_seconds": round(normalized.size / TARGET_SR, 6),
        "gain_adjustment_db": round(gain_db, 4),
        "measured_lufs_after": _safe_round(realized_lufs),
        "is_silence": bool(is_silence),
    })
    if not is_silence and not math.isnan(realized_lufs):
        report["loudness_error_lu"] = round(realized_lufs - segment.target_lufs, 4)
    report["processing_seconds"] = round(time.perf_counter() - started, 6)
    return report, normalized


def run_mix(timeline: Timeline, output_wav: str, report_path: str) -> dict:
    pipeline_start = time.perf_counter()
    total_length = int(round(timeline.duration * TARGET_SR))
    voice_bus = np.zeros(total_length, dtype=np.float64)
    active_mask = np.zeros(total_length, dtype=bool)
    segment_reports: list[dict] = []
    cache: dict = {}

    for segment in timeline.segments:
        report, processed = process_segment(segment, cache)
        segment_reports.append(report)
        if processed is None:
            continue
        start = int(round(segment.start * TARGET_SR))
        end = min(start + processed.size, total_length)
        if end > start:
            voice_bus[start:end] += processed[: end - start]
            active_mask[start:end] = True

    music = _load_reference(timeline.music_path, total_length)
    ducked_music, gain_envelope = apply_ducking(music, voice_bus, TARGET_SR)
    recovery_time = measure_recovery_time(gain_envelope, active_mask, TARGET_SR)
    max_reduction_db = round(
        float(-20.0 * np.log10(max(np.min(gain_envelope), 1e-8))), 4)

    sfx_bus = np.zeros(total_length, dtype=np.float64)
    for sfx_path in timeline.sfx_paths:
        sfx_bus += _load_reference(sfx_path, total_length)

    raw_mix = voice_bus + ducked_music + sfx_bus
    limited_mix, limiter_stats = soft_limit(raw_mix)
    save_wav_stereo(output_wav, limited_mix, TARGET_SR)

    report = {
        "status": "ok",
        "project": timeline.project,
        "output": {
            "mix_wav": output_wav,
            "sample_rate": TARGET_SR,
            "channels": 2,
            "bit_depth": 16,
            "duration_seconds": round(total_length / TARGET_SR, 6),
        },
        "summary": {
            "segment_count": len(timeline.segments),
            "aligned_count": sum(1 for item in segment_reports if item["aligned"]),
            "unalignable_count": sum(1 for item in segment_reports if item["alignment_error"]),
            "silence_count": sum(1 for item in segment_reports if item["is_silence"]),
            "overlapping_segments": _detect_overlaps(timeline),
            "ducking": {
                "max_gain_reduction_db": max_reduction_db,
                "recovery_time_seconds": recovery_time,
                "recovery_limit_seconds": 0.2,
            },
            "limiter": limiter_stats,
            "total_processing_seconds": round(time.perf_counter() - pipeline_start, 6),
        },
        "segments": segment_reports,
    }
    import json
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def _detect_overlaps(timeline: Timeline) -> list[list[int]]:
    overlaps = []
    ordered = sorted(timeline.segments, key=lambda seg: (seg.start, seg.end))
    for pos in range(1, len(ordered)):
        previous, current = ordered[pos - 1], ordered[pos]
        if current.start < previous.end and previous.duration > 0 and current.duration > 0:
            overlaps.append([previous.index, current.index])
    return overlaps


def _safe_round(value: float) -> float | None:
    if value is None or math.isnan(value):
        return None
    return round(float(value), 4)
