from __future__ import annotations

import math

import numpy as np

from .ffmpeg_tools import decode_audio
from .models import Manifest, VideoInfo
from .timeline import TimelinePlan


SYNC_ANALYSIS_RATE = 4000
DEFAULT_SYNC_WINDOW_SECONDS = 60.0
DEFAULT_SEARCH_RANGE_MS = 2000.0
MAX_COMPENSATION_MS = 500.0


def decimate_mono(signal: np.ndarray, sample_rate: int, target_rate: int) -> np.ndarray:
    mono = np.asarray(signal, dtype=np.float32)
    if mono.ndim == 2:
        mono = np.mean(mono, axis=1, dtype=np.float32)
    if mono.ndim != 1:
        raise ValueError("signal must be one- or two-dimensional")
    if sample_rate == target_rate:
        return np.ascontiguousarray(mono)
    if sample_rate % target_rate == 0:
        factor = sample_rate // target_rate
        usable = (len(mono) // factor) * factor
        if usable == 0:
            return np.empty(0, dtype=np.float32)
        return np.ascontiguousarray(
            mono[:usable].reshape(-1, factor).mean(axis=1, dtype=np.float32)
        )

    output_length = max(1, int(round(len(mono) * target_rate / sample_rate)))
    width = max(1, int(round(sample_rate / target_rate / 2)))
    if width > 1 and len(mono) >= width:
        cumulative = np.cumsum(mono, dtype=np.float64)
        cumulative = np.concatenate(([0.0], cumulative))
        smoothed = (cumulative[width:] - cumulative[:-width]) / width
        smoothed = np.concatenate(
            (
                np.full(width // 2, smoothed[0] if len(smoothed) else 0.0),
                smoothed,
                np.full(width - 1 - width // 2, smoothed[-1] if len(smoothed) else 0.0),
            )
        )[: len(mono)]
    else:
        smoothed = mono
    positions = (np.arange(output_length, dtype=np.float64) + 0.5) * sample_rate / target_rate
    return np.interp(positions, np.arange(len(smoothed)), smoothed).astype(np.float32)


def shift_audio(audio: np.ndarray, shift_samples: int) -> np.ndarray:
    shifted = np.zeros_like(audio)
    source = np.asarray(audio)
    if shift_samples == 0:
        return np.array(source, copy=True)
    if abs(shift_samples) >= source.shape[0]:
        return shifted
    if shift_samples > 0:
        shifted[shift_samples:] = source[:-shift_samples]
    else:
        shifted[:shift_samples] = source[-shift_samples:]
    return shifted


def estimate_sync(
    synthesized_mono: np.ndarray,
    reference_mono: np.ndarray | None,
    *,
    output_sample_rate: int,
    reference_stream_index: int | None,
    window_seconds: float = DEFAULT_SYNC_WINDOW_SECONDS,
    search_range_ms: float = DEFAULT_SEARCH_RANGE_MS,
    compensation_limit_ms: float = MAX_COMPENSATION_MS,
) -> dict:
    base = {
        "status": "no_reference",
        "reference_present": reference_mono is not None,
        "reference_stream_index": reference_stream_index,
        "analysis_sample_rate": SYNC_ANALYSIS_RATE,
        "analysis_window_seconds": window_seconds,
        "search_range_ms": search_range_ms,
        "compensation_limit_ms": compensation_limit_ms,
        "estimated_offset_ms": None,
        "applied_shift_samples": 0,
        "applied_shift_ms": 0.0,
        "confidence": None,
        "sign_convention": "positive estimated offset delays synthesized audio",
    }
    if reference_mono is None:
        return base

    synth = np.asarray(synthesized_mono, dtype=np.float32).reshape(-1)
    reference = np.asarray(reference_mono, dtype=np.float32).reshape(-1)
    window_samples = max(1, int(round(window_seconds * SYNC_ANALYSIS_RATE)))
    count = min(len(synth), len(reference), window_samples)
    if count < 2:
        base["status"] = "insufficient_signal"
        return base
    x = synth[:count].astype(np.float64)
    y = reference[:count].astype(np.float64)
    x -= np.mean(x)
    y -= np.mean(y)
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if x_norm < 1e-12 or y_norm < 1e-12:
        base["status"] = "insufficient_signal"
        return base
    x /= x_norm
    y /= y_norm
    tapering = np.hanning(count)
    xw = x * tapering
    yw = y * tapering

    nfft = 1 << max(1, (2 * count - 1).bit_length())
    cross = np.fft.irfft(np.fft.rfft(xw, nfft) * np.conj(np.fft.rfft(yw, nfft)), nfft)
    max_lag = min(
        count - 1,
        int(round(search_range_ms * SYNC_ANALYSIS_RATE / 1000.0)),
    )
    lags = np.arange(-max_lag, max_lag + 1)
    correlation = np.concatenate((cross[-max_lag:], cross[: max_lag + 1]))
    peak_index = int(np.argmax(correlation))
    peak_lag = float(lags[peak_index])

    delta = 0.0
    if 0 < peak_index < len(correlation) - 1:
        left = float(correlation[peak_index - 1])
        center = float(correlation[peak_index])
        right = float(correlation[peak_index + 1])
        denominator = left - 2.0 * center + right
        if abs(denominator) > 1e-15:
            delta = 0.5 * (left - right) / denominator
            delta = max(-0.5, min(0.5, delta))
    refined_lag = peak_lag + delta
    estimated_offset_ms = -refined_lag * 1000.0 / SYNC_ANALYSIS_RATE
    shift_samples = int(round(-refined_lag * output_sample_rate / SYNC_ANALYSIS_RATE))

    aligned_synth = shift_audio(synth[:count], -int(round(refined_lag)))
    overlap_a = aligned_synth.astype(np.float64)
    overlap_b = reference[:count].astype(np.float64)
    overlap_a -= np.mean(overlap_a)
    overlap_b -= np.mean(overlap_b)
    denominator = float(np.linalg.norm(overlap_a) * np.linalg.norm(overlap_b))
    confidence = (
        float(np.dot(overlap_a, overlap_b) / denominator)
        if denominator > 1e-12
        else 0.0
    )

    base.update(
        {
            "estimated_offset_ms": round(estimated_offset_ms, 3),
            "confidence": round(confidence, 4),
            "peak_lag_samples_at_analysis_rate": round(refined_lag, 3),
        }
    )
    if abs(estimated_offset_ms) > compensation_limit_ms + 1e-9:
        base["status"] = "out_of_range"
        return base
    if not math.isfinite(confidence) or confidence < 0.05:
        base["status"] = "low_confidence"
        return base
    base["status"] = "compensated"
    base["applied_shift_samples"] = shift_samples
    base["applied_shift_ms"] = round(shift_samples * 1000.0 / output_sample_rate, 3)
    return base


def estimate_sync_for_output(
    synthesized: np.ndarray,
    output_sample_rate: int,
    video: VideoInfo,
    *,
    window_seconds: float = DEFAULT_SYNC_WINDOW_SECONDS,
) -> dict:
    reference_index = video.selected_reference_index
    if reference_index is None:
        return estimate_sync(
            decimate_mono(synthesized, output_sample_rate, SYNC_ANALYSIS_RATE),
            None,
            output_sample_rate=output_sample_rate,
            reference_stream_index=None,
            window_seconds=window_seconds,
        )
    reference = decode_audio(
        video.path,
        stream_selector=f"0:{reference_index}",
        sample_rate=SYNC_ANALYSIS_RATE,
        channels=1,
        duration_seconds=window_seconds,
    )[:, 0]
    synth = decimate_mono(synthesized, output_sample_rate, SYNC_ANALYSIS_RATE)
    return estimate_sync(
        synth,
        reference,
        output_sample_rate=output_sample_rate,
        reference_stream_index=reference_index,
        window_seconds=window_seconds,
    )


def dry_run_sync(
    manifest: Manifest,
    video: VideoInfo,
    plan: TimelinePlan,
    *,
    window_seconds: float = DEFAULT_SYNC_WINDOW_SECONDS,
) -> dict:
    reference_index = video.selected_reference_index
    preview_samples = max(2, int(round(window_seconds * SYNC_ANALYSIS_RATE)))
    timeline_preview = np.zeros(preview_samples, dtype=np.float32)
    used_languages: list[str] = []
    decoded: dict[str, np.ndarray] = {}
    for segment in plan.segments:
        if segment.language_id not in decoded:
            language = manifest.language_by_id[segment.language_id]
            decoded[segment.language_id] = decode_audio(
                language.path,
                sample_rate=SYNC_ANALYSIS_RATE,
                channels=1,
                duration_seconds=window_seconds,
            )[:, 0]
            used_languages.append(segment.language_id)
        start = int(
            round(segment.start_sample * SYNC_ANALYSIS_RATE / manifest.audio.sample_rate)
        )
        end = int(
            round(segment.end_sample * SYNC_ANALYSIS_RATE / manifest.audio.sample_rate)
        )
        if start >= preview_samples:
            continue
        end = min(end, preview_samples)
        source = decoded[segment.language_id]
        length = min(end - start, max(0, len(source) - start))
        if length > 0:
            timeline_preview[start : start + length] = source[start : start + length]

    if reference_index is None:
        reference = None
    elif timeline_preview.size == 0:
        reference = None
    else:
        reference = decode_audio(
            video.path,
            stream_selector=f"0:{reference_index}",
            sample_rate=SYNC_ANALYSIS_RATE,
            channels=1,
            duration_seconds=window_seconds,
        )[:, 0]
    result = estimate_sync(
        timeline_preview,
        reference,
        output_sample_rate=manifest.audio.sample_rate,
        reference_stream_index=reference_index,
        window_seconds=window_seconds,
    )
    result["preview_languages"] = used_languages
    return result

