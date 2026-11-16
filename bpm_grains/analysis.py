"""Deterministic tempo estimation from an onset-envelope autocorrelation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .errors import LowConfidenceError

MIN_BPM = 40.0
MAX_BPM = 240.0
CONFIDENCE_THRESHOLD = 0.70
MAX_ANALYSIS_SECONDS = 120.0


@dataclass(frozen=True)
class BpmResult:
    bpm: float
    confidence: float
    beat_samples: int


def round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def _onset_envelope(mono: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
    if mono.size == 0:
        return np.zeros(0, dtype=np.float64), sample_rate / 220.0

    frame_size = 2048
    hop = max(1, round_half_up(sample_rate * 0.005))
    frame_count = 1 + max(0, mono.size - frame_size) // hop
    padded = np.pad(mono, (0, max(0, frame_size - mono.size)))
    frame_starts = np.arange(frame_count, dtype=np.int64)[:, None] * hop
    frame_index = frame_starts + np.arange(frame_size, dtype=np.int64)[None, :]
    frames = padded[frame_index] * np.hanning(frame_size)[None, :]

    magnitude = np.abs(np.fft.rfft(frames, axis=1))
    log_magnitude = np.log1p(magnitude * 10.0)
    spectral_flux = np.concatenate(
        (
            np.zeros(1, dtype=np.float64),
            np.maximum(0.0, np.diff(log_magnitude, axis=0)).sum(axis=1),
        )
    )

    rms = np.sqrt(np.mean(frames * frames, axis=1))
    energy_rise = np.maximum(0.0, np.diff(rms, prepend=rms[0]))
    flux_norm = _normalize_positive(spectral_flux)
    energy_norm = _normalize_positive(energy_rise)

    combined = 0.75 * flux_norm + 0.25 * energy_norm
    combined = np.convolve(combined, np.array([0.25, 0.5, 0.25]), mode="same")
    return combined, sample_rate / float(hop)


def _normalize_positive(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    shifted = values - np.median(values)
    shifted = np.maximum(shifted, 0.0)
    peak = float(shifted.max()) if shifted.size else 0.0
    if peak > 0.0:
        shifted /= peak
    return shifted


def _autocorrelation(envelope: np.ndarray) -> np.ndarray:
    centered = envelope - float(envelope.mean())
    length = centered.size
    if length < 4:
        return np.zeros(length, dtype=np.float64)

    fft_size = 1
    while fft_size < 2 * length:
        fft_size *= 2
    spectrum = np.fft.rfft(centered, fft_size)
    raw = np.fft.irfft(spectrum * np.conj(spectrum), fft_size)[:length]
    raw /= np.maximum(1, length - np.arange(length, dtype=np.int64))
    if raw[0] <= 0.0:
        return np.zeros(length, dtype=np.float64)
    return raw / raw[0]


def _interpolated(series: np.ndarray, position: float) -> float:
    if position <= 0.0 or position >= series.size - 1:
        return 0.0
    left = int(math.floor(position))
    fraction = position - left
    return float(series[left] * (1.0 - fraction) + series[left + 1] * fraction)


def _tempo_candidates(
    autocorrelation: np.ndarray, frames_per_second: float
) -> list[tuple[float, float, float]]:
    candidates: list[tuple[float, float, float]] = []
    length = autocorrelation.size
    harmonic_weights = ((1, 1.0), (2, 0.60), (3, 0.35), (4, 0.25))

    for bpm in np.arange(MIN_BPM, MAX_BPM + 0.25, 0.25):
        period_frames = 60.0 * frames_per_second / float(bpm)
        weighted_values: list[tuple[float, float]] = []
        for harmonic, weight in harmonic_weights:
            lag = harmonic * period_frames
            if lag < length - 2:
                value = max(0.0, _interpolated(autocorrelation, lag))
                weighted_values.append((value, weight))
        if not weighted_values:
            continue
        score = sum(value * weight for value, weight in weighted_values) / sum(
            weight for _, weight in weighted_values
        )
        # A very small prior stabilizes the common 60-180 BPM game-music range
        # without changing the measured period.
        log_ratio = math.log2(float(bpm) / 120.0)
        score *= 1.0 + 0.01 * math.exp(-((log_ratio / 0.9) ** 2))
        candidates.append((float(score), float(bpm), float(period_frames)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def _onset_interval_stats(
    envelope: np.ndarray, frames_per_second: float
) -> tuple[float | None, float]:
    if envelope.size < 4:
        return None, 0.0

    threshold = max(0.08, float(np.percentile(envelope, 75)) * 0.45)
    interior = envelope[1:-1]
    is_peak = (
        (interior >= envelope[:-2])
        & (interior > envelope[2:])
        & (interior >= threshold)
    )
    peak_indices = np.flatnonzero(is_peak) + 1
    if peak_indices.size < 3:
        return None, 0.0

    minimum_gap = max(1, round_half_up(frames_per_second * 0.10))
    filtered: list[int] = []
    for index in peak_indices:
        if not filtered or index - filtered[-1] >= minimum_gap:
            filtered.append(int(index))
    if len(filtered) < 3:
        return None, 0.0

    intervals = np.diff(np.asarray(filtered, dtype=np.float64)) / frames_per_second
    intervals = intervals[(intervals >= 60.0 / MAX_BPM * 0.75)]
    intervals = intervals[intervals <= 60.0 / MIN_BPM * 1.25]
    if intervals.size < 2:
        return None, 0.0

    median_interval = float(np.median(intervals))
    if median_interval <= 0.0:
        return None, 0.0
    relative_error = np.abs(intervals - median_interval) / median_interval
    consistency = float(np.mean(relative_error <= 0.12))
    bpm = 60.0 / median_interval
    if not MIN_BPM <= bpm <= MAX_BPM:
        return None, consistency
    return bpm, consistency


def detect_bpm(
    samples: np.ndarray,
    sample_rate: int,
    *,
    raise_on_low_confidence: bool = True,
) -> BpmResult:
    """Estimate quarter-note BPM and a normalized [0, 1] confidence."""

    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 2:
        mono = values.mean(axis=1)
    else:
        mono = values
    mono = np.asarray(mono, dtype=np.float64)
    max_analysis_frames = int(sample_rate * MAX_ANALYSIS_SECONDS)
    if mono.size > max_analysis_frames:
        mono = mono[:max_analysis_frames]
    if mono.size:
        mono = mono - float(mono.mean())

    envelope, frames_per_second = _onset_envelope(mono, sample_rate)
    autocorrelation = _autocorrelation(envelope)
    candidates = _tempo_candidates(autocorrelation, frames_per_second)
    if not candidates:
        confidence = 0.0
        bpm = 120.0
    else:
        best_score, best_bpm, _ = candidates[0]

        # Prefer the faster octave when the half-tempo interpretation wins only
        # narrowly. This common autocorrelation tie is especially visible near
        # the 220 BPM range boundary.
        double_candidates = [
            item
            for item in candidates
            if 1.97 * best_bpm <= item[1] <= 2.03 * best_bpm
            and item[0] >= best_score * 0.97
        ]
        if double_candidates:
            double_candidate = max(double_candidates, key=lambda item: item[0])
            best_score, best_bpm, _ = double_candidate

        interval_bpm, interval_consistency = _onset_interval_stats(
            envelope, frames_per_second
        )

        # Autocorrelation has an unavoidable half/double-tempo ambiguity.
        # When regular onset intervals indicate double the ACF result, prefer
        # the faster, onset-consistent interpretation provided it also has a
        # strong autocorrelation score.
        if (
            interval_bpm is not None
            and interval_consistency >= 0.55
            and best_bpm < interval_bpm * 0.65
        ):
            near_interval = [
                item
                for item in candidates
                if abs(item[1] - interval_bpm) <= max(0.5, interval_bpm * 0.02)
            ]
            if near_interval:
                interval_candidate = max(near_interval, key=lambda item: item[0])
                if interval_candidate[0] >= best_score * 0.72:
                    best_score, best_bpm, _ = interval_candidate

        bpm = best_bpm
        confidence = best_score * (0.75 + 0.25 * interval_consistency)
        confidence = float(np.clip(confidence, 0.0, 1.0))

    nearest_integer = float(round(float(bpm)))
    if abs(float(bpm) - nearest_integer) <= 0.30:
        bpm = nearest_integer
    else:
        bpm = round(float(bpm), 6)
    confidence = round(float(confidence), 6)
    beat_samples = round_half_up(60.0 * sample_rate / bpm)
    result = BpmResult(bpm=bpm, confidence=confidence, beat_samples=beat_samples)

    if raise_on_low_confidence and confidence < CONFIDENCE_THRESHOLD:
        raise LowConfidenceError(
            f"BPM confidence {confidence:.3f} is below "
            f"{CONFIDENCE_THRESHOLD:.2f}"
        )
    return result

