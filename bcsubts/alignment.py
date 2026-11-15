from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .subtitles import Cue


@dataclass
class AlignmentResult:
    offset_ms: float
    offset_samples: int
    coarse_offset_ms: float
    confidence: float
    residual_variance: float
    residual_rmse: float
    search_range_ms: int
    coarse_step_ms: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "estimated_offset_ms": round(self.offset_ms, 6),
            "offset_samples_48k": self.offset_samples,
            "coarse_offset_ms": round(self.coarse_offset_ms, 6),
            "confidence": round(self.confidence, 6),
            "residual_variance": round(self.residual_variance, 9),
            "residual_rmse": round(self.residual_rmse, 9),
            "search_range_ms": self.search_range_ms,
            "coarse_step_ms": self.coarse_step_ms,
            "fine_step_samples": 1,
        }


def _downmix(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 1:
        return array
    return np.mean(array, axis=1)


def _moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or values.size == 0:
        return values
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(values, kernel, mode="same")


def _frame_envelope(samples: np.ndarray, sample_rate: int = 48000, frame_ms: int = 10) -> np.ndarray:
    mono = _downmix(samples)
    frame_size = int(sample_rate * frame_ms / 1000)
    frame_count = mono.size // frame_size
    if frame_count == 0:
        return np.zeros(0, dtype=np.float64)
    frames = mono[: frame_count * frame_size].reshape(frame_count, frame_size)
    envelope = np.sqrt(np.mean(frames * frames, axis=1))
    return _moving_average(envelope, max(1, int(50 / frame_ms)))


def _pseudo_envelope(cues: Iterable[Cue], frame_count: int, sample_rate: int, frame_ms: int) -> np.ndarray:
    pseudo = np.zeros(frame_count, dtype=np.float64)
    for cue in cues:
        weight = 1.0 + 0.15 * math.log1p(max(0, len(cue.text)))
        start = max(0, int(round(cue.start_ms * sample_rate / 1000 / (sample_rate * frame_ms / 1000))))
        end = min(
            frame_count,
            int(round(cue.end_ms * sample_rate / 1000 / (sample_rate * frame_ms / 1000))),
        )
        if end > start:
            pseudo[start:end] = np.maximum(pseudo[start:end], weight)
    return pseudo


def _normalized_lag_score(audio: np.ndarray, pseudo: np.ndarray, lag: int) -> float:
    n = audio.size
    if lag >= 0:
        if n - lag <= 0:
            return 0.0
        a = audio[lag:]
        p = pseudo[: n - lag]
    else:
        if n + lag <= 0:
            return 0.0
        a = audio[: n + lag]
        p = pseudo[-lag:]
    if a.size == 0:
        return 0.0
    numerator = float(np.dot(a, p))
    denominator = math.sqrt(float(np.dot(a, a)) * float(np.dot(p, p)))
    return numerator / denominator if denominator > 1e-15 else 0.0


def _coarse_search(
    audio: np.ndarray,
    pseudo: np.ndarray,
    max_lag_frames: int,
) -> tuple[int, float]:
    n = min(audio.size, pseudo.size)
    if n == 0:
        return 0, 0.0
    max_lag_frames = min(max_lag_frames, max(0, n - 1))
    audio = audio[:n]
    pseudo = pseudo[:n]
    best_lag = 0
    best_score = -1.0
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        score = _normalized_lag_score(audio, pseudo, lag)
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag, best_score


def _sample_energy_prefix(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = np.abs(_downmix(samples))
    # A short average suppresses individual waveform zero crossings.
    smoothed = _moving_average(mono, max(1, sample_rate // 1000))
    return np.concatenate(([0.0], np.cumsum(smoothed)))


def _fine_search(
    cues: Iterable[Cue],
    energy_prefix: np.ndarray,
    sample_rate: int,
    coarse_samples: int,
    max_refine_samples: int,
) -> tuple[int, float]:
    n_samples = energy_prefix.size - 1
    intervals = [
        (
            max(0, int(round(cue.start_ms * sample_rate / 1000))),
            min(n_samples, int(round(cue.end_ms * sample_rate / 1000))),
            1.0 + 0.15 * math.log1p(max(0, len(cue.text))),
        )
        for cue in cues
    ]
    intervals = [(s, e, w) for s, e, w in intervals if e > s]
    if not intervals:
        return coarse_samples, 0.0

    best_lag = coarse_samples
    best_score = -1.0
    low = max(-2000 * sample_rate // 1000, coarse_samples - max_refine_samples)
    high = min(2000 * sample_rate // 1000, coarse_samples + max_refine_samples)
    for lag in range(low, high + 1):
        score = 0.0
        total_weight = 0.0
        for start, end, weight in intervals:
            shifted_start = max(0, min(n_samples, start + lag))
            shifted_end = max(0, min(n_samples, end + lag))
            if shifted_end > shifted_start:
                score += weight * (
                    energy_prefix[shifted_end] - energy_prefix[shifted_start]
                ) / (shifted_end - shifted_start)
                total_weight += weight
        if total_weight > 0:
            score /= total_weight
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag, best_score


def estimate_offset(
    samples: np.ndarray,
    cues: Iterable[Cue],
    sample_rate: int = 48000,
    search_ms: int = 2000,
    coarse_step_ms: int = 10,
) -> AlignmentResult:
    cue_list = list(cues)
    frame_ms = coarse_step_ms
    envelope = _frame_envelope(samples, sample_rate, frame_ms)
    pseudo = _pseudo_envelope(cue_list, envelope.size, sample_rate, frame_ms)
    max_lag_frames = int(round(search_ms / coarse_step_ms))
    coarse_lag, coarse_confidence = _coarse_search(envelope, pseudo, max_lag_frames)
    coarse_samples = coarse_lag * int(sample_rate * coarse_step_ms / 1000)
    energy_prefix = _sample_energy_prefix(samples, sample_rate)
    fine_samples, _ = _fine_search(
        cue_list,
        energy_prefix,
        sample_rate,
        coarse_samples,
        int(sample_rate * coarse_step_ms / 1000),
    )
    offset_ms = fine_samples * 1000.0 / sample_rate

    # Residual after applying the sample-level offset to the pseudo VAD.
    offset_frames = int(round(offset_ms / coarse_step_ms))
    aligned = np.zeros_like(pseudo)
    if offset_frames >= 0:
        if offset_frames < pseudo.size:
            aligned[offset_frames:] = pseudo[: pseudo.size - offset_frames]
    else:
        aligned[: pseudo.size + offset_frames] = pseudo[-offset_frames:]
    audio_norm = envelope / envelope.max() if envelope.size and envelope.max() > 0 else envelope
    pseudo_norm = aligned / aligned.max() if aligned.size and aligned.max() > 0 else aligned
    residual = audio_norm - pseudo_norm if audio_norm.size else np.zeros(0)
    residual_variance = float(np.var(residual)) if residual.size else 0.0
    residual_rmse = float(np.sqrt(np.mean(residual * residual))) if residual.size else 0.0

    # Confidence combines normalized correlation with residual agreement.
    confidence = max(0.0, min(1.0, coarse_confidence * (1.0 - min(1.0, residual_rmse))))
    return AlignmentResult(
        offset_ms=offset_ms,
        offset_samples=fine_samples,
        coarse_offset_ms=coarse_samples * 1000.0 / sample_rate,
        confidence=confidence,
        residual_variance=residual_variance,
        residual_rmse=residual_rmse,
        search_range_ms=search_ms,
        coarse_step_ms=coarse_step_ms,
    )
