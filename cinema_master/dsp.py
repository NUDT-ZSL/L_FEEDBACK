"""Deterministic NumPy DSP primitives for the mastering pipeline."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .layouts import channel_weights


EPS = 1.0e-20


def resample_audio(
    data: np.ndarray, source_rate: int, target_rate: int
) -> Tuple[np.ndarray, int]:
    """Band-limited FFT resampling with no added leading/trailing delay."""

    if data.ndim != 2:
        raise ValueError("resample_audio expects (frames, channels)")
    source_frames, channels = data.shape
    if source_frames == 0:
        return np.zeros((0, channels), dtype=np.float64), 0
    target_frames = int(round(source_frames * float(target_rate) / source_rate))
    target_frames = max(1, target_frames)
    if source_rate == target_rate:
        return np.ascontiguousarray(data.astype(np.float64, copy=True)), target_frames

    pad_in = min(max(0, source_frames - 1), int(round(0.02 * source_rate)))
    if pad_in:
        padded = np.pad(data, ((pad_in, pad_in), (0, 0)), mode="reflect")
    else:
        padded = data
    padded_frames = padded.shape[0]
    padded_target = max(1, int(round(padded_frames * float(target_rate) / source_rate)))
    destination_bins = padded_target // 2 + 1

    output = np.empty((target_frames, channels), dtype=np.float64)
    pad_out = int(round(pad_in * float(target_rate) / source_rate))
    scale = float(padded_target) / float(padded_frames)
    for channel in range(channels):
        spectrum = np.fft.rfft(padded[:, channel])
        destination = np.zeros(destination_bins, dtype=np.complex128)
        bins = min(spectrum.shape[0], destination_bins)
        destination[:bins] = spectrum[:bins]
        resampled = np.fft.irfft(destination, n=padded_target) * scale
        start = pad_out
        end = start + target_frames
        if end <= resampled.shape[0]:
            output[:, channel] = resampled[start:end]
        else:
            available = max(0, resampled.shape[0] - start)
            output[:available, channel] = resampled[start : start + available]
            output[available:, channel] = 0.0
    return output, target_frames


def _k_weighting_coefficients(sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    # Coefficients follow the ITU-R BS.1770 K-weighting filter design.
    f0 = 1681.974450955533
    gain_db = 3.999843853973347
    quality = 0.7071752369554196
    k = math.tan(math.pi * f0 / sample_rate)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / quality + k * k
    pre_b = np.array(
        [
            (vh + vb * k / quality + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / quality + k * k) / a0,
        ],
        dtype=np.float64,
    )
    pre_a = np.array(
        [
            1.0,
            2.0 * (k * k - 1.0) / a0,
            (1.0 - k / quality + k * k) / a0,
        ],
        dtype=np.float64,
    )

    f0 = 38.13547087602444
    quality = 0.5003270373238773
    k = math.tan(math.pi * f0 / sample_rate)
    a0 = 1.0 + k / quality + k * k
    rlb_b = np.array([1.0, -2.0, 1.0], dtype=np.float64) / a0
    rlb_a = np.array(
        [
            1.0,
            2.0 * (k * k - 1.0) / a0,
            (1.0 - k / quality + k * k) / a0,
        ],
        dtype=np.float64,
    )
    return np.vstack([pre_b, rlb_b]), np.vstack([pre_a, rlb_a])


def _biquad_response(
    b: np.ndarray, a: np.ndarray, omega: np.ndarray
) -> np.ndarray:
    z1 = np.exp(-1j * omega)
    z2 = z1 * z1
    return (b[0] + b[1] * z1 + b[2] * z2) / (
        a[0] + a[1] * z1 + a[2] * z2
    )


def k_weight(data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply K-weighting in the frequency domain to an offline block."""

    if data.ndim != 2:
        raise ValueError("k_weight expects (frames, channels)")
    frames = data.shape[0]
    if frames == 0:
        return data.copy()
    nfft = 1 << max(0, (frames - 1).bit_length())
    omega = 2.0 * math.pi * np.arange(nfft // 2 + 1, dtype=np.float64) / nfft
    b, a = _k_weighting_coefficients(sample_rate)
    response = _biquad_response(b[0], a[0], omega) * _biquad_response(
        b[1], a[1], omega
    )
    output = np.empty_like(data, dtype=np.float64)
    for channel in range(data.shape[1]):
        spectrum = np.fft.rfft(data[:, channel], n=nfft)
        output[:, channel] = np.fft.irfft(spectrum * response, n=nfft)[:frames]
    return output


def _block_energies(
    squared_weighted: np.ndarray, sample_rate: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = squared_weighted.shape[0]
    block = max(1, int(round(0.4 * sample_rate)))
    hop = max(1, int(round(0.1 * sample_rate)))
    if frames <= block:
        starts = np.array([0], dtype=np.int64)
        lengths = np.array([frames], dtype=np.int64)
    else:
        starts = np.arange(0, frames, hop, dtype=np.int64)
        lengths = np.minimum(block, frames - starts)
    cumulative = np.concatenate(([0.0], np.cumsum(squared_weighted)))
    sums = cumulative[starts + lengths] - cumulative[starts]
    energies = sums / np.maximum(lengths, 1)
    return energies, starts, lengths


def integrated_loudness(
    data: np.ndarray,
    sample_rate: int,
    roles: Sequence[str],
) -> float:
    """ITU-R BS.1770-4 integrated loudness with absolute and relative gating."""

    if data.shape[0] == 0:
        return float("-inf")
    weights = np.asarray(channel_weights(roles), dtype=np.float64)
    filtered = k_weight(data, sample_rate)
    squared = filtered * filtered
    squared_weighted = np.sum(squared * weights[np.newaxis, :], axis=1)
    energies, _, _ = _block_energies(squared_weighted, sample_rate)
    loudness = -0.691 + 10.0 * np.log10(np.maximum(energies, EPS))
    absolute = loudness >= -70.0
    if not np.any(absolute):
        return float("-inf")
    absolute_energy = float(np.mean(energies[absolute]))
    relative_threshold = -0.691 + 10.0 * math.log10(
        max(absolute_energy, EPS)
    ) - 10.0
    gated = absolute & (loudness >= relative_threshold)
    if not np.any(gated):
        return float("-inf")
    return -0.691 + 10.0 * math.log10(max(float(np.mean(energies[gated])), EPS))


def detect_dialogue_activity(
    dialogue: np.ndarray,
    sample_rate: int,
    frame_ms: float,
    overlap: float,
    threshold_dbfs: float,
    min_segment_ms: float,
) -> Dict[str, Any]:
    frames = dialogue.shape[0]
    if frames == 0:
        return {
            "segments": [],
            "frame_samples": 0,
            "hop_samples": 0,
            "frame_count": 0,
            "active_frame_count": 0,
            "energies_dbfs": np.array([], dtype=np.float64),
        }
    frame_samples = max(1, int(round(frame_ms * sample_rate / 1000.0)))
    hop_samples = max(1, int(round(frame_samples * (1.0 - overlap))))
    starts = np.arange(0, frames, hop_samples, dtype=np.int64)
    lengths = np.minimum(frame_samples, frames - starts)
    power = np.mean(dialogue * dialogue, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    sums = cumulative[starts + lengths] - cumulative[starts]
    mean_power = sums / np.maximum(lengths, 1)
    energies = 10.0 * np.log10(np.maximum(mean_power, EPS))
    active = energies >= threshold_dbfs

    raw_segments: List[Tuple[int, int, int, int]] = []
    index = 0
    while index < active.shape[0]:
        if not active[index]:
            index += 1
            continue
        first = index
        while index + 1 < active.shape[0] and active[index + 1]:
            index += 1
        last = index
        start_sample = int(starts[first])
        end_sample = int(min(frames, starts[last] + lengths[last]))
        raw_segments.append((start_sample, end_sample, first, last))
        index += 1

    minimum_samples = int(round(min_segment_ms * sample_rate / 1000.0))
    kept = [
        item
        for item in raw_segments
        if item[1] - item[0] >= minimum_samples
    ]
    segments = []
    for segment_index, (start, end, first, last) in enumerate(kept):
        segments.append(
            {
                "index": segment_index,
                "start_sample": start,
                "end_sample": end,
                "start_sec": round(start / sample_rate, 9),
                "end_sec": round(end / sample_rate, 9),
                "duration_sec": round((end - start) / sample_rate, 9),
                "first_active_frame": int(first),
                "last_active_frame": int(last),
            }
        )
    return {
        "segments": segments,
        "frame_samples": frame_samples,
        "hop_samples": hop_samples,
        "frame_count": int(starts.shape[0]),
        "active_frame_count": int(np.count_nonzero(active)),
        "energies_dbfs": energies,
    }


def active_mask_from_segments(segments: Sequence[Dict[str, Any]], frames: int) -> np.ndarray:
    mask = np.zeros(frames, dtype=bool)
    for segment in segments:
        start = max(0, int(segment["start_sample"]))
        end = min(frames, int(segment["end_sample"]))
        if end > start:
            mask[start:end] = True
    return mask


def soft_knee_gain_db(
    level_db: np.ndarray,
    threshold_db: float,
    ratio: float,
    knee_db: float,
) -> np.ndarray:
    slope = 1.0 / ratio - 1.0
    gain = np.zeros_like(level_db, dtype=np.float64)
    if knee_db <= 0.0:
        above = level_db > threshold_db
        gain[above] = slope * (level_db[above] - threshold_db)
        return gain
    lower = threshold_db - knee_db / 2.0
    upper = threshold_db + knee_db / 2.0
    below = level_db <= lower
    above = level_db >= upper
    middle = ~(below | above)
    gain[above] = slope * (level_db[above] - threshold_db)
    offset = level_db[middle] - lower
    gain[middle] = slope * offset * offset / (2.0 * knee_db)
    return gain


def _one_pole_coefficient(time_ms: float, sample_rate: int) -> float:
    return 1.0 - math.exp(-1.0 / max(1.0, time_ms * 0.001 * sample_rate))


def _smooth_envelope(
    signal: np.ndarray, sample_rate: int, attack_ms: float, release_ms: float
) -> np.ndarray:
    attack = _one_pole_coefficient(attack_ms, sample_rate)
    release = _one_pole_coefficient(release_ms, sample_rate)
    output = np.empty(signal.shape[0], dtype=np.float64)
    value = 0.0
    for index in range(signal.shape[0]):
        sample = float(signal[index])
        coefficient = attack if sample > value else release
        value += coefficient * (sample - value)
        output[index] = value
    return output


def compressor_gain_db(
    dialogue: np.ndarray,
    active_mask: np.ndarray,
    sample_rate: int,
    threshold_dbfs: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    knee_db: float,
    enabled: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    frames = dialogue.shape[0]
    if frames == 0:
        return np.zeros(0, dtype=np.float64), {
            "enabled": enabled,
            "max_reduction_db": 0.0,
            "mean_reduction_active_db": 0.0,
        }
    if not enabled:
        return np.zeros(frames, dtype=np.float64), {
            "enabled": False,
            "max_reduction_db": 0.0,
            "mean_reduction_active_db": 0.0,
            "applied_samples": 0,
        }
    detector = np.max(np.abs(dialogue), axis=1)
    envelope = _smooth_envelope(detector, sample_rate, attack_ms, release_ms)
    level_db = 20.0 * np.log10(np.maximum(envelope, EPS))
    target_gain = soft_knee_gain_db(
        level_db, threshold_dbfs, ratio, knee_db
    )
    target_gain = np.where(active_mask, target_gain, 0.0)
    # The gain control itself gets attack/release smoothing to prevent clicks.
    gain = _smooth_envelope(target_gain, sample_rate, attack_ms, release_ms)
    gain = np.minimum(gain, 0.0)

    active_reduction = -gain[active_mask]
    stats = {
        "enabled": True,
        "max_reduction_db": round(
            max(0.0, float(np.max(-gain)) if frames else 0.0), 6
        ),
        "mean_reduction_active_db": round(
            float(np.mean(active_reduction)) if active_reduction.size else 0.0, 6
        ),
        "applied_samples": int(active_mask.sum()),
        "detector": "linked_dialogue_peak",
    }
    return gain, stats


def _oversampled_linked_peak(
    data: np.ndarray, sample_rate: int, factor: int
) -> np.ndarray:
    frames = data.shape[0]
    high = np.zeros(frames * factor, dtype=np.float64)
    for channel in range(data.shape[1]):
        resampled, _ = resample_audio(
            data[:, channel : channel + 1], sample_rate, sample_rate * factor
        )
        high = np.maximum(high, np.abs(resampled[:, 0]))
    return high


def estimate_true_peak(
    data: np.ndarray, sample_rate: int, factor: int = 4
) -> Tuple[float, float]:
    if data.size == 0:
        return 0.0, float("-inf")
    high_peak = _oversampled_linked_peak(data, sample_rate, factor)
    peak = float(np.max(high_peak))
    if peak <= EPS:
        return 0.0, float("-inf")
    return peak, 20.0 * math.log10(peak)


def _rolling_future_min(values: np.ndarray, width: int) -> np.ndarray:
    count = values.shape[0]
    result = np.empty(count, dtype=np.float64)
    candidates: deque = deque()
    for index in range(count - 1, -1, -1):
        value = float(values[index])
        while candidates and float(values[candidates[-1]]) >= value:
            candidates.pop()
        candidates.append(index)
        limit = index + width
        while candidates and candidates[0] > limit:
            candidates.popleft()
        result[index] = values[candidates[0]]
    return result


def apply_lookahead_limiter(
    mix: np.ndarray,
    sample_rate: int,
    ceiling_dbtp: float,
    lookahead_ms: float,
    release_ms: float,
    oversample: int,
    safety_margin_db: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Sample-linked lookahead limiter driven by oversampled true peak."""

    frames = mix.shape[0]
    ceiling = 10.0 ** (ceiling_dbtp / 20.0)
    safe_ceiling = 10.0 ** ((ceiling_dbtp - safety_margin_db) / 20.0)
    high_peak = _oversampled_linked_peak(mix, sample_rate, oversample)
    required_high = np.minimum(1.0, safe_ceiling / np.maximum(high_peak, EPS))
    usable = (frames * oversample // oversample) * oversample
    required_block = required_high[:usable].reshape(frames, oversample).min(axis=1)
    lookahead_frames = max(1, int(round(lookahead_ms * sample_rate / 1000.0)))
    target = _rolling_future_min(required_block, lookahead_frames + 1)

    attack_ms = min(2.0, max(0.2, lookahead_ms * 0.25))
    attack = _one_pole_coefficient(attack_ms, sample_rate)
    release = _one_pole_coefficient(release_ms, sample_rate)
    gain = np.ones(frames, dtype=np.float64)
    value = 1.0
    for index in range(frames):
        goal = float(target[index])
        coefficient = attack if goal < value else release
        value += coefficient * (goal - value)
        gain[index] = value

    limited = mix * gain[:, np.newaxis]
    peak, peak_db = estimate_true_peak(limited, sample_rate, oversample)
    trim = 1.0
    if peak > ceiling:
        trim = (ceiling * 0.999999) / peak
        gain *= trim
        limited = mix * gain[:, np.newaxis]
        peak, peak_db = estimate_true_peak(limited, sample_rate, oversample)

    reduction_db = -20.0 * np.log10(np.maximum(gain, EPS))
    stats = {
        "triggered": bool(np.min(gain) < 0.9999999),
        "max_reduction_db": round(max(0.0, float(np.max(reduction_db))), 6),
        "lookahead_samples": lookahead_frames,
        "oversample": oversample,
        "safety_margin_db": safety_margin_db,
        "safety_trim_db": round(float(-20.0 * math.log10(trim)), 6),
        "measured_true_peak_dbtp": round(float(peak_db), 6),
        "ceiling_dbtp": ceiling_dbtp,
    }
    return limited, gain, stats


def apply_gain_db(data: np.ndarray, gain_db: float) -> np.ndarray:
    if gain_db == 0.0:
        return data
    return data * (10.0 ** (gain_db / 20.0))


def audio_duration(frames: int, sample_rate: int) -> float:
    return frames / float(sample_rate)


def db_to_linear(value_db: float) -> float:
    return 10.0 ** (value_db / 20.0)


def linear_to_db(value: float) -> float:
    return 20.0 * math.log10(max(value, EPS))

