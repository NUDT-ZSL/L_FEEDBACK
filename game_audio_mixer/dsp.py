from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import lfilter, resample_poly


SAMPLE_RATE = 48000
SUPPORTED_SAMPLE_RATES = (
    8000,
    11025,
    12000,
    16000,
    22050,
    24000,
    32000,
    44100,
    48000,
    88200,
    96000,
    176400,
    192000,
)

# ITU-R BS.1770 K-weighting coefficients at 48 kHz.
_K_SHELF_B = np.array(
    [1.53512485958697, -2.69169618940638, 1.19839281085285], dtype=np.float64
)
_K_SHELF_A = np.array([1.0, -1.69065929318241, 0.73248077421585], dtype=np.float64)
_K_HIGHPASS_B = np.array([1.0, -2.0, 1.0], dtype=np.float64)
_K_HIGHPASS_A = np.array(
    [1.0, -1.99004745483398, 0.99007225036621], dtype=np.float64
)


def db_to_amplitude(value_db: float) -> float:
    return float(10.0 ** (value_db / 20.0))


def amplitude_to_db(value: float, floor_db: float = -240.0) -> Optional[float]:
    if not math.isfinite(value) or value <= 0.0:
        return None
    return max(floor_db, 20.0 * math.log10(value))


def resampling_ratio(source_rate: int, target_rate: int = SAMPLE_RATE) -> Tuple[int, int]:
    divisor = math.gcd(source_rate, target_rate)
    return target_rate // divisor, source_rate // divisor


def resample_audio(
    audio: np.ndarray, source_rate: int, target_rate: int = SAMPLE_RATE
) -> Tuple[np.ndarray, str, int, int]:
    up, down = resampling_ratio(source_rate, target_rate)
    if source_rate == target_rate:
        return (
            np.asarray(audio, dtype=np.float64),
            "none (source is already 48000 Hz)",
            up,
            down,
        )
    resampled = resample_poly(
        np.asarray(audio, dtype=np.float64),
        up,
        down,
        axis=0,
        window=("kaiser", 5.0),
        padtype="constant",
    )
    method = (
        "scipy.signal.resample_poly "
        f"up={up} down={down} window=kaiser(beta=5.0) padtype=constant"
    )
    return resampled, method, up, down


def k_weight(audio: np.ndarray) -> np.ndarray:
    weighted = lfilter(_K_SHELF_B, _K_SHELF_A, audio, axis=0)
    return lfilter(_K_HIGHPASS_B, _K_HIGHPASS_A, weighted, axis=0)


def _block_powers(
    weighted: np.ndarray, block_frames: int, hop_frames: int
) -> Tuple[np.ndarray, np.ndarray]:
    frame_count = weighted.shape[0]
    if frame_count == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    if frame_count < block_frames:
        power = float(np.mean(np.square(weighted, dtype=np.float64)))
        return np.array([0], dtype=np.int64), np.array([power], dtype=np.float64)
    starts = np.arange(0, frame_count - block_frames + 1, hop_frames, dtype=np.int64)
    if starts.size == 0:
        starts = np.array([0], dtype=np.int64)
    cumulative = np.concatenate(
        (
            np.zeros((1, weighted.shape[1]), dtype=np.float64),
            np.cumsum(np.square(weighted, dtype=np.float64), axis=0, dtype=np.float64),
        ),
        axis=0,
    )
    sums = cumulative[starts + block_frames] - cumulative[starts]
    powers = np.sum(sums / float(block_frames), axis=1)
    return starts, powers


def integrated_loudness(audio: np.ndarray) -> Optional[float]:
    """Return BS.1770 integrated loudness in LUFS, or None for silence."""
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] == 0:
        return None
    weighted = k_weight(values)
    block_frames = int(round(0.4 * SAMPLE_RATE))
    hop_frames = int(round(0.1 * SAMPLE_RATE))
    _, powers = _block_powers(weighted, block_frames, hop_frames)
    if powers.size == 0:
        return None

    with np.errstate(divide="ignore"):
        block_loudness = -0.691 + 10.0 * np.log10(powers)
    absolute_gated = powers[block_loudness > -70.0]
    if absolute_gated.size == 0 or float(np.mean(absolute_gated)) <= 0.0:
        return None

    absolute_loudness = -0.691 + 10.0 * math.log10(float(np.mean(absolute_gated)))
    relative_gate = absolute_loudness - 10.0
    with np.errstate(divide="ignore"):
        rel_loudness = -0.691 + 10.0 * np.log10(powers)
    relative_gated = powers[rel_loudness > relative_gate]
    if relative_gated.size == 0 or float(np.mean(relative_gated)) <= 0.0:
        return None
    return -0.691 + 10.0 * math.log10(float(np.mean(relative_gated)))


def short_term_loudness(audio: np.ndarray) -> List[Dict[str, Any]]:
    """Return non-overlapping-step (1 s) 3 s short-term loudness windows."""
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    frame_count = values.shape[0]
    if frame_count == 0:
        return []
    weighted = k_weight(values)
    full_window = int(round(3.0 * SAMPLE_RATE))
    hop = int(round(1.0 * SAMPLE_RATE))
    window = min(full_window, frame_count)
    if frame_count <= window:
        starts = [0]
    else:
        starts = list(range(0, frame_count - window + 1, hop))
        final_start = frame_count - window
        if starts[-1] != final_start:
            starts.append(final_start)
    segments: List[Dict[str, Any]] = []
    for start in starts:
        stop = start + window
        power = float(np.mean(np.square(weighted[start:stop], dtype=np.float64)))
        loudness: Optional[float]
        if power <= 0.0:
            loudness = None
        else:
            loudness = round(-0.691 + 10.0 * math.log10(power), 6)
        segments.append(
            {
                "start_s": round(start / SAMPLE_RATE, 6),
                "end_s": round(stop / SAMPLE_RATE, 6),
                "lufs": loudness,
            }
        )
    return segments


def true_peak(audio: np.ndarray, oversampling: int = 4) -> Tuple[float, int]:
    """Estimate true peak and return (linear peak, source-frame index)."""
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] == 0:
        return 0.0, 0
    upsampled = resample_poly(
        values,
        oversampling,
        1,
        axis=0,
        window=("kaiser", 5.0),
        padtype="constant",
    )
    magnitudes = np.max(np.abs(upsampled), axis=1)
    peak_index = int(np.argmax(magnitudes))
    peak = float(magnitudes[peak_index])
    return peak, min(values.shape[0] - 1, int(peak_index // oversampling))


@dataclass
class CompressorResult:
    audio: np.ndarray
    max_gain_reduction_db: float
    active_fraction: float


def compress_dynamic_range(
    audio: np.ndarray,
    threshold_db: float = -24.0,
    ratio: float = 2.5,
    knee_db: float = 6.0,
    attack_ms: float = 10.0,
    release_ms: float = 180.0,
) -> CompressorResult:
    """Stereo-linked, feed-forward soft-knee peak compressor."""
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] == 0:
        return CompressorResult(values.copy(), 0.0, 0.0)

    peak = np.max(np.abs(values), axis=1)
    with np.errstate(divide="ignore"):
        level_db = 20.0 * np.log10(np.maximum(peak, 1e-12))
    over = level_db - threshold_db
    target_reduction = np.zeros_like(level_db)
    if knee_db > 0.0:
        lower = -knee_db / 2.0
        upper = knee_db / 2.0
        knee_zone = (over > lower) & (over <= upper)
        target_reduction[knee_zone] = (1.0 / ratio - 1.0) * (
            (over[knee_zone] - lower) ** 2
        ) / (2.0 * knee_db)
        above = over > upper
        target_reduction[above] = (1.0 / ratio - 1.0) * over[above]
    else:
        above = over > 0.0
        target_reduction[above] = (1.0 / ratio - 1.0) * over[above]

    attack_coefficient = math.exp(-1.0 / max(1.0, attack_ms * 0.001 * SAMPLE_RATE))
    release_coefficient = math.exp(-1.0 / max(1.0, release_ms * 0.001 * SAMPLE_RATE))
    smoothed = np.empty_like(target_reduction)
    state = 0.0
    for index, target in enumerate(target_reduction):
        coefficient = attack_coefficient if target < state else release_coefficient
        state = coefficient * state + (1.0 - coefficient) * float(target)
        smoothed[index] = state

    gain = np.power(10.0, smoothed / 20.0)
    compressed = values * gain[:, None]
    max_reduction = abs(float(np.min(smoothed)))
    active_fraction = float(np.mean(smoothed < -0.01))
    return CompressorResult(compressed, max_reduction, active_fraction)


@dataclass
class LimiterResult:
    audio: np.ndarray
    enabled: bool
    pre_true_peak: float
    post_true_peak: float
    safety_gain_db: float
    reduction_mask: np.ndarray


def soft_limit(
    audio: np.ndarray,
    threshold_db: float = -1.0,
    ceiling_db: float = -0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth tanh limiter with continuous value and slope at threshold."""
    values = np.asarray(audio, dtype=np.float64)
    threshold = db_to_amplitude(threshold_db)
    ceiling = db_to_amplitude(ceiling_db)
    room = max(ceiling - threshold, 1e-9)
    magnitude = np.abs(values)
    mask = magnitude > threshold
    output = values.copy()
    output[mask] = np.sign(values[mask]) * (
        threshold + room * np.tanh((magnitude[mask] - threshold) / room)
    )
    reduction_mask = np.any(mask, axis=1)
    return output, reduction_mask


def enforce_true_peak(
    audio: np.ndarray,
    ceiling_db: float = -0.01,
    max_iterations: int = 4,
    lookahead_ms: float = 5.0,
    release_ms: float = 50.0,
) -> Tuple[np.ndarray, float, float]:
    """Apply a local look-ahead gain envelope for intersample peaks.

    A final global gain is used only if local iterations leave a residual
    overshoot. This avoids reducing the loudness of the whole program because
    of one isolated transient.
    """
    values = np.asarray(audio, dtype=np.float64)
    ceiling = db_to_amplitude(ceiling_db)
    maximum_reduction = 0.0
    for _ in range(max_iterations):
        upsampled = resample_poly(
            values, 4, 1, axis=0, window=("kaiser", 5.0), padtype="constant"
        )
        magnitudes = np.max(np.abs(upsampled), axis=1)
        desired = np.minimum(
            1.0, ceiling / np.maximum(magnitudes, 1e-12)
        )
        if float(np.min(desired)) >= 0.999999:
            break
        base_frames = values.shape[0]
        # Check a small neighborhood around each base-rate sample so the
        # polyphase filter group delay cannot move a peak outside the mask.
        neighborhood = max(3, int(round(0.25 * SAMPLE_RATE * 4 / 1000.0)) * 2 + 1)
        smoothed_up = minimum_filter1d(desired, size=neighborhood, mode="nearest")
        sample_indices = np.minimum(
            np.arange(base_frames) * 4 + 2, smoothed_up.size - 1
        )
        envelope = smoothed_up[sample_indices].copy()
        lookahead_frames = max(1, int(round(lookahead_ms * 0.001 * SAMPLE_RATE)))
        envelope = minimum_filter1d(
            envelope, size=lookahead_frames * 2 + 1, mode="nearest"
        )
        release_coefficient = math.exp(
            -1.0 / max(1.0, release_ms * 0.001 * SAMPLE_RATE)
        )
        state = 1.0
        for index in range(envelope.size):
            target = float(envelope[index])
            if target < state:
                state = target
            else:
                state = release_coefficient * state + (
                    1.0 - release_coefficient
                ) * target
            envelope[index] = state
        if float(np.min(envelope)) < 1.0:
            maximum_reduction = max(
                maximum_reduction,
                -20.0 * math.log10(max(float(np.min(envelope)), 1e-12)),
            )
        values = values * envelope[:, None]

    # Residual global trim is kept as a strict guarantee of the delivery cap.
    global_gain = 1.0
    while True:
        peak, _ = true_peak(values)
        if peak <= ceiling:
            break
        gain = ceiling / max(peak, 1e-12)
        values = values * gain
        global_gain *= gain
    if global_gain < 1.0:
        maximum_reduction = max(
            maximum_reduction, -20.0 * math.log10(max(global_gain, 1e-12))
        )
    post_peak, _ = true_peak(values)
    return values, post_peak, -maximum_reduction


def apply_limiter(
    audio: np.ndarray,
    threshold_db: float = -1.0,
    ceiling_db: float = -0.2,
    true_peak_ceiling_db: float = -0.01,
) -> LimiterResult:
    values = np.asarray(audio, dtype=np.float64)
    pre_peak, _ = true_peak(values)
    threshold = db_to_amplitude(threshold_db)
    enabled = pre_peak > threshold
    reduction_mask = np.zeros(values.shape[0], dtype=bool)
    safety_gain_db = 0.0
    if enabled:
        values, reduction_mask = soft_limit(values, threshold_db, ceiling_db)
        values, _, safety_gain_db = enforce_true_peak(
            values, ceiling_db=true_peak_ceiling_db
        )
    post_peak, _ = true_peak(values)
    return LimiterResult(
        audio=values,
        enabled=enabled,
        pre_true_peak=pre_peak,
        post_true_peak=post_peak,
        safety_gain_db=safety_gain_db,
        reduction_mask=reduction_mask,
    )


def quantize_pcm24(audio: np.ndarray) -> np.ndarray:
    """Quantize to the 24-bit grid, returned as limited float64 samples."""
    max_positive = 8388607.0
    quantized = np.rint(np.asarray(audio, dtype=np.float64) * max_positive)
    quantized = np.clip(quantized, -8388608.0, max_positive)
    return quantized / max_positive
