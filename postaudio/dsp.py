"""Deterministic NumPy DSP primitives implemented for this project."""

from __future__ import annotations

from fractions import Fraction
import math
from typing import Any

import numpy as np

from .util import linear_to_db, round_half_even


def _blackman_harris(index: np.ndarray, length: int) -> np.ndarray:
    if length <= 1:
        return np.ones_like(index, dtype=np.float64)
    phase = 2.0 * np.pi * index / float(length - 1)
    return (
        0.35875
        - 0.48829 * np.cos(phase)
        + 0.14128 * np.cos(2.0 * phase)
        - 0.01168 * np.cos(3.0 * phase)
    )


def _design_polyphase(
    source_rate: int,
    target_rate: int,
    half_width: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    divisor = math.gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    offsets = np.arange(-half_width + 1, half_width + 1, dtype=np.float64)
    cutoff = 0.5 * min(1.0, target_rate / source_rate)
    kernels = np.empty((up, offsets.size), dtype=np.float64)
    for phase in range(up):
        tau = float(phase) / float(up)
        # A Blackman-Harris windowed sinc evaluated at the exact phase offset.
        sinc_arg = 2.0 * cutoff * (tau + offsets)
        sinc = np.ones_like(sinc_arg, dtype=np.float64)
        nonzero = np.abs(sinc_arg) >= 1e-15
        sinc[nonzero] = np.sin(np.pi * sinc_arg[nonzero]) / (np.pi * sinc_arg[nonzero])
        window = _blackman_harris(offsets + half_width, 2 * half_width)
        kernel = 2.0 * cutoff * sinc * window
        total = float(np.sum(kernel, dtype=np.float64))
        if abs(total) > 1e-15:
            kernel /= total
        kernels[phase] = kernel
    return kernels, offsets, up, down


def resample_polyphase(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
    half_width: int = 64,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rational polyphase FIR resampling with zero-padded boundaries."""
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("samples must be 2-D")
    if source_rate == target_rate:
        return data.copy(), {
            "resampled": False,
            "rational_ratio": "1/1",
            "rational_approximation": False,
            "filter_taps": 0,
            "filter_half_width": 0,
        }
    if target_rate <= 0 or source_rate <= 0:
        raise ValueError("sample rates must be positive")
    kernels, offsets, up, down = _design_polyphase(source_rate, target_rate, half_width)
    input_frames = int(data.shape[0])
    output_frames = round_half_even(Fraction(input_frames * target_rate, source_rate))
    channels = int(data.shape[1])
    padded = np.zeros((input_frames + 2 * half_width, channels), dtype=np.float64)
    padded[half_width : half_width + input_frames, :] = data
    output = np.empty((output_frames, channels), dtype=np.float64)
    taps = offsets.size
    j = offsets.astype(np.int64)
    for start in range(0, output_frames, chunk_size):
        stop = min(output_frames, start + chunk_size)
        n = np.arange(start, stop, dtype=np.int64)
        positions = n * down
        q, phase = np.divmod(positions, up)
        indices = q[:, None] - j[None, :] + half_width
        gathered = padded[indices, :]
        weights = kernels[phase, :]
        output[start:stop, :] = np.sum(gathered * weights[:, :, None], axis=1, dtype=np.float64)
    stats = {
        "resampled": True,
        "rational_ratio": f"{up}/{down}",
        "rational_approximation": False,
        "approximation_kind": "exact reduced integer ratio",
        "filter_taps": int(taps),
        "filter_half_width": int(half_width),
        "window": "blackman-harris",
        "cutoff_normalized": float(0.5 * min(1.0, target_rate / source_rate)),
    }
    return output, stats


def make_sync_template(sample_rate: int, sync: dict[str, Any]) -> np.ndarray:
    frequency = float(sync.get("frequency_hz", 2000.0))
    cycles = int(sync.get("cycles", 8))
    amplitude = float(sync.get("amplitude", 0.35))
    count = max(2, round_half_even(Fraction(cycles * sample_rate, int(frequency))))
    n = np.arange(count, dtype=np.float64)
    phase = 2.0 * np.pi * frequency * n / float(sample_rate)
    envelope = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(count, dtype=np.float64) / float(count - 1))
    return (amplitude * np.sin(phase) * envelope).astype(np.float64)


def detect_sync_marker(
    samples: np.ndarray,
    sample_rate: int,
    expected_position: int,
    sync: dict[str, Any] | None,
    default_threshold: int,
) -> dict[str, Any]:
    """Locate a known sync pulse using normalized cross-correlation."""
    sync = sync or {}
    data = np.asarray(samples, dtype=np.float64)
    template = make_sync_template(sample_rate, sync)
    count = template.size
    total = int(data.shape[0])
    template_energy = float(np.sum(template * template))
    if total < count or template_energy <= 1e-20:
        return {
            "detected": False,
            "actual_position": None,
            "offset_samples": 0,
            "threshold_samples": int(sync.get("threshold_samples", default_threshold)),
            "correlation": 0.0,
            "search_start": 0,
            "search_end": 0,
        }
    threshold = int(sync.get("threshold_samples", default_threshold))
    radius = int(sync.get("search_radius_samples", total))
    search_start = max(0, int(expected_position) - radius)
    search_end = min(total - count + 1, int(expected_position) + radius + 1)
    if search_end <= search_start:
        search_start, search_end = 0, total - count + 1
    detection_threshold = float(sync.get("detection_threshold", 0.35))
    candidates = [np.mean(data, axis=1)]
    candidates.extend(data[:, channel] for channel in range(data.shape[1]))
    best_peak = 0.0
    best_index = 0
    best_channel = 0
    for channel, signal in enumerate(candidates):
        segment = signal[search_start : search_end + count - 1]
        corr = np.correlate(segment, template, mode="valid").astype(np.float64)
        squared = signal * signal
        cumulative = np.concatenate(([0.0], np.cumsum(squared, dtype=np.float64)))
        energy = cumulative[
            search_start + count : search_start + count + corr.size
        ] - cumulative[search_start : search_start + corr.size]
        denom = np.sqrt(np.maximum(energy, 0.0) * template_energy)
        normalized = np.divide(corr, denom, out=np.zeros_like(corr), where=denom > 1e-20)
        peak_index = int(np.argmax(np.abs(normalized)))
        peak_value = float(normalized[peak_index])
        if abs(peak_value) > abs(best_peak):
            best_peak = peak_value
            best_index = peak_index
            best_channel = channel
    detected = abs(best_peak) >= detection_threshold
    peak_value = best_peak
    peak_index = best_index
    actual = search_start + peak_index if detected else None
    offset = (actual - int(expected_position)) if detected else 0
    return {
        "detected": detected,
        "actual_position": actual,
        "offset_samples": int(offset),
        "threshold_samples": threshold,
        "correlation": peak_value,
        "detection_channel": int(best_channel),
        "search_start": int(search_start),
        "search_end": int(search_end),
    }


def channel_layout(
    samples: np.ndarray, target_channels: int
) -> tuple[np.ndarray, bool]:
    """Convert mono/stereo source layout to the requested output layout."""
    data = np.asarray(samples, dtype=np.float64)
    source_channels = int(data.shape[1])
    if source_channels == target_channels:
        return data.copy(), False
    if source_channels == 2 and target_channels == 1:
        # -3 dB power-preserving downmix for uncorrelated material.
        mono = (data[:, 0] + data[:, 1]) / math.sqrt(2.0)
        return mono[:, None], True
    if source_channels == 1 and target_channels == 2:
        return np.repeat(data, 2, axis=1), False
    raise ValueError(f"cannot convert {source_channels} channels to {target_channels}")


def apply_lookahead_limiter(
    samples: np.ndarray, limit_linear: float, lookahead_samples: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Zero-latency look-ahead peak limiter.

    The gain at sample n is the minimum required gain over [n, n+look-ahead).
    This pulls future attenuation back in time; output stays on the source
    timeline, so the configured look-ahead is fully delay-compensated.
    """
    data = np.asarray(samples, dtype=np.float64)
    magnitude = np.abs(data)
    pre_peak = float(np.max(magnitude)) if magnitude.size else 0.0
    limited = int(np.count_nonzero(magnitude > limit_linear))
    if limited == 0 or pre_peak <= 0.0:
        return data.copy(), {
            "bypassed": True,
            "lookahead_samples": int(lookahead_samples),
            "delay_compensation_samples": int(lookahead_samples),
            "limited_samples": 0,
            "max_attenuation_db": 0.0,
            "pre_peak_linear": pre_peak,
            "post_peak_linear": pre_peak,
        }
    target = np.ones_like(magnitude, dtype=np.float64)
    np.divide(limit_linear, magnitude, out=target, where=magnitude > limit_linear)
    window = max(1, int(lookahead_samples))
    padded = np.ones((magnitude.shape[0] + window, magnitude.shape[1]), dtype=np.float64)
    padded[: magnitude.shape[0], :] = target
    min_gain = np.ones_like(target, dtype=np.float64)
    # Chunked future-window minimum to avoid a full sliding-view allocation.
    for start in range(0, target.shape[0], 8192):
        stop = min(target.shape[0], start + 8192)
        block = target[start:stop, :]
        future = padded[start : stop + window, :]
        # Cumulative minima are intentionally not used: the deterministic
        # reduction below has a fixed window and order.
        for offset in range(window):
            min_gain[start:stop, :] = np.minimum(min_gain[start:stop, :], future[offset : offset + block.shape[0], :])
    output = data * min_gain
    # A final deterministic safety pass covers multiplication roundoff only.
    post_magnitude = np.abs(output)
    overs = post_magnitude > limit_linear
    if np.any(overs):
        output = np.where(overs, np.sign(output) * limit_linear, output)
    min_used = float(np.min(min_gain)) if min_gain.size else 1.0
    return output, {
        "bypassed": False,
        "lookahead_samples": int(lookahead_samples),
        "delay_compensation_samples": int(lookahead_samples),
        "limited_samples": limited,
        "max_attenuation_db": float(max(0.0, -linear_to_db(min_used))),
        "pre_peak_linear": pre_peak,
        "post_peak_linear": float(np.max(np.abs(output))) if output.size else 0.0,
    }


def design_k_weighting(sample_rate: int) -> dict[str, Any]:
    """Derive BS.1770-4 K-weighting biquads by bilinear transform."""
    fs = float(sample_rate)
    # Stage 1: 4 dB high shelf at 1681 Hz, Q = 1/sqrt(2).
    shelf_f = 1681.9745
    shelf_q = 0.70717525
    shelf_gain = 3.9998438
    k = math.tan(math.pi * shelf_f / fs)
    vh = 10.0 ** (shelf_gain / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / shelf_q + k * k
    shelf_b = np.array(
        [
            (vh + vb * k / shelf_q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / shelf_q + k * k) / a0,
        ],
        dtype=np.float64,
    )
    shelf_a = np.array(
        [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / shelf_q + k * k) / a0],
        dtype=np.float64,
    )
    # Stage 2: RLB high-pass at 38 Hz, Q = 0.5.
    highpass_f = 38.13547
    highpass_q = 0.50032705
    k = math.tan(math.pi * highpass_f / fs)
    a0 = 1.0 + k / highpass_q + k * k
    highpass_b = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    highpass_a = np.array(
        [1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / highpass_q + k * k) / a0],
        dtype=np.float64,
    )
    return {
        "stage_1": {"b": shelf_b, "a": shelf_a},
        "stage_2": {"b": highpass_b, "a": highpass_a},
        "sample_rate": sample_rate,
    }


def _biquad(signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Transposed direct-form II biquad in deterministic float64 order."""
    output = np.empty_like(signal, dtype=np.float64)
    z1 = 0.0
    z2 = 0.0
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    a1, a2 = float(a[1]), float(a[2])
    for index in range(signal.size):
        value = float(signal[index])
        result = b0 * value + z1
        z1 = b1 * value - a1 * result + z2
        z2 = b2 * value - a2 * result
        output[index] = result
    return output


def k_weight(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    coefficients = design_k_weighting(sample_rate)
    data = np.asarray(samples, dtype=np.float64)
    weighted = np.empty_like(data)
    for channel in range(data.shape[1]):
        stage = _biquad(data[:, channel], coefficients["stage_1"]["b"], coefficients["stage_1"]["a"])
        weighted[:, channel] = _biquad(stage, coefficients["stage_2"]["b"], coefficients["stage_2"]["a"])
    return weighted


def integrated_loudness(samples: np.ndarray, sample_rate: int) -> float | None:
    """ITU-R BS.1770 integrated loudness with 400 ms / 75% overlap gating."""
    data = np.asarray(samples, dtype=np.float64)
    if data.shape[0] == 0:
        return None
    weighted = k_weight(data, sample_rate)
    block_frames = round_half_even(Fraction(sample_rate * 4, 10))
    hop_frames = round_half_even(Fraction(sample_rate, 10))
    if block_frames <= 0 or hop_frames <= 0 or weighted.shape[0] < block_frames:
        return None
    powers = []
    for start in range(0, weighted.shape[0] - block_frames + 1, hop_frames):
        block = weighted[start : start + block_frames, :]
        channel_power = np.mean(block * block, axis=0, dtype=np.float64)
        powers.append(float(np.sum(channel_power, dtype=np.float64)))
    z = np.asarray(powers, dtype=np.float64)
    if z.size == 0:
        return None
    block_loudness = -0.691 + 10.0 * np.log10(np.maximum(z, 1e-30))
    absolute_gate = block_loudness >= -70.0
    if not np.any(absolute_gate):
        return None
    absolute_mean = float(np.mean(z[absolute_gate], dtype=np.float64))
    relative_gate_loudness = -0.691 + 10.0 * math.log10(max(absolute_mean, 1e-30)) - 10.0
    gated = absolute_gate & (block_loudness >= relative_gate_loudness)
    if not np.any(gated):
        return None
    gated_mean = float(np.mean(z[gated], dtype=np.float64))
    return float(-0.691 + 10.0 * math.log10(max(gated_mean, 1e-30)))


def limiter_lookahead_samples(milliseconds: float, sample_rate: int) -> int:
    return max(1, round_half_even(Fraction(str(milliseconds)) * sample_rate / 1000))
