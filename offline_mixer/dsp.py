"""NumPy DSP primitives for the offline mixer.

The loudness implementation follows ITU-R BS.1770: the standard 48 kHz
K-weighting prefilter is applied, channel weights are used, and integrated
loudness uses 400 ms blocks with the absolute and relative gates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .manifest import DuckingParams, LoudnessParams, NoiseGateParams


LOUDNESS_OFFSET = -0.691
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0
PEAK_CEILING = 0.999

# BS.1770-4 48 kHz prefilter coefficients.
K_SHELF_B = (1.53512485958697, -2.69169618940638, 1.19839281085285)
K_SHELF_A = (1.0, -1.69065929318241, 0.73248077421585)
K_HIGHPASS_B = (1.0, -2.0, 1.0)
K_HIGHPASS_A = (1.0, -1.99004745483398, 0.99007225036621)

CHANNEL_WEIGHTS = {
    1: np.array([1.0], dtype=np.float64),
    2: np.array([1.0, 1.0], dtype=np.float64),
    6: np.array([1.0, 1.0, 1.0, 0.0, 1.41, 1.41], dtype=np.float64),
}


@dataclass(frozen=True)
class LoudnessCurve:
    starts: np.ndarray
    centers: np.ndarray
    loudness_lufs: np.ndarray
    energies: np.ndarray


@dataclass(frozen=True)
class DenoiseResult:
    signal: np.ndarray
    input_snr_db: float
    output_snr_db: float
    max_allowed_snr_db: float
    snr_constraint_met: bool
    attenuation_scale: float
    noise_floor_dbfs: float
    gate_threshold_db: float
    gate_window_ms: float
    gate_hop_ms: float
    max_gate_reduction_db: float


@dataclass(frozen=True)
class LoudnessResult:
    signal: np.ndarray
    target_lufs: float
    integrated_lufs: float
    max_abs_deviation_lu: float
    active_frame_count: int
    within_tolerance: bool
    iterations: int
    peak_before_guard: float
    peak_guard_scale_db: float
    gain_min_db: float
    gain_max_db: float


@dataclass(frozen=True)
class DuckCurve:
    gain_db: np.ndarray
    release_end_samples: tuple[tuple[int, int], ...]
    attack_start_samples: tuple[int, ...]


def as_2d(signal: np.ndarray) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError("audio must be a 1-D or 2-D array")
    return array


def channel_weights(channels: int) -> np.ndarray:
    try:
        return CHANNEL_WEIGHTS[channels]
    except KeyError as exc:
        raise ValueError(f"unsupported channel count for loudness: {channels}") from exc


def db_to_amp(db: np.ndarray | float) -> np.ndarray | float:
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 20.0)


def amp_to_db(value: np.ndarray | float, *, floor: float = 1e-20) -> np.ndarray | float:
    array = np.asarray(value, dtype=np.float64)
    return 20.0 * np.log10(np.maximum(array, floor))


def power_to_db(value: np.ndarray | float, *, floor: float = 1e-20) -> np.ndarray | float:
    array = np.asarray(value, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(array, floor))


def remove_dc(signal: np.ndarray) -> np.ndarray:
    array = as_2d(signal).copy()
    if array.shape[0]:
        array -= np.mean(array, axis=0, keepdims=True)
    return array


def apply_channel_matrix(signal: np.ndarray, matrix: Sequence[Sequence[float]]) -> np.ndarray:
    array = as_2d(signal)
    coefficients = np.asarray(matrix, dtype=np.float64)
    if coefficients.ndim != 2:
        raise ValueError("channel matrix must be two-dimensional")
    if coefficients.shape[1] != array.shape[1]:
        raise ValueError(
            f"matrix expects {coefficients.shape[1]} input channels, got {array.shape[1]}"
        )
    return array @ coefficients.T


def _biquad_impulse_response(
    impulse: np.ndarray,
    b: Sequence[float],
    a: Sequence[float],
) -> np.ndarray:
    b0, b1, b2 = b
    _, a1, a2 = a
    output = np.empty_like(impulse)
    x1 = x2 = y1 = y2 = 0.0
    for index, sample in enumerate(impulse):
        current = b0 * sample + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        output[index] = current
        x2, x1 = x1, sample
        y2, y1 = y1, current
    return output


def k_weighting_impulse_response(sample_rate: int = 48_000) -> np.ndarray:
    if sample_rate != 48_000:
        raise ValueError("the bundled BS.1770 coefficients require 48 kHz")
    # The slower high-pass pole has |z| ~= 0.995, so 16k samples is far
    # beyond its audible/measurable tail.
    length = 16_384
    impulse = np.zeros(length, dtype=np.float64)
    impulse[0] = 1.0
    first = _biquad_impulse_response(impulse, K_SHELF_B, K_SHELF_A)
    return _biquad_impulse_response(first, K_HIGHPASS_B, K_HIGHPASS_A)


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, int(math.ceil(math.log2(max(1, value)))))


def apply_k_weighting(signal: np.ndarray, sample_rate: int = 48_000) -> np.ndarray:
    """Apply the BS.1770 K-weighting prefilter using FFT overlap-add."""

    array = as_2d(signal)
    if array.shape[0] == 0:
        return array.copy()
    ir = k_weighting_impulse_response(sample_rate)
    block = 1 << 16
    fft_size = _next_power_of_two(block + ir.size - 1)
    response = np.fft.rfft(ir, fft_size)
    output = np.zeros_like(array)
    for channel in range(array.shape[1]):
        source = array[:, channel]
        for start in range(0, source.size, block):
            chunk = source[start : start + block]
            spectrum = np.fft.rfft(chunk, fft_size)
            filtered = np.fft.irfft(spectrum * response, fft_size)
            usable = min(filtered.size, source.size - start)
            output[start : start + usable, channel] += filtered[:usable]
    return output


def _weighted_energies(
    filtered: np.ndarray,
    window: int,
    hop: int,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples, channels = filtered.shape
    if samples == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    if weights.size != channels:
        raise ValueError(
            f"expected {channels} channel weights, received {weights.size}"
        )

    padded = filtered
    if samples < window:
        padded = np.zeros((window, channels), dtype=np.float64)
        padded[:samples] = filtered
        starts = np.array([0], dtype=np.int64)
        usable_samples = window
    else:
        starts = np.arange(0, samples - window + 1, hop, dtype=np.int64)
        final = samples - window
        if starts.size == 0 or starts[-1] != final:
            starts = np.append(starts, final)
        usable_samples = samples

    energies = np.zeros(starts.size, dtype=np.float64)
    for channel in range(channels):
        squared = padded[:, channel].astype(np.float64) ** 2
        cumulative = np.concatenate(([0.0], np.cumsum(squared)))
        channel_energy = (
            cumulative[starts + window] - cumulative[starts]
        ) / float(window)
        energies += weights[channel] * channel_energy
    centers = starts + window // 2
    if usable_samples == window and samples < window:
        centers = np.array([samples // 2], dtype=np.int64)
    return starts, centers, energies


def _gated_integrated_from_energies(energies: np.ndarray) -> float:
    if energies.size == 0 or not np.any(np.isfinite(energies)):
        return float("-inf")
    safe = np.maximum(energies, 1e-30)
    block_loudness = LOUDNESS_OFFSET + power_to_db(safe)
    absolute = block_loudness >= ABSOLUTE_GATE_LUFS
    if not np.any(absolute):
        return float("-inf")
    ungated_energy = float(np.mean(safe[absolute]))
    relative_loudness = LOUDNESS_OFFSET + float(power_to_db(ungated_energy))
    relative_threshold_loudness = relative_loudness + RELATIVE_GATE_LU
    relative = block_loudness >= relative_threshold_loudness
    selected = absolute & relative
    if not np.any(selected):
        selected = absolute
    integrated_energy = float(np.mean(safe[selected]))
    if integrated_energy <= 0.0:
        return float("-inf")
    return LOUDNESS_OFFSET + float(power_to_db(integrated_energy))


def integrated_loudness(
    signal: np.ndarray,
    sample_rate: int = 48_000,
    *,
    window_ms: float = 400.0,
    hop_ms: float = 100.0,
    weights: np.ndarray | None = None,
    already_k_weighted: bool = False,
) -> float:
    array = as_2d(signal)
    if array.shape[0] == 0:
        return float("-inf")
    weights = channel_weights(array.shape[1]) if weights is None else np.asarray(weights)
    filtered = array if already_k_weighted else apply_k_weighting(array, sample_rate)
    window = max(1, int(round(window_ms * sample_rate / 1000.0)))
    hop = max(1, int(round(hop_ms * sample_rate / 1000.0)))
    _, _, energies = _weighted_energies(filtered, window, hop, weights)
    return _gated_integrated_from_energies(energies)


def short_term_curve(
    signal: np.ndarray,
    sample_rate: int = 48_000,
    *,
    window_ms: float = 400.0,
    hop_ms: float = 50.0,
    weights: np.ndarray | None = None,
    already_k_weighted: bool = False,
) -> LoudnessCurve:
    array = as_2d(signal)
    if array.shape[0] == 0:
        empty_int = np.empty(0, dtype=np.int64)
        return LoudnessCurve(empty_int, empty_int, np.empty(0), np.empty(0))
    weights = channel_weights(array.shape[1]) if weights is None else np.asarray(weights)
    filtered = array if already_k_weighted else apply_k_weighting(array, sample_rate)
    window = max(1, int(round(window_ms * sample_rate / 1000.0)))
    hop = max(1, int(round(hop_ms * sample_rate / 1000.0)))
    starts, centers, energies = _weighted_energies(filtered, window, hop, weights)
    loudness = np.full(energies.shape, -np.inf, dtype=np.float64)
    valid = energies > 0.0
    loudness[valid] = LOUDNESS_OFFSET + power_to_db(energies[valid])
    return LoudnessCurve(starts, centers, loudness, energies)


def _moving_sample_power(signal: np.ndarray, window: int) -> np.ndarray:
    power = np.mean(as_2d(signal) ** 2, axis=1)
    samples = power.size
    if samples == 0:
        return power
    left = (window - 1) // 2
    right = window - 1 - left
    padded = np.concatenate(
        (np.zeros(left, dtype=np.float64), power, np.zeros(right, dtype=np.float64))
    )
    cumulative = np.concatenate(([0.0], np.cumsum(padded)))
    return (cumulative[window : window + samples] - cumulative[:samples]) / float(window)


def _merge_intervals(
    intervals: Iterable[tuple[int, int]],
    *,
    min_gap_samples: int = 0,
    limit: int | None = None,
) -> list[tuple[int, int]]:
    sorted_intervals = sorted(
        (max(0, int(start)), int(end))
        for start, end in intervals
        if int(end) > int(start)
    )
    merged: list[list[int]] = []
    for start, end in sorted_intervals:
        if limit is not None:
            end = min(end, limit)
        if end <= start:
            continue
        if merged and start - merged[-1][1] <= min_gap_samples:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def interval_mask(intervals: Iterable[tuple[int, int]], samples: int) -> np.ndarray:
    mask = np.zeros(samples, dtype=bool)
    for start, end in intervals:
        lo = max(0, int(start))
        hi = min(samples, int(end))
        if hi > lo:
            mask[lo:hi] = True
    return mask


@dataclass(frozen=True)
class ActivityResult:
    intervals: tuple[tuple[int, int], ...]
    mask: np.ndarray
    noise_floor_db: float
    threshold_db: float


def detect_activity(
    signal: np.ndarray,
    sample_rate: int,
    params: DuckingParams,
) -> ActivityResult:
    array = as_2d(signal)
    samples = array.shape[0]
    if samples == 0:
        return ActivityResult((), np.zeros(0, dtype=bool), -120.0, params.vad_threshold_db)
    window = max(1, int(round(params.vad_window_ms * sample_rate / 1000.0)))
    moving_power = _moving_sample_power(array, window)
    nonzero = moving_power[moving_power > 1e-24]
    if nonzero.size:
        floor_power = float(np.percentile(nonzero, 10.0))
        loud_power = float(np.percentile(nonzero, 90.0))
    else:
        floor_power = 1e-24
        loud_power = 1e-24
    floor_db = float(power_to_db(floor_power))
    loud_db = float(power_to_db(loud_power))
    if loud_db - floor_db >= params.vad_snr_db:
        threshold_db = max(params.vad_threshold_db, floor_db + params.vad_snr_db)
    else:
        # A continuously active clip has no reliable noise-only percentile;
        # fall back to the explicit absolute activity threshold.
        threshold_db = params.vad_threshold_db
    active = power_to_db(moving_power) >= threshold_db

    if not np.any(active):
        return ActivityResult((), active, floor_db, threshold_db)
    changes = np.diff(np.concatenate(([0], active.astype(np.int8), [0])))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    min_gap = max(0, int(round(params.min_gap_ms * sample_rate / 1000.0)))
    intervals = _merge_intervals(
        zip(starts.tolist(), ends.tolist()),
        min_gap_samples=min_gap,
        limit=samples,
    )
    merged_mask = interval_mask(intervals, samples)
    return ActivityResult(tuple(intervals), merged_mask, floor_db, threshold_db)


def _frame_power(signal: np.ndarray, window: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    array = as_2d(signal)
    samples = array.shape[0]
    if samples == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    padded = array
    if samples < window:
        padded = np.zeros((window, array.shape[1]), dtype=np.float64)
        padded[:samples] = array
        starts = np.array([0], dtype=np.int64)
    else:
        starts = np.arange(0, samples - window + 1, hop, dtype=np.int64)
        final = samples - window
        if starts.size == 0 or starts[-1] != final:
            starts = np.append(starts, final)
    powers = np.zeros(starts.size, dtype=np.float64)
    for channel in range(padded.shape[1]):
        cumulative = np.concatenate(
            ([0.0], np.cumsum(padded[:, channel].astype(np.float64) ** 2))
        )
        powers += (
            cumulative[starts + window] - cumulative[starts]
        ) / float(window)
    powers /= float(padded.shape[1])
    return starts, powers


def _sliding_percentile(
    values: np.ndarray,
    half_window: int,
    percentile: float,
) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    result = np.empty_like(values)
    for index in range(values.size):
        lo = max(0, index - half_window)
        hi = min(values.size, index + half_window + 1)
        result[index] = np.percentile(values[lo:hi], percentile)
    return result


def _smooth_frame_db(
    target_db: np.ndarray,
    attack_ms: float,
    release_ms: float,
    hop_s: float,
) -> np.ndarray:
    if target_db.size == 0:
        return target_db.copy()
    output = np.empty_like(target_db)
    output[0] = target_db[0]
    attack_tau = attack_ms / 1000.0
    release_tau = release_ms / 1000.0
    for index in range(1, target_db.size):
        target = target_db[index]
        previous = output[index - 1]
        tau = attack_tau if target < previous else release_tau
        if tau <= 0.0:
            alpha = 0.0
        else:
            alpha = math.exp(-hop_s / tau)
        output[index] = target + (previous - target) * alpha
    return output


def _interpolate_frame_curve(
    starts: np.ndarray,
    window: int,
    values: np.ndarray,
    samples: int,
) -> np.ndarray:
    if samples == 0:
        return np.empty(0, dtype=np.float64)
    if values.size == 0:
        return np.zeros(samples, dtype=np.float64)
    anchors = starts.astype(np.float64) + window / 2.0
    source = np.arange(samples, dtype=np.float64)
    return np.interp(source, anchors, values)


def _input_snr(
    frame_power: np.ndarray,
    active_frames: np.ndarray,
    floor_power: float,
) -> float:
    if np.any(active_frames) and np.any(~active_frames):
        signal_power = float(np.mean(frame_power[active_frames]))
        noise_power = float(np.mean(frame_power[~active_frames]))
    elif np.any(active_frames):
        signal_power = float(np.mean(frame_power[active_frames]))
        noise_power = floor_power
    else:
        return float("inf")
    if noise_power <= 0.0:
        return float("inf")
    return float(power_to_db(signal_power / noise_power))


def _scaled_frame_snr(
    frame_power: np.ndarray,
    active_frames: np.ndarray,
    gain_db: np.ndarray,
    scale: float,
    floor_power: float,
) -> float:
    if np.any(active_frames) and np.any(~active_frames):
        active_out = float(
            np.mean(
                frame_power[active_frames]
                * np.power(10.0, scale * gain_db[active_frames] / 10.0)
            )
        )
        noise_out = float(
            np.mean(
                frame_power[~active_frames]
                * np.power(10.0, scale * gain_db[~active_frames] / 10.0)
            )
        )
    elif np.any(active_frames):
        active_out = float(
            np.mean(
                frame_power[active_frames]
                * np.power(10.0, scale * gain_db[active_frames] / 10.0)
            )
        )
        noise_out = floor_power
    else:
        return float("inf")
    if noise_out <= 0.0:
        return float("inf")
    return float(power_to_db(active_out / noise_out))


def denoise_dialogue(
    signal: np.ndarray,
    sample_rate: int,
    params: NoiseGateParams,
) -> DenoiseResult:
    array = as_2d(signal)
    samples = array.shape[0]
    if samples == 0:
        return DenoiseResult(
            array.copy(),
            float("inf"),
            float("inf"),
            float("inf"),
            True,
            0.0,
            -120.0,
            params.threshold_db,
            params.window_ms,
            params.hop_ms,
            0.0,
        )
    window = max(1, int(round(params.window_ms * sample_rate / 1000.0)))
    hop = max(1, int(round(params.hop_ms * sample_rate / 1000.0)))
    starts, frame_power = _frame_power(array, window, hop)
    frame_db = np.asarray(power_to_db(frame_power))
    estimate_half = max(
        0, int(round(params.noise_estimate_window_ms / params.hop_ms / 2.0))
    )
    floor_power = _sliding_percentile(
        np.maximum(frame_power, 1e-30),
        estimate_half,
        params.noise_percentile,
    )
    floor_db = np.asarray(power_to_db(floor_power))

    below = frame_db < params.threshold_db
    reduction = params.attenuation_db + np.maximum(0.0, floor_db - frame_db)
    attenuation_target = np.where(below, -np.minimum(reduction, 60.0), 0.0)
    smoothed_gain_db = _smooth_frame_db(
        attenuation_target,
        params.attack_ms,
        params.release_ms,
        hop / float(sample_rate),
    )
    sample_gain_db = _interpolate_frame_curve(
        starts, window, smoothed_gain_db, samples
    )

    active_frames = frame_db >= params.threshold_db
    if not np.any(active_frames):
        active_frames = np.ones_like(active_frames, dtype=bool)
    if np.any(active_frames) and np.any(~active_frames):
        input_noise_power = float(np.mean(frame_power[~active_frames]))
    else:
        input_noise_power = float(np.median(floor_power))
    input_snr_db = _input_snr(frame_power, active_frames, input_noise_power)
    max_allowed = input_snr_db + params.snr_margin_db
    output_snr_db = _scaled_frame_snr(
        frame_power,
        active_frames,
        smoothed_gain_db,
        1.0,
        input_noise_power,
    )

    attenuation_scale = 1.0
    if not math.isinf(output_snr_db) and output_snr_db > max_allowed:
        low, high = 0.0, 1.0
        for _ in range(40):
            middle = (low + high) / 2.0
            trial = _scaled_frame_snr(
                frame_power,
                active_frames,
                smoothed_gain_db,
                middle,
                input_noise_power,
            )
            if trial > max_allowed:
                high = middle
            else:
                low = middle
        attenuation_scale = low
        output_snr_db = _scaled_frame_snr(
            frame_power,
            active_frames,
            smoothed_gain_db,
            attenuation_scale,
            input_noise_power,
        )

    output = array * np.power(10.0, attenuation_scale * sample_gain_db / 20.0)[:, None]
    return DenoiseResult(
        signal=output,
        input_snr_db=float(input_snr_db),
        output_snr_db=float(output_snr_db),
        max_allowed_snr_db=float(max_allowed),
        snr_constraint_met=bool(output_snr_db <= max_allowed + 1e-6),
        attenuation_scale=float(attenuation_scale),
        noise_floor_dbfs=float(np.median(floor_db)),
        gate_threshold_db=params.threshold_db,
        gate_window_ms=params.window_ms,
        gate_hop_ms=params.hop_ms,
        max_gate_reduction_db=float(np.min(attenuation_scale * sample_gain_db)),
    )


def _smoothstep_transition(
    start_value: float,
    target_value: float,
    length: int,
    duration_ms: float,
    sample_rate: int,
) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    if start_value == target_value:
        return np.full(length, target_value, dtype=np.float64)
    if duration_ms <= 0.0:
        return np.full(length, target_value, dtype=np.float64)
    duration = max(1, int(round(duration_ms * sample_rate / 1000.0)))
    positions = np.arange(1, length + 1, dtype=np.float64)
    fraction = np.minimum(positions / float(duration), 1.0)
    eased = 0.5 - 0.5 * np.cos(np.pi * fraction)
    return start_value + (target_value - start_value) * eased


def duck_gain_curve(
    samples: int,
    intervals: Iterable[tuple[int, int]],
    *,
    depth_db: float,
    attack_ms: float,
    release_ms: float,
    sample_rate: int = 48_000,
) -> DuckCurve:
    merged = _merge_intervals(intervals, limit=samples)
    gain = np.zeros(samples, dtype=np.float64)
    attack_starts: list[int] = []
    release_ends: list[tuple[int, int]] = []
    release_samples = max(
        0, int(round(release_ms * sample_rate / 1000.0))
    )
    for _, end in merged:
        release_ends.append((end, min(samples, end + release_samples)))
    cursor = 0
    current = 0.0
    for start, end in merged:
        if start > cursor:
            segment = _smoothstep_transition(
                current, 0.0, start - cursor, release_ms, sample_rate
            )
            gain[cursor:start] = segment
            current = float(segment[-1])
        active = _smoothstep_transition(
            current, -depth_db, end - start, attack_ms, sample_rate
        )
        gain[start:end] = active
        current = float(active[-1])
        attack_starts.append(start)
        cursor = end
    if cursor < samples:
        segment = _smoothstep_transition(
            current, 0.0, samples - cursor, release_ms, sample_rate
        )
        gain[cursor:] = segment
    return DuckCurve(
        gain_db=gain,
        release_end_samples=tuple(release_ends),
        attack_start_samples=tuple(attack_starts),
    )


def _active_fraction(
    active_mask: np.ndarray,
    starts: np.ndarray,
    window: int,
) -> np.ndarray:
    if active_mask.size == 0 or starts.size == 0:
        return np.zeros(starts.size, dtype=np.float64)
    required = int(np.max(starts)) + window
    if required > active_mask.size:
        padded = np.zeros(required, dtype=bool)
        padded[: active_mask.size] = active_mask
        active_mask = padded
    cumulative = np.concatenate(([0], np.cumsum(active_mask.astype(np.int64))))
    return (cumulative[starts + window] - cumulative[starts]) / float(window)


def _frame_correction_samples(
    starts: np.ndarray,
    window: int,
    corrections: np.ndarray,
    sample_count: int,
    params: LoudnessParams,
    sample_rate: int,
) -> np.ndarray:
    hop_seconds = (
        (starts[1] - starts[0]) / float(sample_rate)
        if starts.size > 1
        else params.window_ms / 1000.0
    )
    smoothed = _smooth_frame_db(
        corrections,
        params.attack_ms,
        params.release_ms,
        hop_seconds,
    )
    return _interpolate_frame_curve(starts, window, smoothed, sample_count)


def normalize_dialogue_loudness(
    signal: np.ndarray,
    sample_rate: int,
    target_lufs: float,
    active_intervals: Iterable[tuple[int, int]],
    params: LoudnessParams,
) -> LoudnessResult:
    array = as_2d(signal)
    samples = array.shape[0]
    weights = channel_weights(array.shape[1])
    active_mask = interval_mask(active_intervals, samples)
    if samples == 0:
        return LoudnessResult(
            array.copy(),
            target_lufs,
            float("-inf"),
            float("inf"),
            0,
            False,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    window = int(round(params.window_ms * sample_rate / 1000.0))
    hop = max(1, int(round(0.05 * sample_rate)))
    gain_samples = np.zeros(samples, dtype=np.float64)
    if np.any(active_mask):
        gain_samples, iterations = _solve_leveling_gain(
            array,
            active_mask,
            target_lufs,
            params,
            weights,
            sample_rate,
            window,
            hop,
        )
    else:
        iterations = 0
    output = array * np.power(10.0, gain_samples / 20.0)[:, None]
    final_curve = short_term_curve(
        output,
        sample_rate,
        window_ms=params.window_ms,
        hop_ms=50.0,
        weights=weights,
    )
    fraction = _active_fraction(active_mask, final_curve.starts, window)
    active_frames = (fraction >= 0.5) & np.isfinite(final_curve.loudness_lufs)
    if np.any(active_frames):
        deviation = float(
            np.max(np.abs(final_curve.loudness_lufs[active_frames] - target_lufs))
        )
    else:
        deviation = float("inf")

    peak_before_guard = float(np.max(np.abs(output))) if output.size else 0.0
    guard_scale_db = 0.0
    if peak_before_guard > PEAK_CEILING:
        scale = PEAK_CEILING / peak_before_guard
        output *= scale
        guard_scale_db = float(amp_to_db(scale))
        final_curve = short_term_curve(
            output,
            sample_rate,
            window_ms=params.window_ms,
            hop_ms=50.0,
            weights=weights,
        )
        fraction = _active_fraction(active_mask, final_curve.starts, window)
        active_frames = (fraction >= 0.5) & np.isfinite(final_curve.loudness_lufs)
        if np.any(active_frames):
            deviation = float(
                np.max(
                    np.abs(final_curve.loudness_lufs[active_frames] - target_lufs)
                )
            )

    return LoudnessResult(
        signal=output,
        target_lufs=target_lufs,
        integrated_lufs=float(integrated_loudness(output, sample_rate, weights=weights)),
        max_abs_deviation_lu=deviation,
        active_frame_count=int(np.count_nonzero(active_frames)),
        within_tolerance=bool(deviation <= params.tolerance_lu + 1e-9),
        iterations=iterations,
        peak_before_guard=peak_before_guard,
        peak_guard_scale_db=guard_scale_db,
        gain_min_db=float(np.min(gain_samples)),
        gain_max_db=float(np.max(gain_samples)),
    )


def _measurement_smoothing_frames(params: LoudnessParams, hop_ms: float = 50.0) -> int:
    # The gain curve follows the 400 ms loudness window, with extra smoothing
    # driven by the release/attack parameters. Keeping this span odd preserves
    # a centered, look-ahead-free result at the frame-grid level.
    span_ms = (
        0.5 * params.window_ms
        + 0.5 * params.release_ms
        + 0.25 * params.attack_ms
    )
    frames = max(3, int(round(span_ms / hop_ms)))
    if frames % 2 == 0:
        frames += 1
    return min(frames, 31)


def _solve_leveling_gain(
    signal: np.ndarray,
    active_mask: np.ndarray,
    target_lufs: float,
    params: LoudnessParams,
    weights: np.ndarray,
    sample_rate: int,
    window: int,
    hop: int,
) -> tuple[np.ndarray, int]:
    samples = signal.shape[0]
    filtered = apply_k_weighting(signal, sample_rate)
    weighted_power = np.sum(filtered * filtered * weights[None, :], axis=1)
    if samples < window:
        weighted_power = np.pad(weighted_power, (0, window - samples))
        starts = np.array([0], dtype=np.int64)
    else:
        starts = np.arange(0, samples - window + 1, hop, dtype=np.int64)
        final = samples - window
        if starts.size == 0 or starts[-1] != final:
            starts = np.append(starts, final)
    centers = starts + window / 2.0
    active_fraction = _active_fraction(active_mask, starts, window)
    active_frames = active_fraction >= 0.5
    gain_samples = np.zeros(samples, dtype=np.float64)
    if not np.any(active_frames):
        return gain_samples, 0

    cumulative_power = np.concatenate(([0.0], np.cumsum(weighted_power)))
    smoothing_frames = _measurement_smoothing_frames(
        params, hop * 1000.0 / sample_rate
    )
    half = smoothing_frames // 2
    kernel = np.ones(smoothing_frames, dtype=np.float64) / smoothing_frames
    sample_index = np.arange(samples, dtype=np.float64)
    best_deviation = float("inf")
    best_gain = gain_samples.copy()
    iterations = 0

    # Iterate against K-weighted frame energy. Since K filtering is linear and
    # the leveling gain is deliberately slow, energy can be updated without a
    # fresh FFT pass on every iteration.
    for iteration in range(100):
        iterations = iteration + 1
        frame_energy = (
            cumulative_power[starts + window] - cumulative_power[starts]
        ) / float(window)
        loudness = np.full(frame_energy.shape, -np.inf, dtype=np.float64)
        valid = frame_energy > 0.0
        loudness[valid] = LOUDNESS_OFFSET + power_to_db(frame_energy[valid])
        correction = np.zeros_like(loudness)
        controllable = active_frames & np.isfinite(loudness)
        if not np.any(controllable):
            break
        correction[controllable] = target_lufs - loudness[controllable]
        correction = np.clip(correction, -12.0, 12.0)
        deviation = float(np.max(np.abs(correction[controllable])))
        if deviation < best_deviation:
            best_deviation = deviation
            best_gain = gain_samples.copy()
        if deviation <= 0.25:
            break
        smoothed = np.convolve(
            np.pad(correction, (half, half), mode="edge"),
            kernel,
            mode="valid",
        )
        update = np.interp(sample_index, centers, smoothed)
        gain_samples = np.clip(
            gain_samples + update,
            -params.max_gain_db,
            params.max_gain_db,
        )
        weighted_power = np.maximum(weighted_power, 0.0)
        # weighted_power is static; frame energy above is recomputed from the
        # accumulated gain below by replacing the cumulative sum.
        updated_power = np.sum(
            (
                filtered
                * np.power(10.0, gain_samples / 20.0)[:, None]
            )
            ** 2
            * weights[None, :],
            axis=1,
        )
        if samples < window:
            updated_power = np.pad(updated_power, (0, window - samples))
        cumulative_power = np.concatenate(
            ([0.0], np.cumsum(updated_power))
        )

    # A small number of actual-filter corrections remove the approximation
    # error from the fast energy update above.
    current = signal * np.power(10.0, best_gain / 20.0)[:, None]
    actual_best_deviation = float("inf")
    actual_best_gain = best_gain.copy()
    for _ in range(8):
        curve = short_term_curve(
            current,
            sample_rate,
            window_ms=params.window_ms,
            hop_ms=50.0,
            weights=weights,
        )
        fraction = _active_fraction(active_mask, curve.starts, window)
        controllable = (fraction >= 0.5) & np.isfinite(curve.loudness_lufs)
        if not np.any(controllable):
            break
        correction = np.zeros_like(curve.loudness_lufs)
        correction[controllable] = (
            target_lufs - curve.loudness_lufs[controllable]
        )
        correction = np.clip(correction, -6.0, 6.0)
        deviation = float(np.max(np.abs(correction[controllable])))
        if deviation < actual_best_deviation:
            actual_best_deviation = deviation
            actual_best_gain = best_gain.copy()
        if deviation <= 0.35:
            break
        smoothed = np.convolve(
            np.pad(correction, (half, half), mode="edge"),
            kernel,
            mode="valid",
        )
        update = np.interp(sample_index, curve.centers, smoothed)
        best_gain = np.clip(
            best_gain + update,
            -params.max_gain_db,
            params.max_gain_db,
        )
        current = signal * np.power(10.0, best_gain / 20.0)[:, None]
    return actual_best_gain, iterations + 8


def apply_track_gain(signal: np.ndarray, gain_db: float) -> np.ndarray:
    return as_2d(signal) * float(db_to_amp(gain_db))


def guard_output_peak(
    signal: np.ndarray,
    ceiling: float = PEAK_CEILING,
) -> tuple[np.ndarray, float, float]:
    array = as_2d(signal)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak <= ceiling or peak <= 0.0:
        return array, peak, 0.0
    scale = ceiling / peak
    return array * scale, peak, float(amp_to_db(scale))


def concatenate_intervals(
    signal: np.ndarray,
    intervals: Iterable[tuple[int, int]],
) -> np.ndarray:
    array = as_2d(signal)
    pieces: list[np.ndarray] = []
    for start, end in intervals:
        lo = max(0, int(start))
        hi = min(array.shape[0], int(end))
        if hi > lo:
            pieces.append(array[lo:hi])
    if not pieces:
        return np.empty((0, array.shape[1]), dtype=np.float64)
    return np.concatenate(pieces, axis=0)


def short_term_deviation(
    signal: np.ndarray,
    sample_rate: int,
    target_lufs: float,
    *,
    active_intervals: Iterable[tuple[int, int]] | None = None,
    segment_range: tuple[int, int] | None = None,
) -> tuple[float, int]:
    array = as_2d(signal)
    if array.shape[0] == 0:
        return float("inf"), 0
    weights = channel_weights(array.shape[1])
    curve = short_term_curve(array, sample_rate, weights=weights)
    window = int(round(400.0 * sample_rate / 1000.0))
    if active_intervals is None:
        fraction = np.ones(curve.starts.size, dtype=np.float64)
    else:
        active_mask = interval_mask(active_intervals, array.shape[0])
        fraction = _active_fraction(active_mask, curve.starts, window)
    active = (fraction >= 0.5) & np.isfinite(curve.loudness_lufs)
    if segment_range is not None:
        lo, hi = segment_range
        active &= (curve.centers >= lo) & (curve.centers < hi)
    if not np.any(active):
        return float("inf"), 0
    deviation = float(
        np.max(np.abs(curve.loudness_lufs[active] - target_lufs))
    )
    return deviation, int(np.count_nonzero(active))


def peak_dbfs(signal: np.ndarray) -> float:
    array = as_2d(signal)
    if array.size == 0:
        return float("-inf")
    peak = float(np.max(np.abs(array)))
    if peak <= 0.0:
        return float("-inf")
    return float(amp_to_db(peak))
