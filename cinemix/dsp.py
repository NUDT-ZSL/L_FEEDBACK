from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class LoudnessResult:
    lufs: float
    ungated_lufs: float
    block_count: int
    gated_block_count: int
    duration_seconds: float


@dataclass(frozen=True)
class TruePeakResult:
    dbtp: float
    linear: float
    channel_dbtp: tuple[float, ...]


@dataclass(frozen=True)
class Biquad:
    b: tuple[float, float, float]
    a: tuple[float, float, float]


def _next_power_of_two(value: int) -> int:
    return 1 << max(1, int(value - 1).bit_length())


@lru_cache(maxsize=32)
def k_weighting_stages(sample_rate: int) -> tuple[Biquad, Biquad]:
    """Return the two BS.1770-4 K-weighting biquads for any PCM rate."""
    fs = float(sample_rate)
    shelf_f = 1681.974450955533
    shelf_gain_db = 3.999843853973347
    shelf_q = 0.7071752369554196
    shelf_k = math.tan(math.pi * shelf_f / fs)
    vh = 10.0 ** (shelf_gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    denominator = 1.0 + shelf_k / shelf_q + shelf_k * shelf_k
    shelf_b = (
        (vh + vb * shelf_k / shelf_q + shelf_k * shelf_k) / denominator,
        2.0 * (shelf_k * shelf_k - vh) / denominator,
        (vh - vb * shelf_k / shelf_q + shelf_k * shelf_k) / denominator,
    )
    shelf_a = (
        1.0,
        2.0 * (shelf_k * shelf_k - 1.0) / denominator,
        (1.0 - shelf_k / shelf_q + shelf_k * shelf_k) / denominator,
    )

    highpass_f = 38.13547087602444
    highpass_q = 0.5003270373238773
    highpass_k = math.tan(math.pi * highpass_f / fs)
    highpass_denominator = 1.0 + highpass_k / highpass_q + highpass_k * highpass_k
    highpass_b = (1.0, -2.0, 1.0)
    highpass_a = (
        1.0,
        2.0 * (highpass_k * highpass_k - 1.0) / highpass_denominator,
        (1.0 - highpass_k / highpass_q + highpass_k * highpass_k)
        / highpass_denominator,
    )
    return (
        Biquad(shelf_b, shelf_a),
        Biquad(highpass_b, highpass_a),
    )


@lru_cache(maxsize=64)
def _biquad_impulse(b: tuple[float, ...], a: tuple[float, ...], length: int = 16384) -> np.ndarray:
    impulse = np.zeros(length, dtype=np.float64)
    impulse[0] = b[0]
    if length > 1:
        impulse[1] = b[1] - a[1] * impulse[0]
    if length > 2:
        impulse[2] = (
            b[2]
            - a[1] * impulse[1]
            - a[2] * impulse[0]
        )
    for index in range(3, length):
        impulse[index] = (
            -a[1] * impulse[index - 1]
            - a[2] * impulse[index - 2]
        )
    return impulse


def _fft_convolve_causal(signal: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    frames = signal.shape[0]
    if frames == 0:
        return signal.copy()
    kernel = impulse
    while abs(kernel[-1]) < 1e-15 and kernel.size > 256:
        kernel = kernel[:-1]
    fft_length = _next_power_of_two(frames + kernel.size - 1)
    spectrum = np.fft.rfft(signal, n=fft_length, axis=0)
    kernel_spectrum = np.fft.rfft(kernel, n=fft_length)
    if signal.ndim == 1:
        product = spectrum * kernel_spectrum
    else:
        product = spectrum * kernel_spectrum[:, None]
    return np.fft.irfft(product, n=fft_length, axis=0)[:frames]


def k_weight(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply the ITU-R BS.1770 K-weighting filter to (frames, channels) audio."""
    if samples.ndim == 1:
        working = samples[:, None]
    else:
        working = np.asarray(samples, dtype=np.float64)
    output = working
    for stage in k_weighting_stages(sample_rate):
        impulse = _biquad_impulse(stage.b, stage.a)
        output = _fft_convolve_causal(output, impulse)
    if samples.ndim == 1:
        return output[:, 0]
    return output


def _loudness_from_power(power: np.ndarray) -> float:
    if power.size == 0 or not np.isfinite(power).all() or float(np.mean(power)) <= 0.0:
        return float("-inf")
    return float(-0.691 + 10.0 * math.log10(float(np.mean(power))))


def integrated_loudness(
    samples: np.ndarray,
    sample_rate: int,
    *,
    block_seconds: float = 0.400,
    hop_seconds: float = 0.100,
    absolute_gate_lufs: float = -70.0,
    relative_gate_lu: float = -10.0,
) -> LoudnessResult:
    """Measure integrated program loudness using BS.1770 K-weighting and gating."""
    working = np.asarray(samples, dtype=np.float64)
    if working.ndim == 1:
        working = working[:, None]
    frames = working.shape[0]
    duration = frames / sample_rate if frames else 0.0
    if frames == 0 or not np.all(np.isfinite(working)):
        return LoudnessResult(float("-inf"), float("-inf"), 0, 0, duration)

    weighted = k_weight(working, sample_rate)
    block_frames = max(1, round(block_seconds * sample_rate))
    hop_frames = max(1, round(hop_seconds * sample_rate))
    if frames < block_frames:
        block_frames = frames
        hop_frames = frames

    starts = np.arange(0, frames - block_frames + 1, hop_frames, dtype=np.int64)
    if starts.size == 0:
        starts = np.array([0], dtype=np.int64)
    cumulative = np.concatenate(
        (
            np.zeros((1, working.shape[1]), dtype=np.float64),
            np.cumsum(weighted * weighted, axis=0, dtype=np.float64),
        ),
        axis=0,
    )
    channel_powers = (
        cumulative[starts + block_frames] - cumulative[starts]
    ) / float(block_frames)
    channel_weights = np.ones(working.shape[1], dtype=np.float64)
    block_power = np.sum(channel_powers * channel_weights[None, :], axis=1)
    block_loudness = -0.691 + 10.0 * np.log10(
        np.maximum(block_power, np.finfo(np.float64).tiny)
    )

    ungated = _loudness_from_power(block_power)
    absolute_mask = block_loudness > absolute_gate_lufs
    if not np.any(absolute_mask):
        return LoudnessResult(float("-inf"), ungated, int(starts.size), 0, duration)
    absolute_gated_power = float(np.mean(block_power[absolute_mask]))
    absolute_gated_loudness = _loudness_from_power(np.array([absolute_gated_power]))
    relative_threshold = absolute_gated_loudness + relative_gate_lu
    final_mask = absolute_mask & (block_loudness > relative_threshold)
    if not np.any(final_mask):
        final_mask = absolute_mask
    final_power = float(np.mean(block_power[final_mask]))
    final_lufs = _loudness_from_power(np.array([final_power]))
    return LoudnessResult(
        lufs=final_lufs,
        ungated_lufs=ungated,
        block_count=int(starts.size),
        gated_block_count=int(np.count_nonzero(final_mask)),
        duration_seconds=duration,
    )


@lru_cache(maxsize=64)
def _interpolation_kernel(phase: int, phases: int, half_taps: int) -> np.ndarray:
    offsets = np.arange(-half_taps, half_taps + 1, dtype=np.float64)
    fraction = phase / float(phases)
    distance = fraction - offsets
    window = np.kaiser(offsets.size, 8.5)
    weights = np.sinc(distance) * window
    return weights / float(np.sum(weights))


def _oversample_phase(
    signal: np.ndarray, weights: np.ndarray, half_taps: int
) -> np.ndarray:
    reversed_weights = weights[::-1]
    convolution = np.convolve(signal, reversed_weights, mode="full")
    return convolution[half_taps : half_taps + signal.size]


def true_peak(
    samples: np.ndarray, sample_rate: int, *, oversample: int = 4) -> TruePeakResult:
    """Measure true peak using band-limited 4x polyphase oversampling."""
    if oversample < 1:
        raise ValueError("oversample must be >= 1")
    working = np.asarray(samples, dtype=np.float64)
    if working.ndim == 1:
        working = working[:, None]
    if working.size == 0 or not np.all(np.isfinite(working)):
        return TruePeakResult(float("-inf"), 0.0, ())

    half_taps = 32
    channel_peaks: list[float] = []
    for channel in range(working.shape[1]):
        source = working[:, channel]
        peak = float(np.max(np.abs(source)))
        for phase in range(1, oversample):
            weights = _interpolation_kernel(phase, oversample, half_taps)
            upsampled = _oversample_phase(source, weights, half_taps)
            interior = upsampled[half_taps : max(half_taps, upsampled.size - half_taps)]
            if interior.size:
                peak = max(peak, float(np.max(np.abs(interior))))
        channel_peaks.append(peak)
    linear = max(channel_peaks)
    dbtp = 20.0 * math.log10(linear) if linear > 0.0 else float("-inf")
    channel_db = tuple(
        20.0 * math.log10(value) if value > 0.0 else float("-inf")
        for value in channel_peaks
    )
    return TruePeakResult(dbtp, linear, channel_db)


def sample_peak_dbfs(samples: np.ndarray) -> float:
    working = np.asarray(samples, dtype=np.float64)
    if working.size == 0:
        return float("-inf")
    peak = float(np.max(np.abs(working)))
    return 20.0 * math.log10(peak) if peak > 0.0 else float("-inf")


def resample_signal(
    samples: np.ndarray,
    input_rate: int,
    output_rate: int,
    *,
    half_taps_base: int = 32,
    kaiser_beta: float = 9.0,
    chunk_elements: int = 2_000_000,
) -> np.ndarray:
    """High-quality direct rational resampler with sample-accurate time base.

    The band-limited kernel is evaluated at the exact rational output instant
    rather than at a rounded nearest input sample. This keeps timeline error
    below one output sample and avoids cumulative drift.
    """
    working = np.asarray(samples, dtype=np.float64)
    if working.ndim == 1:
        working = working[:, None]
        squeeze = True
    else:
        squeeze = False
    if input_rate == output_rate or working.shape[0] == 0:
        return working[:, 0] if squeeze else working.copy()

    source_frames, channels = working.shape
    gcd = math.gcd(input_rate, output_rate)
    step = input_rate // gcd
    phase_count = output_rate // gcd
    output_frames = round(source_frames * output_rate / input_rate)
    if output_frames == 0:
        empty = np.zeros((0, channels), dtype=np.float64)
        return empty[:, 0] if squeeze else empty

    ratio = output_rate / float(input_rate)
    cutoff = min(1.0, ratio)
    half_taps = max(
        half_taps_base,
        math.ceil(half_taps_base / max(cutoff, 1e-4)),
    )
    half_taps = min(half_taps, 512)
    offsets = np.arange(-half_taps, half_taps + 1, dtype=np.int64)
    inverse_step = pow(step % phase_count, -1, phase_count)
    padded = np.pad(working, ((half_taps, half_taps), (0, 0)), mode="constant")
    output = np.zeros((output_frames, channels), dtype=np.float64)
    chunk_size = max(1, chunk_elements // offsets.size)

    for phase in range(phase_count):
        first = (phase * inverse_step) % phase_count
        output_indices = np.arange(first, output_frames, phase_count, dtype=np.int64)
        if output_indices.size == 0:
            continue
        fraction = phase / float(phase_count)
        distance = fraction - offsets.astype(np.float64)
        window = np.kaiser(offsets.size, kaiser_beta)
        weights = cutoff * np.sinc(cutoff * distance) * window
        weights /= float(np.sum(weights))
        for chunk_start in range(0, output_indices.size, chunk_size):
            indices = output_indices[chunk_start : chunk_start + chunk_size]
            base = (indices * step) // phase_count
            gather = padded[base[:, None] + offsets[None, :] + half_taps, :]
            for channel in range(channels):
                output[indices, channel] = np.sum(
                    gather[:, :, channel] * weights[None, :], axis=1
                )

    if squeeze:
        return output[:, 0]
    return output


def _fade_shape(progress: np.ndarray, curve: str) -> np.ndarray:
    clipped = np.clip(progress, 0.0, 1.0)
    if curve == "linear":
        return np.sqrt(clipped)
    if curve == "equal_power":
        return np.sin(0.5 * np.pi * clipped)
    if curve == "s_curve":
        smooth = clipped * clipped * (3.0 - 2.0 * clipped)
        return np.sin(0.5 * np.pi * smooth)
    raise ValueError(f"Unsupported fade curve: {curve}")


def crossfade_gains(
    progress: np.ndarray | float, curve: str
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Return (outgoing, incoming) gains for a normalized crossfade position."""
    shape = _fade_shape(np.asarray(progress, dtype=np.float64), curve)
    complement = _fade_shape(1.0 - np.asarray(progress, dtype=np.float64), curve)
    return complement, shape


def interval_envelope(
    local_frames: int,
    sample_rate: int,
    track_start_seconds: float,
    start_seconds: float,
    end_seconds: float,
    curve: str,
    direction: str,
) -> np.ndarray:
    """Build a fade-in or fade-out envelope on a track-local sample grid.

    Boundary positions are rounded to the nearest master sample. A transition
    therefore cannot drift from the configured timeline by more than half a
    sample (well below the 1 ms acceptance limit).
    """
    if direction not in {"in", "out"}:
        raise ValueError("direction must be 'in' or 'out'")
    start_index = round(start_seconds * sample_rate)
    end_index = round(end_seconds * sample_rate)
    span = max(1, end_index - start_index)
    track_offset = round(track_start_seconds * sample_rate)
    global_indices = np.arange(local_frames, dtype=np.int64) + track_offset
    progress = (global_indices - start_index) / float(span)
    shaped = _fade_shape(progress, curve)

    if direction == "in":
        envelope = np.where(global_indices < start_index, 0.0, 1.0)
        inside = (global_indices >= start_index) & (global_indices < end_index)
        envelope[inside] = shaped[inside]
        return envelope

    envelope = np.where(global_indices >= end_index, 0.0, 1.0)
    inside = (global_indices >= start_index) & (global_indices < end_index)
    complement = _fade_shape(1.0 - progress, curve)
    envelope[inside] = complement[inside]
    return envelope


def pan_track(samples: np.ndarray, pan: float) -> np.ndarray:
    """Apply a stereo balance or equal-power mono pan law."""
    working = np.asarray(samples, dtype=np.float64)
    if working.ndim != 2:
        raise ValueError("samples must be (frames, channels)")
    if working.shape[1] == 1:
        angle = (float(np.clip(pan, -1.0, 1.0)) + 1.0) * (math.pi / 4.0)
        return np.concatenate(
            (working[:, 0:1] * math.cos(angle), working[:, 0:1] * math.sin(angle)),
            axis=1,
        )
    if working.shape[1] == 2:
        left = working[:, 0:1].copy()
        right = working[:, 1:2].copy()
        if pan < 0.0:
            right *= 1.0 + pan
        elif pan > 0.0:
            left *= 1.0 - pan
        return np.concatenate((left, right), axis=1)
    raise ValueError("only mono or stereo tracks can be panned")


def downmix_to_mono(samples: np.ndarray) -> np.ndarray:
    working = np.asarray(samples, dtype=np.float64)
    if working.ndim == 1:
        return working
    if working.shape[1] == 1:
        return working[:, 0]
    return np.mean(working, axis=1)
