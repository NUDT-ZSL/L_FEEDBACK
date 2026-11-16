from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal

from .loudness import soft_limit


@dataclass(frozen=True)
class VariantParameters:
    pitch_semitones: float
    time_stretch: float
    lowpass_hz: float


@dataclass
class TransientAlignment:
    samples: np.ndarray
    reference_sample: int
    predicted_sample: int
    initial_detected_sample: int | None
    initial_deviation_ms: float | None
    detected_sample: int | None
    deviation_ms: float | None
    aligned: bool
    iterations: int
    error_code: str | None = None
    error_message: str | None = None


def derive_parameters(
    skill_id: str,
    random_seed: int,
    variant_index: int,
    sample_rate: int,
) -> VariantParameters:
    payload = (
        f"skillgen-v1|{skill_id}|{random_seed}|{variant_index}".encode("utf-8")
    )
    digest = hashlib.sha256(payload).digest()
    seed_sequence = int.from_bytes(digest, "big", signed=False)
    rng = np.random.Generator(np.random.PCG64(seed_sequence))
    pitch = float(rng.uniform(-3.0, 3.0))
    stretch = float(rng.uniform(0.9, 1.1))
    upper_cutoff = max(401.0, (sample_rate / 2.0) * 0.98)
    cutoff = float(rng.uniform(400.0, upper_cutoff))
    return VariantParameters(
        pitch_semitones=pitch,
        time_stretch=stretch,
        lowpass_hz=cutoff,
    )


def _principal_argument(phase: np.ndarray) -> np.ndarray:
    return (phase + np.pi) % (2.0 * np.pi) - np.pi


def _phase_vocoder_stft(
    samples: np.ndarray, n_fft: int, hop: int
) -> tuple[np.ndarray, int, int]:
    pad = n_fft // 2
    padded = np.pad(samples, (pad, pad + n_fft), mode="constant")
    frame_count = 1 + max(0, (padded.size - n_fft) // hop)
    if frame_count < 1:
        padded = np.pad(padded, (0, n_fft), mode="constant")
        frame_count = 1
    window = signal.get_window("hann", n_fft, fftbins=True).astype(np.float64)
    frames = np.empty((frame_count, n_fft), dtype=np.float64)
    for frame_index in range(frame_count):
        start = frame_index * hop
        frames[frame_index] = padded[start : start + n_fft] * window
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    return spectrum, pad, padded.size


def _phase_vocoder_istft(
    spectrum: np.ndarray,
    n_fft: int,
    hop: int,
    pad: int,
    output_length: int,
) -> np.ndarray:
    frames = np.fft.irfft(spectrum, n=n_fft, axis=1)
    window = signal.get_window("hann", n_fft, fftbins=True).astype(np.float64)
    total = (frames.shape[0] - 1) * hop + n_fft
    output = np.zeros(total, dtype=np.float64)
    normalization = np.zeros(total, dtype=np.float64)
    for frame_index, frame in enumerate(frames):
        start = frame_index * hop
        output[start : start + n_fft] += frame * window
        normalization[start : start + n_fft] += window * window
    valid = normalization > 1e-12
    output[valid] /= normalization[valid]
    if pad > 0:
        output = output[pad:]
    if output.size < output_length:
        output = np.pad(output, (0, output_length - output.size), mode="constant")
    else:
        output = output[:output_length]
    return np.ascontiguousarray(output, dtype=np.float64)


def time_stretch(
    samples: np.ndarray,
    duration_factor: float,
    sample_rate: int,
) -> np.ndarray:
    source = np.asarray(samples, dtype=np.float64)
    if source.size == 0:
        return source.copy()
    expected_length = max(1, int(round(source.size * duration_factor)))
    if abs(duration_factor - 1.0) < 1e-9 or expected_length == source.size:
        return source.copy()
    if sample_rate >= 32000:
        n_fft = 1024
    elif sample_rate >= 16000:
        n_fft = 512
    else:
        n_fft = 256
    n_fft = min(n_fft, max(64, 1 << int(math.floor(math.log2(max(64, source.size))))))
    hop_analysis = max(1, n_fft // 4)
    hop_synthesis = max(1, int(round(hop_analysis * duration_factor)))

    spectrum, pad, _ = _phase_vocoder_stft(source, n_fft, hop_analysis)
    bin_omega = 2.0 * np.pi * np.arange(spectrum.shape[1], dtype=np.float64) / n_fft
    magnitude = np.abs(spectrum)
    previous_phase = np.angle(spectrum[0])
    previous_output_phase = previous_phase.copy()
    output_spectrum = np.empty_like(spectrum)
    output_spectrum[0] = magnitude[0] * np.exp(1j * previous_phase)
    for frame_index in range(1, spectrum.shape[0]):
        phase = np.angle(spectrum[frame_index])
        expected = bin_omega * hop_analysis
        residual = _principal_argument(phase - previous_phase - expected)
        true_frequency = bin_omega + residual / hop_analysis
        output_phase = previous_output_phase + true_frequency * hop_synthesis
        output_spectrum[frame_index] = magnitude[frame_index] * np.exp(
            1j * output_phase
        )
        previous_phase = phase
        previous_output_phase = output_phase
    return _phase_vocoder_istft(
        output_spectrum, n_fft, hop_synthesis, pad, expected_length
    )


def _fit_length(samples: np.ndarray, length: int) -> np.ndarray:
    if samples.size == length:
        return np.ascontiguousarray(samples, dtype=np.float64)
    if samples.size > length:
        return np.ascontiguousarray(samples[:length], dtype=np.float64)
    return np.pad(
        samples, (0, length - samples.size), mode="constant"
    ).astype(np.float64, copy=False)


def _resample_to_length(samples: np.ndarray, output_length: int) -> np.ndarray:
    if samples.size == 0:
        return np.zeros(output_length, dtype=np.float64)
    if output_length == samples.size:
        return samples.copy()
    fraction = Fraction(output_length, samples.size).limit_denominator(2000)
    resampled = signal.resample_poly(
        samples,
        fraction.numerator,
        fraction.denominator,
        padtype="constant",
    )
    return _fit_length(np.asarray(resampled, dtype=np.float64), output_length)


def second_order_lowpass(
    samples: np.ndarray, sample_rate: int, cutoff_hz: float
) -> np.ndarray:
    nyquist = sample_rate / 2.0
    cutoff = min(max(float(cutoff_hz), 1.0), np.nextafter(nyquist, 0.0))
    sos = signal.butter(2, cutoff, btype="lowpass", fs=sample_rate, output="sos")
    if samples.size < 3:
        return np.asarray(samples, dtype=np.float64).copy()
    padlen = min(samples.size - 1, 3 * (2 * len(sos) + 1))
    return np.asarray(
        signal.sosfiltfilt(sos, samples, padlen=padlen), dtype=np.float64
    )


def synthesize_variant(
    base_samples: np.ndarray,
    sample_rate: int,
    parameters: VariantParameters,
    target_duration: float,
    reference_sample: int,
) -> tuple[np.ndarray, int]:
    source = np.asarray(base_samples, dtype=np.float64)
    pitch_ratio = 2.0 ** (parameters.pitch_semitones / 12.0)
    pitch_length = max(1, int(round(source.size / pitch_ratio)))
    pitch_shifted = _resample_to_length(source, pitch_length)
    stretched_length = max(
        1, int(round(pitch_shifted.size * parameters.time_stretch))
    )
    stretched = time_stretch(
        pitch_shifted,
        stretched_length / pitch_shifted.size,
        sample_rate,
    )
    filtered = second_order_lowpass(
        stretched, sample_rate, parameters.lowpass_hz
    )
    target_samples = max(1, int(round(target_duration * sample_rate)))
    output = _fit_length(filtered, target_samples)
    predicted = int(round(reference_sample * parameters.time_stretch / pitch_ratio))
    predicted = min(max(predicted, 0), target_samples - 1)
    return output, predicted


def detect_first_transient(
    samples: np.ndarray,
    sample_rate: int,
    expected_sample: int | None,
    search_ms: float = 100.0,
) -> int | None:
    source = np.asarray(samples, dtype=np.float64)
    if source.size < 8:
        return None
    global_peak = float(np.max(np.abs(source)))
    if global_peak < 1e-8:
        return None

    window_size = max(3, int(round(0.004 * sample_rate)))
    window_size |= 1
    window = signal.get_window("hann", window_size, fftbins=True)
    window /= float(np.sum(window))
    energy = np.convolve(source * source, window, mode="same")
    envelope = np.sqrt(np.maximum(energy, 0.0))
    lag = max(1, int(round(0.008 * sample_rate)))
    attack = np.zeros_like(envelope)
    attack[lag:] = np.maximum(0.0, envelope[lag:] - envelope[:-lag])
    edge = min(source.size, lag + window_size)
    if expected_sample is None or expected_sample >= edge:
        attack[:edge] = 0.0
    if edge < source.size and (
        expected_sample is None or expected_sample <= source.size - edge
    ):
        attack[-edge:] = 0.0
    smooth_size = max(1, int(round(0.001 * sample_rate)))
    if smooth_size > 1:
        attack = np.convolve(
            attack, np.ones(smooth_size, dtype=np.float64) / smooth_size, mode="same"
        )

    if expected_sample is None:
        center = int(np.argmax(envelope))
    else:
        center = int(expected_sample)
    half_window = max(1, int(round(search_ms * sample_rate / 1000.0)))
    start = max(0, center - half_window)
    end = min(source.size, center + half_window)
    if end <= start:
        return None
    region = attack[start:end]
    regional_max = float(np.max(region))
    if regional_max <= 1e-10:
        return None
    candidates = np.flatnonzero(region >= 0.20 * regional_max)
    if candidates.size == 0:
        return None
    coarse = start + int(candidates[0])

    refine_size = max(window_size, int(round(0.008 * sample_rate)))
    refine_start = max(0, coarse - refine_size)
    refine_end = min(source.size, coarse + refine_size)
    local_peak = float(np.max(np.abs(source[refine_start:refine_end])))
    if local_peak <= 0.0:
        return None
    threshold = max(0.08 * local_peak, 0.01 * global_peak)
    above = np.flatnonzero(
        np.abs(source[refine_start:refine_end]) >= threshold
    )
    if above.size == 0:
        return coarse
    return refine_start + int(above[0])


def _shift_linear(samples: np.ndarray, offset: float) -> np.ndarray:
    source = np.asarray(samples, dtype=np.float64)
    positions = np.arange(source.size, dtype=np.float64) + offset
    shifted = np.interp(
        positions,
        np.arange(source.size, dtype=np.float64),
        source,
        left=0.0,
        right=0.0,
    )
    return np.ascontiguousarray(shifted, dtype=np.float64)


def align_transient(
    samples: np.ndarray,
    sample_rate: int,
    reference_sample: int,
    predicted_sample: int,
    tolerance_ms: float = 5.0,
) -> TransientAlignment:
    current = np.asarray(samples, dtype=np.float64).copy()
    initial_detected = detect_first_transient(
        current, sample_rate, expected_sample=predicted_sample, search_ms=120.0
    )
    if initial_detected is None:
        return TransientAlignment(
            samples=current,
            reference_sample=reference_sample,
            predicted_sample=predicted_sample,
            initial_detected_sample=None,
            initial_deviation_ms=None,
            detected_sample=None,
            deviation_ms=None,
            aligned=False,
            iterations=0,
            error_code="E_TRANSIENT",
            error_message="could not detect the first transient in generated variant",
        )

    initial_deviation = (
        (initial_detected - reference_sample) * 1000.0 / sample_rate
    )
    detected = initial_detected
    iterations = 0
    for iteration in range(4):
        residual = (detected - reference_sample) * 1000.0 / sample_rate
        if abs(residual) <= tolerance_ms:
            break
        offset = float(detected - reference_sample)
        current = _shift_linear(current, offset)
        iterations = iteration + 1
        next_detected = detect_first_transient(
            current,
            sample_rate,
            expected_sample=reference_sample,
            search_ms=max(20.0, tolerance_ms * 4.0),
        )
        if next_detected is None:
            break
        detected = next_detected

    residual = (detected - reference_sample) * 1000.0 / sample_rate
    aligned = abs(residual) <= tolerance_ms
    error_code = None
    error_message = None
    if not aligned:
        error_code = "E_TRANSIENT"
        error_message = (
            f"first transient remains {residual:.4f} ms from reference "
            f"(tolerance +/-{tolerance_ms:.1f} ms)"
        )
    return TransientAlignment(
        samples=current,
        reference_sample=reference_sample,
        predicted_sample=predicted_sample,
        initial_detected_sample=initial_detected,
        initial_deviation_ms=float(initial_deviation),
        detected_sample=int(detected),
        deviation_ms=float(residual),
        aligned=aligned,
        iterations=iterations,
        error_code=error_code,
        error_message=error_message,
    )


def final_peak(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def apply_emergency_limit(samples: np.ndarray) -> np.ndarray:
    if final_peak(samples) <= 1.0:
        return np.asarray(samples, dtype=np.float64).copy()
    return soft_limit(samples)
