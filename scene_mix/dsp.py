from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SAMPLE_RATE = 48_000
BLOCK_SECONDS = 0.400
HOP_SECONDS = 0.100
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_DB = -10.0
LOUDNESS_OFFSET = -0.691
DEFAULT_TARGET_LUFS = -23.0
GAIN_LIMIT_DB = 12.0
FADE_SECONDS = 0.050
LIMITER_KNEE = 0.80

# ITU-R BS.1770-4 K-weighting coefficients for 48 kHz.
K_STAGE1_B = np.array(
    [1.53512485958697, -2.69169618940638, 1.19839281085285], dtype=np.float64
)
K_STAGE1_A = np.array(
    [1.0, -1.69065929318241, 0.73248077421585], dtype=np.float64
)
K_STAGE2_B = np.array([1.0, -2.0, 1.0], dtype=np.float64)
K_STAGE2_A = np.array(
    [1.0, -1.99004745483398, 0.99007225036621], dtype=np.float64
)


def db_to_linear(value_db: float) -> float:
    return float(10.0 ** (value_db / 20.0))


def linear_to_db(value: float, floor: float = 1e-20) -> float:
    return float(20.0 * math.log10(max(float(value), floor)))


def _biquad_response(omega: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    z1 = np.exp(-1j * omega)
    z2 = z1 * z1
    return (b[0] + b[1] * z1 + b[2] * z2) / (
        a[0] + a[1] * z1 + a[2] * z2
    )


def k_filter_48k(samples: np.ndarray) -> np.ndarray:
    """Apply the BS.1770 K-weighting filters using FFT convolution.

    The input is zero-padded by two seconds so the causal IIR response does
    not wrap into the returned samples.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] == 0:
        return x.copy()

    padding = SAMPLE_RATE * 2
    nfft = 1
    while nfft < x.shape[0] + padding:
        nfft <<= 1

    omega = 2.0 * np.pi * np.fft.rfftfreq(nfft)
    response = _biquad_response(omega, K_STAGE1_B, K_STAGE1_A)
    response *= _biquad_response(omega, K_STAGE2_B, K_STAGE2_A)

    result = np.empty_like(x)
    for channel in range(x.shape[1]):
        spectrum = np.fft.rfft(x[:, channel], n=nfft)
        result[:, channel] = np.fft.irfft(spectrum * response, n=nfft)[
            : x.shape[0]
        ]
    return result


def _block_power(
    weighted: np.ndarray, block_samples: int, hop_samples: int
) -> Tuple[np.ndarray, np.ndarray]:
    n = weighted.shape[0]
    if n < block_samples:
        padded = np.zeros((block_samples, weighted.shape[1]), dtype=np.float64)
        padded[:n, :] = weighted
        power = np.sum(np.mean(padded * padded, axis=0))
        return np.array([power], dtype=np.float64), np.array([0], dtype=np.int64)

    starts = np.arange(0, n - block_samples + 1, hop_samples, dtype=np.int64)
    power = np.zeros(starts.shape[0], dtype=np.float64)
    for channel in range(weighted.shape[1]):
        square = weighted[:, channel] * weighted[:, channel]
        cumulative = np.concatenate(([0.0], np.cumsum(square, dtype=np.float64)))
        channel_power = (
            cumulative[starts + block_samples] - cumulative[starts]
        ) / float(block_samples)
        power += channel_power
    return power, starts


def integrated_loudness_from_weighted(
    weighted: np.ndarray, sample_rate: int = SAMPLE_RATE
) -> Optional[float]:
    """Measure gated integrated loudness of K-weighted audio."""
    if weighted.ndim == 1:
        weighted = weighted[:, None]
    block_samples = int(round(BLOCK_SECONDS * sample_rate))
    hop_samples = int(round(HOP_SECONDS * sample_rate))
    power, _ = _block_power(weighted, block_samples, hop_samples)
    if power.size == 0:
        return None

    with np.errstate(divide="ignore"):
        block_loudness = LOUDNESS_OFFSET + 10.0 * np.log10(power)
    absolute = block_loudness > ABSOLUTE_GATE_LUFS
    if not np.any(absolute):
        return None

    absolute_power = float(np.mean(power[absolute]))
    if absolute_power <= 0.0:
        return None
    relative_gate = (
        LOUDNESS_OFFSET + 10.0 * math.log10(absolute_power) + RELATIVE_GATE_DB
    )
    gated = absolute & (block_loudness >= relative_gate)
    if not np.any(gated):
        return None
    final_power = float(np.mean(power[gated]))
    if final_power <= 0.0:
        return None
    return float(LOUDNESS_OFFSET + 10.0 * math.log10(final_power))


def integrated_loudness(
    samples: np.ndarray, sample_rate: int = SAMPLE_RATE
) -> Optional[float]:
    return integrated_loudness_from_weighted(
        k_filter_48k(samples), sample_rate=sample_rate
    )


def _normalised_correlation(
    reference: np.ndarray, track: np.ndarray, max_lag: int
) -> Tuple[np.ndarray, int]:
    """Return energy-normalised cross-correlation values and their first lag."""
    ref = np.asarray(reference, dtype=np.float64)
    trk = np.asarray(track, dtype=np.float64)
    ref = ref - np.mean(ref)
    trk = trk - np.mean(trk)

    n = min(ref.shape[0], trk.shape[0])
    ref = ref[:n]
    trk = trk[:n]
    nfft = 1
    while nfft < 2 * n + 1:
        nfft <<= 1

    ref_spectrum = np.fft.rfft(ref, n=nfft)
    trk_spectrum = np.fft.rfft(trk, n=nfft)
    cross = trk_spectrum * np.conj(ref_spectrum)
    correlation = np.fft.irfft(cross, n=nfft)

    max_lag = int(max(0, min(max_lag, n - 1)))
    # correlation[nfft-k] is lag -k, hence this slice is already in
    # increasing lag order from -max_lag through -1.
    negative = correlation[nfft - max_lag : nfft]
    positive = correlation[: max_lag + 1]
    values = np.concatenate((negative, positive))

    ref_energy = np.concatenate(
        ([0.0], np.cumsum(ref * ref, dtype=np.float64))
    )
    trk_energy = np.concatenate(
        ([0.0], np.cumsum(trk * trk, dtype=np.float64))
    )
    negative_lags = np.arange(-max_lag, 0, dtype=np.int64)
    positive_lags = np.arange(0, max_lag + 1, dtype=np.int64)
    negative_norm = np.sqrt(
        (ref_energy[n] - ref_energy[-negative_lags])
        * trk_energy[n + negative_lags]
    )
    positive_norm = np.sqrt(
        ref_energy[n - positive_lags]
        * (trk_energy[n] - trk_energy[positive_lags])
    )
    norms = np.concatenate((negative_norm, positive_norm))
    values = values / np.maximum(norms, 1e-20)
    return values, -max_lag


def estimate_offset_samples(reference: np.ndarray, track: np.ndarray) -> int:
    """Estimate integer delay of track content relative to reference.

    A positive result means the track content is delayed; aligning it requires
    removing that many leading samples (or adding samples when negative).
    The search is intentionally independent of any ground-truth timeline field.
    """
    ref = np.asarray(reference, dtype=np.float64)
    trk = np.asarray(track, dtype=np.float64)
    if ref.ndim > 1:
        ref = np.mean(ref, axis=1)
    if trk.ndim > 1:
        trk = np.mean(trk, axis=1)
    if ref.size == 0 or trk.size == 0:
        raise ValueError("cannot estimate offset for empty audio")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(trk)):
        raise ValueError("cannot estimate offset for non-finite audio")

    # ±1 second covers the specified ±300 ms with headroom and keeps the
    # search independent of timeline metadata.
    max_lag = min(SAMPLE_RATE, min(ref.size, trk.size) - 1)
    values, first_lag = _normalised_correlation(ref, trk, max_lag)
    return int(first_lag + int(np.argmax(values)))


def align_to_reference(
    samples: np.ndarray, offset_samples: int, output_length: int
) -> np.ndarray:
    """Crop or zero-pad a track so y[n] = x[n + offset] on a common timeline."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    output = np.zeros((output_length, x.shape[1]), dtype=np.float64)

    source_start = max(0, int(offset_samples))
    destination_start = max(0, -int(offset_samples))
    count = min(
        output_length - destination_start,
        x.shape[0] - source_start,
    )
    if count > 0:
        output[
            destination_start : destination_start + count, :
        ] = x[source_start : source_start + count, :]
    return output


def make_gain_envelope(
    scene_gains_db: Sequence[float],
    boundaries: Sequence[int],
    output_length: int,
    sample_rate: int = SAMPLE_RATE,
    fade_seconds: float = FADE_SECONDS,
) -> np.ndarray:
    """Build a linear crossfaded gain envelope centred on scene boundaries."""
    if not scene_gains_db:
        return np.ones(output_length, dtype=np.float64)
    gains = np.asarray([db_to_linear(v) for v in scene_gains_db], dtype=np.float64)
    envelope = np.full(output_length, gains[0], dtype=np.float64)

    interior = [
        int(boundary)
        for boundary in boundaries
        if 0 < int(boundary) < output_length
    ]
    edges = [0] + interior + [output_length]
    for index, gain in enumerate(gains):
        start = max(0, min(edges[index], output_length))
        end = max(start, min(edges[index + 1], output_length))
        envelope[start:end] = gain

    fade_samples = max(1, int(round(fade_seconds * sample_rate)))
    for index, boundary in enumerate(interior):
        if index + 1 >= gains.size:
            break
        left = max(0, boundary - fade_samples)
        right = min(output_length, boundary + fade_samples)
        if right <= left:
            continue
        positions = np.arange(left, right, dtype=np.float64)
        denominator = max(1.0, float(2 * fade_samples))
        t = np.clip((positions - (boundary - fade_samples)) / denominator, 0.0, 1.0)
        envelope[left:right] = gains[index] + (gains[index + 1] - gains[index]) * t
    return envelope


def soft_limit(samples: np.ndarray, knee: float = LIMITER_KNEE) -> np.ndarray:
    """Apply a smooth knee that asymptotically limits peaks to 1.0."""
    x = np.asarray(samples, dtype=np.float64)
    magnitude = np.abs(x)
    limited = magnitude.copy()
    over = magnitude > knee
    limited[over] = knee + (1.0 - knee) * np.tanh(
        (magnitude[over] - knee) / (1.0 - knee)
    )
    return np.sign(x) * limited


def peak_dbfs(samples: np.ndarray) -> float:
    return linear_to_db(float(np.max(np.abs(samples))) if np.size(samples) else 0.0)


def detect_clipped(samples: np.ndarray, minimum_samples: int = 8) -> bool:
    x = np.asarray(samples, dtype=np.float64)
    return int(np.count_nonzero(np.abs(x) >= 0.999)) >= minimum_samples


def detect_silent(
    samples: np.ndarray, threshold_dbfs: float = -80.0
) -> bool:
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return True
    return peak_dbfs(x) <= threshold_dbfs


def scene_loudness_values(
    aligned_mix: np.ndarray,
    scenes: Sequence[Dict[str, object]],
    sample_rate: int = SAMPLE_RATE,
) -> List[Optional[float]]:
    """Measure each scene from one K-filtered aligned pre-gain mix."""
    weighted = k_filter_48k(aligned_mix)
    values: List[Optional[float]] = []
    for scene in scenes:
        start = int(scene["start_sample"])
        end = int(scene["end_sample"])
        start = max(0, min(start, weighted.shape[0]))
        end = max(start, min(end, weighted.shape[0]))
        values.append(
            integrated_loudness_from_weighted(
                weighted[start:end, :], sample_rate=sample_rate
            )
        )
    return values
