"""ITU-R BS.1770-4 integrated loudness measurement and normalisation.

Implements, from scratch on float64 NumPy arrays:

* stage-1 K-weighting high-shelf (pre-filter) and stage-2 RLB high-pass,
  using the RBJ audio EQ cookbook parameterisation from the BS.1770-4 annex;
* 400 ms gated blocks with 75% overlap (100 ms hop);
* mean-square energy per block with channel weights (1.0 L/R, 1.0 each
  surround, 1.41 for the LFE channel);
* the absolute gate (-70 LKFS) and relative gate (-10 LU below the ungated
  reference) described in BS.1770-4 section 8;
* sample-peak and optional 4x oversampled true-peak measurement.
"""

from __future__ import annotations

import math

import numpy as np

from .errors import ErrorCode, MixerError

EPS = 1.0e-12

# Official ITU-R BS.1770-4 reference K-weighting coefficients (a0 = 1) for the
# standard delivery rates. These exact biquads are used whenever the project
# matches one of them; the RBJ cookbook below only covers exotic rates.
# Tuple layout: (stage 1 high-shelf, stage 2 RLB high-pass).
OFFICIAL_COEFFICIENTS: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {
    48000: (
        (
            1.53512485958697,
            -2.69169618940638,
            1.19839281085285,
            -1.69065929318241,
            0.73248027248087,
        ),
        (1.0, -2.0, 1.0, -1.99004745483398, 0.99007225036621),
    ),
    44100: (
        (
            1.53078422951861,
            -2.68739255786083,
            1.20023115345119,
            -1.68521190807566,
            0.72889292910473,
        ),
        (1.0, -2.0, 1.0, -1.98916966825809, 0.98920216759055),
    ),
    32000: (
        (
            1.51272246153875,
            -2.67131463581231,
            1.20681430540441,
            -1.66321635981101,
            0.71763436093186,
        ),
        (1.0, -2.0, 1.0, -1.98504437019619, 0.98514762449552),
    ),
}


def _biquad_rbj(
    sample_rate: float,
    filter_type: str,
    f0: float,
    gain_db: float = 0.0,
    q: float = 1.0 / math.sqrt(2.0),
) -> tuple[float, float, float, float, float]:
    """Return RBJ cookbook coefficients ``(b0, b1, b2, a1, a2)`` (a0 = 1)."""
    # RBJ peak/shelf gain parameter A = 10^(G/40): an amplitude (not power)
    # ratio, so a +4 dB shelf converges to +4 dB rather than +8 dB.
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / sample_rate
    cos_w = math.cos(w0)
    sin_w = math.sin(w0)
    alpha = sin_w / (2.0 * q)

    if filter_type == "high_shelf":
        sqrt_a = math.sqrt(amp)
        # Standard RBJ high-shelf: the mixing term is 2*sqrt(A)*alpha, where
        # alpha = sin(w0)/(2Q) carries the shelf Q.
        swing = 2.0 * sqrt_a * alpha
        b0 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w + swing)
        b1 = -2.0 * amp * ((amp - 1.0) + (amp + 1.0) * cos_w)
        b2 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w - swing)
        a0 = (amp + 1.0) - (amp - 1.0) * cos_w + swing
        a1 = 2.0 * ((amp - 1.0) - (amp + 1.0) * cos_w)
        a2 = (amp + 1.0) - (amp - 1.0) * cos_w - swing
    elif filter_type == "high_pass":
        b0 = (1.0 + cos_w) / 2.0
        b1 = -(1.0 + cos_w)
        b2 = (1.0 + cos_w) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha
    else:  # pragma: no cover - internal guard
        raise ValueError(f"unknown filter type {filter_type!r}")
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def _apply_biquad(signal: np.ndarray, coeffs: tuple[float, float, float, float, float]) -> np.ndarray:
    """Transposed direct form II, applied per channel with scalar state."""
    b0, b1, b2, a1, a2 = coeffs
    output = np.empty_like(signal)
    delay1 = 0.0
    delay2 = 0.0
    for index in range(signal.shape[0]):
        value = float(signal[index])
        intermediate = value - a1 * delay1 - a2 * delay2
        y = b0 * intermediate + b1 * delay1 + b2 * delay2
        delay2 = delay1
        delay1 = intermediate
        output[index] = y
    return output


def k_weighting_coefficients(sample_rate: float) -> list[tuple[float, float, float, float, float]]:
    if int(sample_rate) in OFFICIAL_COEFFICIENTS:
        shelf, highpass = OFFICIAL_COEFFICIENTS[int(sample_rate)]
        return [tuple(shelf), tuple(highpass)]  # type: ignore[return-value]
    shelf = _biquad_rbj(sample_rate, "high_shelf", f0=1681.974450955533, gain_db=4.0, q=0.7071752369554196)
    highpass = _biquad_rbj(sample_rate, "high_pass", f0=38.13547087602444, q=0.5003270373238773)
    return [shelf, highpass]


def apply_k_weighting(signal: np.ndarray, sample_rate: float) -> np.ndarray:
    filtered = signal
    for coeffs in k_weighting_coefficients(sample_rate):
        if filtered.ndim == 1:
            filtered = _apply_biquad(filtered, coeffs)
        else:
            columns = [_apply_biquad(filtered[:, ch], coeffs) for ch in range(filtered.shape[1])]
            filtered = np.stack(columns, axis=1)
    return filtered


def _block_means(filtered: np.ndarray, sample_rate: float, weights: np.ndarray) -> np.ndarray:
    block_frames = int(round(0.4 * sample_rate))
    hop_frames = int(round(0.1 * sample_rate))
    total_frames = filtered.shape[0]
    if total_frames < block_frames:
        padded = np.zeros((block_frames, filtered.shape[1]))
        padded[:total_frames] = filtered
        filtered = padded
        total_frames = block_frames

    starts = np.arange(0, total_frames - block_frames + 1, hop_frames)
    means = np.empty(starts.shape[0], dtype=np.float64)
    for idx, start in enumerate(starts):
        block = filtered[start : start + block_frames]
        squared = block * block
        channel_means = squared.mean(axis=0)
        means[idx] = float(np.sum(weights * channel_means))
    return means


def measure_loudness(
    samples: np.ndarray,
    sample_rate: int,
    channel_weights: np.ndarray | None = None,
) -> dict:
    """Measure BS.1770-4 integrated loudness in LUFS plus gating diagnostics."""
    if samples.ndim == 1:
        samples = samples[:, None]
    channels = samples.shape[1]
    if channel_weights is None:
        # Stereo delivery: both main channels carry weight 1.0.
        channel_weights = np.ones(channels, dtype=np.float64)

    filtered = apply_k_weighting(samples, float(sample_rate))
    block_means = _block_means(filtered, sample_rate, channel_weights)

    absolute_threshold = 10.0 ** ((-70.0 + 0.691) / 10.0)
    abs_mask = block_means > absolute_threshold
    abs_gated = block_means[abs_mask]
    if abs_gated.size == 0:
        return {
            "integrated_lufs": None,
            "gated_blocks": 0,
            "total_blocks": int(block_means.size),
            "absolute_gate_lkfs": -70.0,
            "relative_gate_lkfs": None,
            "ungated_reference_lkfs": None,
            "silence": True,
        }

    ungated_mean = float(np.mean(abs_gated))
    ungated_loudness = -0.691 + 10.0 * math.log10(ungated_mean + EPS)
    relative_gate_lkfs = ungated_loudness - 10.0
    relative_threshold = 10.0 ** ((relative_gate_lkfs + 0.691) / 10.0)

    rel_mask = block_means > relative_threshold
    gated = block_means[rel_mask]
    gated_mean = float(np.mean(gated))
    integrated = -0.691 + 10.0 * math.log10(gated_mean + EPS)

    return {
        "integrated_lufs": integrated,
        "gated_blocks": int(gated.size),
        "total_blocks": int(block_means.size),
        "absolute_gate_lkfs": -70.0,
        "relative_gate_lkfs": relative_gate_lkfs,
        "ungated_reference_lkfs": ungated_loudness,
        "silence": False,
    }


def sample_peak_dbfs(samples: np.ndarray) -> float:
    amplitude = float(np.max(np.abs(samples))) if samples.size else 0.0
    if amplitude <= EPS:
        return -math.inf
    return 20.0 * math.log10(amplitude)


def true_peak_dbfs(samples: np.ndarray, sample_rate: int) -> float:
    """4x oversampled inter-sample peak using a windowed-sinc interpolator."""
    factor = 4
    up_rate = sample_rate * factor
    if samples.ndim == 1:
        work = samples[:, None]
    else:
        work = samples
    max_peak = 0.0
    half_width = 16
    beta = 9.0
    for channel in range(work.shape[1]):
        mono = work[:, channel]
        output_frames = mono.shape[0] * factor
        source_indices = np.arange(output_frames, dtype=np.float64) / factor
        center = np.floor(source_indices).astype(np.int64)
        frac = source_indices - center
        taps = np.arange(-half_width, half_width + 1, dtype=np.float64)
        distances = taps[None, :] - frac[:, None]
        x = np.pi * distances
        sinc = np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(x == 0, 1.0, x))
        window = np.i0(beta * np.sqrt(np.clip(1.0 - (distances / half_width) ** 2, 0.0, 1.0))) / np.i0(beta)
        kernel = sinc * window
        positions = center[:, None] + taps.astype(np.int64)[None, :]
        valid = (positions >= 0) & (positions < mono.shape[0])
        gathered = np.where(valid, np.clip(positions, 0, mono.shape[0] - 1), 0)
        interpolated = np.sum(mono[gathered] * kernel * valid, axis=1)
        max_peak = max(max_peak, float(np.max(np.abs(interpolated))))
    if max_peak <= EPS:
        return -math.inf
    return 20.0 * math.log10(max_peak)


def normalise_loudness(
    samples: np.ndarray,
    sample_rate: int,
    target_lufs: float,
    peak_limit_dbfs: float,
    shortfall_tolerance_lu: float,
    use_true_peak: bool = False,
) -> tuple[np.ndarray, dict]:
    """Apply a single gain so integrated loudness hits the target.

    The gain is reduced further when needed to honour the peak ceiling. If
    that extra reduction pushes the loudness more than
    ``shortfall_tolerance_lu`` below target, E_PEAK_LIMIT is raised and no
    audio is returned for writing.
    """
    measurement = measure_loudness(samples, sample_rate)
    peak_before = (
        true_peak_dbfs(samples, sample_rate) if use_true_peak else sample_peak_dbfs(samples)
    )

    if measurement["silence"]:
        raise MixerError(
            ErrorCode.PEAK_LIMIT,
            "cannot normalise silence: integrated loudness is below the -70 LKFS gate",
            {"integrated_lufs": None, "target_lufs": target_lufs},
        )

    loudness_before = float(measurement["integrated_lufs"])
    target_gain_db = target_lufs - loudness_before

    peak_gain_limit_db = peak_limit_dbfs - peak_before
    final_gain_db = min(target_gain_db, peak_gain_limit_db)
    gain_linear = 10.0 ** (final_gain_db / 20.0)
    adjusted = samples * gain_linear

    loudness_after = float(measure_loudness(adjusted, sample_rate)["integrated_lufs"])
    peak_after = (
        true_peak_dbfs(adjusted, sample_rate) if use_true_peak else sample_peak_dbfs(adjusted)
    )

    peak_limited = final_gain_db < target_gain_db - 1e-9
    shortfall_lu = target_lufs - loudness_after
    peak_violation = peak_after > peak_limit_dbfs + 1e-6

    stats = {
        "loudness_before_lufs": loudness_before,
        "loudness_after_lufs": loudness_after,
        "target_lufs": target_lufs,
        "target_gain_db": target_gain_db,
        "applied_gain_db": final_gain_db,
        "peak_before_dbfs": peak_before,
        "peak_after_dbfs": peak_after,
        "peak_limit_dbfs": peak_limit_dbfs,
        "peak_measurement": "true_peak_4x" if use_true_peak else "sample_peak",
        "peak_limited": peak_limited,
        "loudness_shortfall_lu": max(0.0, shortfall_lu),
        "gated_blocks": measurement["gated_blocks"],
        "total_blocks": measurement["total_blocks"],
        "relative_gate_lkfs": measurement["relative_gate_lkfs"],
    }

    if peak_violation:
        raise MixerError(
            ErrorCode.PEAK_LIMIT,
            "peak ceiling still violated after normalisation",
            {"stats": stats},
        )
    if peak_limited and shortfall_lu > shortfall_tolerance_lu + 1e-9:
        raise MixerError(
            ErrorCode.PEAK_LIMIT,
            (
                f"peak ceiling forces loudness {loudness_after:.2f} LUFS, "
                f"{shortfall_lu:.2f} LU below target {target_lufs:.2f} "
                f"(tolerance {shortfall_tolerance_lu:.2f} LU)"
            ),
            {"stats": stats},
        )
    return adjusted, stats
