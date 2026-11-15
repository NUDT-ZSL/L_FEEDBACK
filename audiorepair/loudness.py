"""ITU-R BS.1770 integrated loudness at 44.1 or 48 kHz.

The implementation uses the standard K-weighting biquad pairs, the exact
400 ms / 100 ms gating blocks, the -70 LKFS absolute gate and the -10 LU
relative gate.  Short segments are measured as one available block so that
they remain usable for local normalisation instead of disappearing behind
the block gate.
"""

from __future__ import annotations

import threading
from typing import Dict, Sequence, Tuple

import numpy as np

BLOCK_SECONDS = 0.400
HOP_SECONDS = 0.100
ABSOLUTE_GATE_LKFS = -70.0
RELATIVE_GATE_LU = 10.0
LOUDNESS_OFFSET = -0.691
IR_TAPS = 8192

_COEFFICIENTS: Dict[int, Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]] = {
    44100: (
        (1.5308412300498355, -2.650979995154729, 1.169079079921587),
        (1.0, -1.6636551132560204, 0.7125954280732254),
        (1.0, -2.0, 1.0),
        (1.0, -1.9891696736297957, 0.9891990357870394),
    ),
    48000: (
        (1.53512485958697, -2.69169618940638, 1.19839281085285),
        (1.0, -1.69065929318241, 0.73248077421585),
        (1.0, -2.0, 1.0),
        (1.0, -1.99004745483398, 0.99007225036621),
    ),
}

CHANNEL_WEIGHTS = {
    1: (1.0,),
    2: (1.0, 1.0),
}

_ir_cache: Dict[int, np.ndarray] = {}
_cache_lock = threading.Lock()


def _biquad_impulse(
    b: Sequence[float], a: Sequence[float], length: int
) -> np.ndarray:
    ir = np.zeros(length, dtype=np.float64)
    xm1 = 0.0
    xm2 = 0.0
    ym1 = 0.0
    ym2 = 0.0
    for index in range(length):
        x0 = 1.0 if index == 0 else 0.0
        y0 = (
            b[0] * x0
            + b[1] * xm1
            + b[2] * xm2
            - a[1] * ym1
            - a[2] * ym2
        )
        ir[index] = y0
        xm2, xm1 = xm1, x0
        ym2, ym1 = ym1, y0
    return ir


def _k_weight_impulse(sample_rate: int) -> np.ndarray:
    with _cache_lock:
        cached = _ir_cache.get(sample_rate)
        if cached is not None:
            return cached
        if sample_rate not in _COEFFICIENTS:
            raise ValueError("BS.1770 coefficients are unavailable")
        pre_b, pre_a, rlb_b, rlb_a = _COEFFICIENTS[sample_rate]
        pre = _biquad_impulse(pre_b, pre_a, IR_TAPS)
        rlb = _biquad_impulse(rlb_b, rlb_a, IR_TAPS)
        combined = np.convolve(pre, rlb)[:IR_TAPS]
        combined.setflags(write=False)
        _ir_cache[sample_rate] = combined
        return combined


def _fft_convolve_same(x: np.ndarray, impulse: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.copy()
    total = x.size + impulse.size - 1
    fft_size = 1 << (int(total - 1).bit_length())
    spectrum = np.fft.rfft(x, n=fft_size) * np.fft.rfft(impulse, n=fft_size)
    return np.fft.irfft(spectrum, n=fft_size)[: x.size]


def k_weight(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply K-weighting to an (frames, channels) or (frames,) array."""
    audio = np.asarray(x, dtype=np.float64)
    was_mono = audio.ndim == 1
    if was_mono:
        audio = audio[:, None]
    if audio.ndim != 2:
        raise ValueError("audio must be one- or two-dimensional")
    impulse = _k_weight_impulse(sample_rate)
    filtered = np.empty_like(audio, dtype=np.float64)
    for channel in range(audio.shape[1]):
        filtered[:, channel] = _fft_convolve_same(audio[:, channel], impulse)
    return filtered[:, 0] if was_mono else filtered


def integrated_loudness(
    samples: np.ndarray,
    sample_rate: int,
    channel_weights: Sequence[float] = None,
) -> float:
    """Return integrated loudness in LKFS/LUFS, or ``-inf`` for silence."""
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[0] == 0:
        return float("-inf")
    if sample_rate not in _COEFFICIENTS:
        raise ValueError("unsupported BS.1770 sample rate")

    weights = (
        np.asarray(channel_weights, dtype=np.float64)
        if channel_weights is not None
        else np.asarray(CHANNEL_WEIGHTS.get(audio.shape[1], ()), dtype=np.float64)
    )
    if weights.size != audio.shape[1]:
        raise ValueError("channel weights do not match audio channels")

    filtered = k_weight(audio, sample_rate)
    block = int(round(BLOCK_SECONDS * sample_rate))
    hop = int(round(HOP_SECONDS * sample_rate))
    if audio.shape[0] >= block:
        starts = np.arange(0, audio.shape[0] - block + 1, hop, dtype=np.int64)
        cumulative = np.vstack(
            (
                np.zeros((1, filtered.shape[1]), dtype=np.float64),
                np.cumsum(filtered * filtered, axis=0, dtype=np.float64),
            )
        )
        block_power = (
            cumulative[starts + block] - cumulative[starts]
        ) / float(block)
    else:
        block_power = np.mean(filtered * filtered, axis=0, keepdims=True)

    weighted_power = block_power @ weights
    positive = weighted_power > 0.0
    if not np.any(positive):
        return float("-inf")
    block_loudness = np.full(weighted_power.shape, -np.inf, dtype=np.float64)
    block_loudness[positive] = LOUDNESS_OFFSET + 10.0 * np.log10(
        weighted_power[positive]
    )
    gated = block_loudness > ABSOLUTE_GATE_LKFS
    if not np.any(gated):
        return float("-inf")
    relative_loudness = (
        LOUDNESS_OFFSET + 10.0 * np.log10(weighted_power[gated].mean())
        - RELATIVE_GATE_LU
    )
    gated &= block_loudness > relative_loudness
    if not np.any(gated):
        return float("-inf")
    return float(
        LOUDNESS_OFFSET + 10.0 * np.log10(weighted_power[gated].mean())
    )


def sample_peak(samples: np.ndarray) -> float:
    audio = np.asarray(samples, dtype=np.float64)
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def peak_dbfs(samples: np.ndarray) -> float:
    peak = sample_peak(samples)
    if peak <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(peak))


def dc_offset(samples: np.ndarray) -> float:
    audio = np.asarray(samples, dtype=np.float64)
    if audio.size == 0:
        return 0.0
    return float(np.mean(audio))

