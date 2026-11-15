"""ITU-R BS.1770 integrated loudness measurement at a fixed 48 kHz rate.

The K-weighting filters (pre-filter shelf + RLB high-pass) use the fixed
48 kHz coefficients from the standard. Because all audio is resampled to
48 kHz before measurement, these coefficients are always valid and the
result is deterministic. The cascaded biquads are realised as a truncated
impulse response (8192 taps; residual energy < 1e-9) applied by linear
convolution, avoiding any dependence on external DSP libraries.

Fixed gating parameters (BS.1770-4):
block length 400 ms, hop 100 ms (75 % overlap),
absolute gate -70 LKFS, relative gate -10 LU.
"""

from __future__ import annotations

import threading

import numpy as np

SAMPLE_RATE = 48000
BLOCK_SEC = 0.400
HOP_SEC = 0.100
ABS_GATE_LKFS = -70.0
REL_GATE_LU = 10.0
LOUDNESS_OFFSET = -0.691

# ITU-R BS.1770 K-weighting coefficients for 48 kHz.
_PRE_B = (1.53512485958697, -2.69169618940638, 1.19839281085285)
_PRE_A = (1.0, -1.69065929318241, 0.73248077421585)
_RLB_B = (1.0, -2.0, 1.0)
_RLB_A = (1.0, -1.99004745483398, 0.99007225036621)

# Channel weights G_i (BS.1770): fronts 1.0, surrounds 1.41, LFE excluded.
# 6-channel order follows the WAV default: FL FR FC LFE BL/SL BR/SR.
CHANNEL_WEIGHTS = {
    1: (1.0,),
    2: (1.0, 1.0),
    6: (1.0, 1.0, 1.0, 0.0, 1.41, 1.41),
}

_IR_TAPS = 8192
_k_ir = None
_k_ir_lock = threading.Lock()


def _biquad_impulse(b, a, n: int) -> np.ndarray:
    ir = np.zeros(n, dtype=np.float64)
    xm1 = xm2 = ym1 = ym2 = 0.0
    for i in range(n):
        x0 = 1.0 if i == 0 else 0.0
        y0 = b[0] * x0 + b[1] * xm1 + b[2] * xm2 - a[1] * ym1 - a[2] * ym2
        ir[i] = y0
        xm2, xm1 = xm1, x0
        ym2, ym1 = ym1, y0
    return ir


def _k_weight_ir() -> np.ndarray:
    global _k_ir
    with _k_ir_lock:
        if _k_ir is None:
            pre = _biquad_impulse(_PRE_B, _PRE_A, _IR_TAPS)
            rlb = _biquad_impulse(_RLB_B, _RLB_A, _IR_TAPS)
            _k_ir = np.convolve(pre, rlb)[:_IR_TAPS]
    return _k_ir


def k_filter(x: np.ndarray) -> np.ndarray:
    """Apply the K-weighting filter to (n, ch) audio at 48 kHz."""
    ir = _k_weight_ir()
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        out[:, c] = np.convolve(x[:, c], ir)[: x.shape[0]]
    return out


def integrated_loudness(x: np.ndarray, weights) -> float:
    """BS.1770 integrated loudness (LKFS) of (n, ch) 48 kHz audio.

    Returns -inf when every gated block falls below the absolute gate
    (silence). Segments shorter than one 400 ms block are measured as a
    single block over the available samples.
    """
    xf = k_filter(x)
    n = xf.shape[0]
    block = int(round(BLOCK_SEC * SAMPLE_RATE))
    hop = int(round(HOP_SEC * SAMPLE_RATE))
    if n >= block:
        starts = np.arange(0, n - block + 1, hop)
        cs = np.vstack([np.zeros((1, xf.shape[1])), np.cumsum(xf ** 2, axis=0)])
        z = (cs[starts + block] - cs[starts]) / block
    else:
        z = (xf ** 2).mean(axis=0, keepdims=True)
    w = np.asarray(weights, dtype=np.float64)
    block_power = z @ w
    with np.errstate(divide="ignore"):
        block_loud = LOUDNESS_OFFSET + 10.0 * np.log10(block_power)
    mask = block_loud > ABS_GATE_LKFS
    if not mask.any():
        return float("-inf")
    mean_power = block_power[mask].mean()
    rel_threshold = LOUDNESS_OFFSET + 10.0 * np.log10(mean_power) - REL_GATE_LU
    mask &= block_loud > rel_threshold
    if not mask.any():
        return float("-inf")
    return float(LOUDNESS_OFFSET + 10.0 * np.log10(block_power[mask].mean()))
