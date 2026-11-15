"""Deterministic polyphase resampling with anti-alias filtering.

A windowed-sinc (Kaiser) interpolation/decimation filter is designed with
fixed parameters, so identical input always yields bit-identical output.
The filter is applied through a polyphase matrix product (BLAS), which is
fast and keeps group delay compensated so the output stays time aligned.
"""

from __future__ import annotations

import math

import numpy as np

# Fixed filter design parameters (do not change: determinism + spec).
HALF_WIDTH = 32          # zero crossings per side -> transition bandwidth
KAISER_BETA = 8.6        # ~60 dB stop-band attenuation
CUTOFF_SCALE = 0.95      # fraction of the lower Nyquist frequency


def _design_filter(up: int, down: int) -> np.ndarray:
    rate = max(up, down)
    n = 2 * HALF_WIDTH * rate + 1
    t = np.arange(n, dtype=np.float64) - (n - 1) // 2
    fc = CUTOFF_SCALE * 0.5 / rate  # cycles per upsampled sample
    h = np.sinc(2.0 * fc * t) * np.kaiser(n, KAISER_BETA)
    h *= up / h.sum()  # unity DC gain after interpolation
    return h


def _upfirdn(x: np.ndarray, h: np.ndarray, up: int, down: int,
             out_len: int) -> np.ndarray:
    tph = -(-len(h) // up)  # taps per polyphase branch
    hp = np.zeros(tph * up, dtype=np.float64)
    hp[: len(h)] = h
    branches = hp.reshape(tph, up).T  # branches[p, k] = h[k*up + p]
    xpad = np.concatenate([np.zeros(tph - 1), x, np.zeros(tph - 1)])
    windows = np.lib.stride_tricks.sliding_window_view(xpad, tph)
    # windows[q] == x[q-tph+1 .. q]; reversed it lines up with the branch.
    gd = (len(h) - 1) // 2  # integer group delay in upsampled samples
    rows = gd + np.arange(out_len, dtype=np.int64) * down
    out = np.empty(out_len, dtype=np.float64)
    chunk = 65536
    for start in range(0, out_len, chunk):
        r = rows[start:start + chunk]
        q, p = r // up, r % up
        w = windows[q][:, ::-1]
        out[start:start + chunk] = np.einsum("mk,mk->m", w, branches[p],
                                             optimize=True)
    return out


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample (n, ch) float64 audio from sr_in to sr_out Hz."""
    sr_in, sr_out = int(sr_in), int(sr_out)
    if sr_in == sr_out:
        return x
    g = math.gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    h = _design_filter(up, down)
    n = x.shape[0]
    out_len = (2 * n * up + down) // (2 * down)  # round(n * sr_out / sr_in)
    out = np.empty((out_len, x.shape[1]), dtype=np.float64)
    for c in range(x.shape[1]):
        out[:, c] = _upfirdn(np.ascontiguousarray(x[:, c]), h, up, down,
                             out_len)
    return out
