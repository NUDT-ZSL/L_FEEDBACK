"""Sample-rate conversion using a windowed-sinc (Kaiser) interpolator.

The filter is a finite-support low-pass sinc multiplied by a Kaiser window:

* half the taps fall on each side of the resampling position;
* the cutoff is ``0.5 * min(1, ratio)`` so it acts as an anti-alias filter on
  downsampling and a reconstruction filter on upsampling;
* the kernel is renormalized per output sample to unity DC gain.

Edge samples are taken from the clamped source index, which keeps output
lengths exact and deterministic.  Computation is vectorized in fixed-size
blocks so memory use stays bounded regardless of signal length.
"""

from __future__ import annotations

import numpy as np

# Total kernel taps (even => symmetric half-support).
NUM_TAPS = 64
# Kaiser beta ~ 9.5 gives roughly 90 dB stop-band attenuation.
KAISER_BETA = 9.5
# Output samples processed per vectorized block.
BLOCK_SIZE = 8192


def resample(signal: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 2-D ``(frames, channels)`` array to ``target_sr``.

    Returns a C-contiguous float32 array with
    ``round(signal.shape[0] * target_sr / source_sr)`` frames.
    """
    if signal.ndim != 2:
        raise ValueError("resample() expects a 2-D (frames, channels) array")

    source = np.ascontiguousarray(signal, dtype=np.float32)
    if source_sr == target_sr:
        return source.copy()

    ratio = float(target_sr) / float(source_sr)
    out_len = int(np.floor(round(source.shape[0] * ratio)))
    channels = source.shape[1]
    output = np.empty((out_len, channels), dtype=np.float32)

    half = NUM_TAPS // 2
    cutoff = 0.5 * min(1.0, ratio)
    # Symmetric integer tap offsets relative to the (fractional) source index.
    taps = np.arange(-half + 1, half + 1, dtype=np.float64)
    window = np.i0(KAISER_BETA * np.sqrt(np.maximum(0.0, 1.0 - (taps / half) ** 2)))

    for block_start in range(0, out_len, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, out_len)
        out_idx = np.arange(block_start, block_end, dtype=np.float64)
        # Exact fractional source position for every output frame.
        src_pos = out_idx / ratio
        src_base = np.floor(src_pos)
        fractional = src_pos - src_base

        # Kernel matrix: rows are output frames, columns are taps.
        x = taps[np.newaxis, :] - fractional[:, np.newaxis]
        kernel = np.sinc(2.0 * cutoff * x) * (2.0 * cutoff)
        kernel *= window[np.newaxis, :]
        kernel /= kernel.sum(axis=1, keepdims=True)

        src_idx = src_base[:, np.newaxis].astype(np.int64) + taps.astype(np.int64)
        np.clip(src_idx, 0, source.shape[0] - 1, out=src_idx)
        gathered = source[src_idx]  # (block, taps, channels)
        block = np.einsum("btc,bt->bc", gathered.astype(np.float64), kernel)
        output[block_start:block_end] = block.astype(np.float32)

    return np.ascontiguousarray(output, dtype=np.float32)
