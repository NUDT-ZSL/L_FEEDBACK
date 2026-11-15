from __future__ import annotations

import math

import numpy as np


def sinc_resample(
    samples: np.ndarray,
    input_rate: int,
    output_rate: int,
    zero_crossings: int = 24,
    rolloff: float = 0.95,
    chunk: int = 4096,
) -> np.ndarray:
    """Windowed-sinc resampling with an anti-alias low-pass for downsampling."""

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if input_rate == output_rate:
        return x.copy()
    if input_rate <= 0 or output_rate <= 0:
        raise ValueError("sample rates must be positive")

    n_in, n_channels = x.shape
    n_out = int(round(n_in * output_rate / input_rate))
    if n_out <= 0:
        return np.zeros((0, n_channels), dtype=np.float64)

    cutoff = min(1.0, output_rate / input_rate) * rolloff
    half_width = max(8, int(math.ceil(zero_crossings / cutoff)))
    offsets = np.arange(-half_width + 1, half_width + 1, dtype=np.int64)
    output = np.zeros((n_out, n_channels), dtype=np.float64)

    for start in range(0, n_out, chunk):
        stop = min(start + chunk, n_out)
        positions = np.arange(start, stop, dtype=np.float64) * input_rate / output_rate
        centers = np.floor(positions).astype(np.int64)
        indices = centers[:, None] + offsets[None, :]
        distances = positions[:, None] - indices
        weights = np.sinc(distances * cutoff) * cutoff
        weights *= 0.5 + 0.5 * np.cos(np.pi * distances / half_width)
        valid = (indices >= 0) & (indices < n_in) & (np.abs(distances) <= half_width)
        clipped = np.clip(indices, 0, n_in - 1)
        weight_sum = np.sum(weights * valid, axis=1)
        for channel in range(n_channels):
            values = x[clipped, channel] * valid
            num = np.sum(weights * values, axis=1)
            output[start:stop, channel] = np.divide(
                num,
                weight_sum,
                out=np.zeros_like(num),
                where=np.abs(weight_sum) > 1e-15,
            )
    return output
