"""采样级线性重采样。"""

from __future__ import annotations

import numpy as np


def resample_linear(frames: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """将音频按线性插值重采样到目标采样率（采样级确定性实现）。

    ``frames`` 形状为 (n,) 或 (n, c)，输出形状为 (m,) 或 (m, c)，
    其中 m = ceil(n * dst_rate / src_rate)。采样率相同时原样返回。
    """
    if src_rate == dst_rate:
        return frames
    if frames.ndim == 1:
        return _resample_mono(frames, src_rate, dst_rate)
    channels = [
        _resample_mono(frames[:, channel], src_rate, dst_rate)
        for channel in range(frames.shape[1])
    ]
    return np.stack(channels, axis=1)


def _resample_mono(mono: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    src_len = int(mono.shape[0])
    if src_len == 0:
        return mono.astype(np.float64, copy=True)
    dst_len = int(np.ceil(src_len * dst_rate / src_rate))
    source_positions = np.arange(dst_len, dtype=np.float64) * (
        src_rate / dst_rate
    )
    index0 = np.floor(source_positions).astype(np.int64)
    index1 = np.minimum(index0 + 1, src_len - 1)
    fraction = source_positions - index0
    return mono[index0] * (1.0 - fraction) + mono[index1] * fraction

