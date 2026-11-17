"""Whole-bar loop extension with equal-length crossfade seams."""

from __future__ import annotations

import numpy as np


def loop_extend(
    data: np.ndarray,
    target_len: int,
    bar_len: int,
    crossfade_len: int,
) -> tuple[np.ndarray, int]:
    """Extend ``data`` to ``target_len`` samples by looping whole bars.

    Each appended copy starts on a bar boundary of the source material
    (the source is assumed to contain an integer number of bars). Seams
    use an equal-length linear crossfade of ``crossfade_len`` samples so
    the splice is sample-level continuous for bar-periodic content.

    Returns (extended_data[:target_len], number_of_appended_loops).
    """
    if data.shape[0] >= target_len:
        return data[:target_len].copy(), 0
    if bar_len <= 0 or data.shape[0] % bar_len != 0:
        raise ValueError("source length must be a positive multiple of bar_len")
    crossfade_len = min(crossfade_len, bar_len // 2)

    fade_in = (np.arange(crossfade_len, dtype=np.float64) / crossfade_len)[:, None]
    fade_out = 1.0 - fade_in

    out = data
    loops = 0
    while out.shape[0] - crossfade_len < target_len:
        seam = out[-crossfade_len:] * fade_out + data[:crossfade_len] * fade_in
        out = np.concatenate([out[:-crossfade_len], seam, data[crossfade_len:]], axis=0)
        loops += 1
    return out[:target_len].copy(), loops
