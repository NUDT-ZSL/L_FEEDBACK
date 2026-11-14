"""ITU-R BS.1770 channel weights for common WAV channel orders.

WAV files store channels in interleaved order. For a 5.1 WAVE/WAVEX file the
conventional order is L, R, C, LFE, Ls, Rs. BS.1770 assigns unity weight to
all of these except the low-frequency-effects channel, whose weight is zero.
"""

from __future__ import annotations

from typing import List


def channel_weights(channel_count: int) -> List[float]:
    """Return BS.1770 weights without changing channel order."""

    if channel_count <= 0:
        raise ValueError("channel_count must be positive")
    if channel_count == 1:
        return [1.0]
    if channel_count == 2:
        return [1.0, 1.0]
    if channel_count == 3:
        return [1.0, 1.0, 1.0]
    if channel_count == 4:
        return [1.0, 1.0, 1.0, 1.0]
    if channel_count == 5:
        return [1.0, 1.0, 1.0, 1.0, 1.0]
    if channel_count == 6:
        return [1.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    if channel_count == 7:
        return [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
    if channel_count == 8:
        return [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    return [1.0] * channel_count

