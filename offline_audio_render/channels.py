"""Channel-layout adaptation: upmix, downmix and spatial expansion."""

from __future__ import annotations

import numpy as np

from .errors import W_DOWNMIX, W_UNSUPPORTED_CHANNELS
from .spatial import LAYOUT_51, LAYOUT_MONO, LAYOUT_STEREO

# Downmix coefficient tables, rows = output channels, cols = source channels.
# 5.1 order is (L, R, C, LFE, Ls, Rs).
DOWNMIX_6_TO_2 = np.array(
    [
        # L out
        [1.0, 0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0],
        # R out
        [0.0, 1.0, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
    ],
    dtype=np.float64,
)

DOWNMIX_6_TO_1 = np.array(
        [
            [
                np.sqrt(0.5),
                np.sqrt(0.5),
                1.0,
                0.0,
                np.sqrt(0.5),
                np.sqrt(0.5),
            ]
        ],
    dtype=np.float64,
)

DOWNMIX_2_TO_1 = np.array([[np.sqrt(0.5), np.sqrt(0.5)]], dtype=np.float64)


def adapt_block(
    block: np.ndarray,
    source_channels: int,
    layout: str,
    spatial_gains: np.ndarray,
) -> tuple[np.ndarray, str | None]:
    """Adapt a ``(frames, source_channels)`` block to the output layout.

    Mono sources are spatialised directly. Multi-channel sources are downmixed
    first when they exceed the output layout, then distributed without extra
    spatial gain (their own channel image already carries the image).
    Returns the adapted block and a warning code when downmixing occurred.
    """
    out_channels = {
        LAYOUT_MONO: 1,
        LAYOUT_STEREO: 2,
        LAYOUT_51: 6,
    }[layout]

    if source_channels == 1:
        mono = block[:, 0] if block.ndim == 2 else block
        return mono[:, None] * spatial_gains[None, :out_channels], None

    if source_channels > out_channels:
        matrix = _downmix_matrix(source_channels, out_channels)
        if matrix is None:
            silence = np.zeros((block.shape[0], out_channels), dtype=np.float64)
            return silence, W_UNSUPPORTED_CHANNELS
        downmixed = block @ matrix.T
        return downmixed * spatial_gains[None, :out_channels], W_DOWNMIX

    if source_channels == 6 and out_channels == 6:
        # Re-fold a surround source through the scene's spatial image; the
        # LFE bed is intentionally discarded because no scene LFE is modelled.
        folded = block.copy().astype(np.float64)
        folded[:, 3] = 0.0
        return folded * spatial_gains[None, :], W_DOWNMIX

    # Fewer channels than the output layout.
    if source_channels == 2 and out_channels == 6:
        expanded = np.zeros((block.shape[0], 6), dtype=np.float64)
        expanded[:, 0] = block[:, 0]
        expanded[:, 1] = block[:, 1]
        return expanded * spatial_gains[None, :], None
    if source_channels == 2 and out_channels == 1:
        downmixed = block @ DOWNMIX_2_TO_1.T
        return downmixed * spatial_gains[None, :], W_DOWNMIX
    if source_channels == out_channels:
        return block.astype(np.float64, copy=True) * spatial_gains[None, :], None
    raise ValueError(
        f"unsupported channel adaptation {source_channels} -> {out_channels}"
    )


def _downmix_matrix(source_channels: int, out_channels: int) -> np.ndarray | None:
    if source_channels == 6 and out_channels == 2:
        return DOWNMIX_6_TO_2
    if source_channels == 6 and out_channels == 1:
        return DOWNMIX_6_TO_1
    if source_channels == 2 and out_channels == 1:
        return DOWNMIX_2_TO_1
    return None
