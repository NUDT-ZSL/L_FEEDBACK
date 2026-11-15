"""Downmix, gain, limiting and timeline alignment DSP."""

from __future__ import annotations

import numpy as np

# ffmpeg standard channel layouts -> ordered channel names.
LAYOUTS: dict[str, list[str]] = {
    "mono": ["FC"],
    "stereo": ["FL", "FR"],
    "2.1": ["FL", "FR", "LFE"],
    "3.0": ["FL", "FR", "FC"],
    "3.0(back)": ["FL", "FR", "BC"],
    "4.0": ["FL", "FR", "FC", "BC"],
    "quad": ["FL", "FR", "BL", "BR"],
    "quad(side)": ["FL", "FR", "SL", "SR"],
    "3.1": ["FL", "FR", "FC", "LFE"],
    "5.0": ["FL", "FR", "FC", "BL", "BR"],
    "5.0(side)": ["FL", "FR", "FC", "SL", "SR"],
    "4.1": ["FL", "FR", "FC", "LFE", "BC"],
    "5.1": ["FL", "FR", "FC", "LFE", "BL", "BR"],
    "5.1(side)": ["FL", "FR", "FC", "LFE", "SL", "SR"],
    "6.0": ["FL", "FR", "FC", "BC", "SL", "SR"],
    "6.0(front)": ["FL", "FR", "FLC", "FRC", "SL", "SR"],
    "hexagonal": ["FL", "FR", "FC", "BL", "BR", "BC"],
    "6.1": ["FL", "FR", "FC", "LFE", "BC", "SL", "SR"],
    "6.1(back)": ["FL", "FR", "FC", "LFE", "BC", "SL", "SR"],
    "6.1(front)": ["FL", "FR", "FLC", "FRC", "LFE", "SL", "SR"],
    "7.0": ["FL", "FR", "FC", "BL", "BR", "SL", "SR"],
    "7.0(front)": ["FL", "FR", "FLC", "FRC", "SL", "SR"],
    "7.1": ["FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"],
    "7.1(wide)": ["FL", "FR", "FC", "LFE", "FLC", "FRC", "SL", "SR"],
}

SQRT1_2 = 0.7071067811865476


# ffmpeg's default guesses when a file carries no channel mask.
DEFAULT_LAYOUT_BY_COUNT = {
    1: "mono",
    2: "stereo",
    3: "2.1",
    4: "quad",
    5: "5.0",
    6: "5.1",
    7: "6.1",
    8: "7.1",
}


def layout_channels(layout: str, channels: int) -> list[str]:
    names = LAYOUTS.get(layout)
    if names and len(names) == channels:
        return names
    default = DEFAULT_LAYOUT_BY_COUNT.get(channels)
    if default:
        return LAYOUTS[default]
    # unknown layout: assume generic front pair + unnamed extras
    return ["FL", "FR"] + [f"CH{i}" for i in range(2, channels)]


def downmix_stereo(pcm: np.ndarray, layout: str) -> np.ndarray:
    """Downmix to stereo following ITU-R BS.775 coefficients."""
    n, ch = pcm.shape
    names = layout_channels(layout, ch)
    if ch == 1:
        return np.repeat(pcm, 2, axis=1)
    if ch == 2 and names == ["FL", "FR"]:
        return pcm.copy()
    left = np.zeros(n)
    right = np.zeros(n)
    for i, name in enumerate(names):
        c = pcm[:, i]
        if name == "FL":
            left += c
        elif name == "FR":
            right += c
        elif name in ("FC", "BC", "TC"):
            left += SQRT1_2 * c
            right += SQRT1_2 * c
        elif name in ("BL", "SL", "FLC", "SSL", "WL", "SDL"):
            left += SQRT1_2 * c
        elif name in ("BR", "SR", "FRC", "SSR", "WR", "SDR"):
            right += SQRT1_2 * c
        # LFE and unknown channels are dropped per BS.775
    return np.stack([left, right], axis=1)


def apply_gain(pcm: np.ndarray, gain_db: float) -> np.ndarray:
    return pcm * (10.0 ** (gain_db / 20.0))


def limit(
    pcm: np.ndarray,
    fs: int,
    ceiling_dbtp: float,
    lookahead_ms: float = 5.0,
    release_ms: float = 100.0,
) -> tuple[np.ndarray, float]:
    """Transparent linked-stereo peak limiter.

    Returns (limited_pcm, max_reduction_db). The gain computer runs at a
    1 ms control rate with a lookahead envelope and one-pole release.
    """
    ceiling = 10.0 ** (ceiling_dbtp / 20.0)
    n = pcm.shape[0]
    if n == 0:
        return pcm, 0.0
    hop = max(1, int(round(0.001 * fs)))
    look = max(hop, int(round(lookahead_ms / 1000.0 * fs)))
    absmax = np.max(np.abs(pcm), axis=1)
    padded = np.concatenate([absmax, np.zeros(look)])
    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(padded, look)[::hop]
    env = windows.max(axis=1)
    raw_gain = np.minimum(1.0, ceiling / np.maximum(env, 1e-12))

    release_coef = np.exp(-1.0 / (release_ms / 1000.0 * fs / hop))
    gain_env = np.empty_like(raw_gain)
    prev = 1.0
    for i, g in enumerate(raw_gain):
        prev = g if g < prev else prev + (1.0 - release_coef) * (g - prev)
        gain_env[i] = prev

    per_sample = np.repeat(gain_env, hop)[:n]
    if per_sample.shape[0] < n:
        per_sample = np.pad(per_sample, (0, n - per_sample.shape[0]), mode="edge")
    out = pcm * per_sample[:, None]
    reduction = -20.0 * np.log10(max(float(gain_env.min()), 1e-12))
    return out, max(0.0, float(reduction))


def align(
    pcm: np.ndarray,
    fs: int,
    offset_ms: float,
    total_ms: float,
) -> np.ndarray:
    """Place pcm at offset_ms on a timeline of total_ms (pad/trim)."""
    total = int(round(total_ms * fs / 1000.0))
    start = int(round(offset_ms * fs / 1000.0))
    out = np.zeros((total, pcm.shape[1]))
    src0 = max(0, -start)
    dst0 = max(0, start)
    count = min(pcm.shape[0] - src0, total - dst0)
    if count > 0:
        out[dst0 : dst0 + count] = pcm[src0 : src0 + count]
    return out
