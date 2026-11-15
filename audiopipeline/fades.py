"""Cosine-window fades and precise loop stitching."""

from __future__ import annotations

import numpy as np

from .config import SAMPLE_RATE


def ms_to_samples(ms: float) -> int:
    return max(0, int(round(float(ms) * SAMPLE_RATE / 1000.0)))


def cosine_gain(length: int, fade_in: bool = True) -> np.ndarray:
    """Equal-gain cosine ramp (0 -> 1 or 1 -> 0).

    Half-sample phase positions are used so the endpoints avoid exact
    zero/unity boundary discontinuities; the curve remains a pure cosine
    window and is exactly complementary for crossfades.
    """
    if length <= 0:
        return np.empty(0, dtype=np.float64)
    phase = (np.arange(length, dtype=np.float64) + 0.5) / length
    if fade_in:
        return 0.5 - 0.5 * np.cos(np.pi * phase)
    return 0.5 + 0.5 * np.cos(np.pi * phase)


def apply_crossfade(tail: np.ndarray, head: np.ndarray) -> np.ndarray:
    """Equal-gain cosine crossfade between two equal-length stereo blocks."""
    n = tail.shape[0]
    fade_out = cosine_gain(n, fade_in=False)[:, None]
    fade_in = cosine_gain(n, fade_in=True)[:, None]
    return tail * fade_out + head * fade_in


def loop_source(source: np.ndarray, needed: int, crossfade_samples: int) -> np.ndarray:
    """Build at least ``needed`` samples by looping ``source``.

    Each loop boundary uses a short equal-gain cosine crossfade to avoid
    clicks.  The crossfade length is clamped to half the source length.
    """
    total = source.shape[0]
    if total == 0:
        return source.copy()
    xf = min(crossfade_samples, total // 2)
    if xf <= 0 or total >= needed:
        if total >= needed:
            return source[:needed].copy()
        reps = int(np.ceil(needed / total))
        return np.concatenate([source] * reps, axis=0)[:needed]

    built = source.copy()
    while built.shape[0] < needed:
        join = built.shape[0] - xf
        base = built
        merged = base[:join].copy()
        blended = apply_crossfade(base[join:], source[:xf])
        merged = np.concatenate([merged, blended, source[xf:]], axis=0)
        built = merged
    return built[:needed]


def apply_edge_fades(
    segment: np.ndarray, head_samples: int, tail_samples: int
) -> np.ndarray:
    """Apply optional leading fade-in and a tail fade-out (in place copy)."""
    out = segment
    n = out.shape[0]
    head_samples = min(head_samples, n)
    tail_samples = min(tail_samples, n)
    if head_samples > 0:
        out = out.copy()
        out[:head_samples] *= cosine_gain(head_samples, fade_in=True)[:, None]
    if tail_samples > 0:
        out = out.copy()
        out[n - tail_samples :] *= cosine_gain(tail_samples, fade_in=False)[:, None]
    return out
