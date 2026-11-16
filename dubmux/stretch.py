"""Deterministic local time-stretch of dub segments onto target intervals."""

from __future__ import annotations

import numpy as np


def stretch_to_length(samples: np.ndarray, target_len: int) -> np.ndarray:
    """Resample to exactly target_len samples via linear interpolation.

    Deterministic: pure NumPy, no randomness, no external state.
    """
    if target_len <= 0:
        return np.zeros(0, dtype=np.float64)
    if len(samples) == 0:
        return np.zeros(target_len, dtype=np.float64)
    if len(samples) == 1:
        return np.full(target_len, samples[0], dtype=np.float64)
    positions = np.linspace(0.0, len(samples) - 1, num=target_len)
    return np.interp(positions, np.arange(len(samples), dtype=np.float64), samples)


def assemble_track(
    segments: list[np.ndarray],
    targets: list[tuple[float, float]],
    total_duration: float,
    rate: int,
) -> tuple[np.ndarray, list[dict]]:
    """Stretch each segment onto its target interval on a silent timeline.

    Returns (track, per-segment alignment records). The track length is
    exactly round(total_duration * rate) so the muxed audio matches the
    video duration metadata.
    """
    total_len = round(total_duration * rate)
    track = np.zeros(total_len, dtype=np.float64)
    records = []
    for index, (seg, (t_start, t_end)) in enumerate(zip(segments, targets)):
        target_dur = t_end - t_start
        target_len = round(target_dur * rate)
        original_len = len(seg)
        stretched = stretch_to_length(seg, target_len)
        offset = round(t_start * rate)
        end = min(offset + target_len, total_len)
        if offset < total_len and end > offset:
            track[offset:end] = stretched[: end - offset]
        original_dur = original_len / rate
        final_dur = (end - offset) / rate
        records.append(
            {
                "index": index,
                "original_duration": round(original_dur, 6),
                "ratio": round(target_dur / original_dur, 6) if original_dur > 0 else None,
                "target_start": round(t_start, 6),
                "target_end": round(t_end, 6),
                "final_duration": round(final_dur, 6),
                "boundary_error": round(abs(final_dur - target_dur), 6),
            }
        )
    return track, records
