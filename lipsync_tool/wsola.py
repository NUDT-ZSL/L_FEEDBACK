"""WSOLA (Waveform Similarity Overlap-Add) time-scale modification.

Deterministic, NumPy-only implementation.  A stretch factor of 1.0 takes an
exact copy fast path; factors outside the configured bounds are clamped by the
caller (:mod:`lipsync_tool.pipeline`).

The reference frame is always anchored to the natural (unmodified) analysis
timeline, while the best-matching continuation frame is searched in a small
neighbourhood of the next natural position.  This avoids the phase drift /
pitch shift that occurs when references follow previously accepted positions.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def wsola_time_stretch(
    signal: np.ndarray,
    stretch_factor: float,
    sample_rate: int,
    frame_ms: float = 32.0,
    tolerance_ms: float = 8.0,
) -> np.ndarray:
    """Stretch ``signal`` in time without changing its sample rate or pitch."""
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    stretch_factor = float(stretch_factor)
    if signal.size == 0 or stretch_factor <= 0.0:
        return signal.copy()
    if abs(stretch_factor - 1.0) < 1e-9:
        return signal.copy()

    frame_length = int(round(sample_rate * frame_ms / 1000.0))
    frame_length = max(256, min(frame_length, 2048))
    if frame_length % 2:
        frame_length += 1
    # 50% overlap gives robust cross-fades; the analysis stride is the
    # synthesis hop scaled by the stretch factor so pitch is preserved.
    synthesis_hop = max(1, frame_length // 2)
    search_tolerance = min(frame_length // 2, max(1, int(round(sample_rate * tolerance_ms / 1000.0))))
    window = np.hanning(frame_length)

    padded = np.pad(signal, (frame_length, frame_length), mode="constant")
    target_frames = int(round(signal.size * stretch_factor))
    output = np.zeros(target_frames + 2 * frame_length)
    weight = np.zeros_like(output)
    max_valid_start = padded.size - frame_length

    analysis_stride = synthesis_hop * stretch_factor
    reference_position = frame_length
    natural_position = float(frame_length + analysis_stride)
    synthesis_position = 0

    while reference_position + frame_length <= padded.size and synthesis_position < target_frames:
        reference = padded[reference_position : reference_position + frame_length]
        natural_index = int(round(natural_position))
        search_start = max(frame_length, natural_index - search_tolerance)
        search_end = min(max_valid_start, natural_index + search_tolerance)
        if search_end <= search_start:
            candidate_position = min(max_valid_start, max(0, natural_index))
        else:
            # Vectorised normalized cross-correlation over candidate positions.
            candidates = np.lib.stride_tricks.sliding_window_view(
                padded[search_start : search_end + frame_length], frame_length
            )
            numerator = candidates @ reference
            candidate_energy = np.sqrt(np.sum(candidates * candidates, axis=1) + 1e-12)
            reference_energy = np.sqrt(np.sum(reference * reference) + 1e-12)
            scores = numerator / (candidate_energy * reference_energy)
            candidate_position = search_start + int(np.argmax(scores))

        candidate_position = int(np.clip(candidate_position, 0, max_valid_start))
        frame = padded[candidate_position : candidate_position + frame_length]
        span = min(frame_length, output.size - synthesis_position)
        end = synthesis_position + span
        output[synthesis_position:end] += frame[:span] * window[:span]
        weight[synthesis_position:end] += window[:span]

        reference_position = natural_index
        natural_position += analysis_stride
        synthesis_position += synthesis_hop

    nonzero = weight > 1e-8
    output[nonzero] /= weight[nonzero]
    output[~nonzero] = 0.0

    result = output[:target_frames]
    if result.size < target_frames:
        result = np.pad(result, (0, target_frames - result.size))
    # WSOLA overlap-add can slightly exceed unit amplitude; normalise softly.
    peak = float(np.max(np.abs(result))) if result.size else 0.0
    if peak > 1.0:
        result = result / peak
    return result


def clamp_stretch_factor(
    factor: float, lower: float, upper: float
) -> Tuple[float, str | None]:
    """Clamp a stretch factor to the allowed bounds, reporting a warning."""
    if factor < lower:
        return float(lower), (
            f"stretch factor {factor:.4f} below lower bound {lower:.4f}; clamped"
        )
    if factor > upper:
        return float(upper), (
            f"stretch factor {factor:.4f} above upper bound {upper:.4f}; clamped"
        )
    return float(factor), None
