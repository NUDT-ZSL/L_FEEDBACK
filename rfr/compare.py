"""Pixel consistency comparison between the GPU and software framebuffers."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def compare_framebuffers(reference: np.ndarray, candidate: np.ndarray,
                         tolerance: int = 2) -> Dict[str, Any]:
    """Compare two RGBA8 images channel-wise with an absolute tolerance.

    Returns a report with the mismatch count, per-channel max difference and
    the bounding box (inclusive min/max row/col) of mismatching pixels.
    """
    if reference.shape != candidate.shape:
        raise ValueError(
            f"Framebuffer shape mismatch: {reference.shape} vs {candidate.shape}")
    a = reference.astype(np.int16)
    b = candidate.astype(np.int16)
    diff = np.abs(a - b)
    mismatched_mask = np.any(diff > tolerance, axis=2)
    mismatched_pixels = int(mismatched_mask.sum())
    report: Dict[str, Any] = {
        "tolerance": int(tolerance),
        "mismatched_pixels": mismatched_pixels,
        "total_pixels": int(reference.shape[0] * reference.shape[1]),
        "max_channel_diff": [int(diff[:, :, c].max()) for c in range(4)],
        "within_tolerance": mismatched_pixels == 0,
    }
    if mismatched_pixels:
        rows, cols = np.nonzero(mismatched_mask)
        report["coordinate_range"] = {
            "min_row": int(rows.min()), "max_row": int(rows.max()),
            "min_col": int(cols.min()), "max_col": int(cols.max()),
        }
    else:
        report["coordinate_range"] = None
    return report
