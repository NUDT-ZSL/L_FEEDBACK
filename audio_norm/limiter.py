"""Look-ahead true-peak limiter keeping output at or below -1.0 dBTP.

The limiter uses 4x oversampled peak detection with a short look-ahead so
that attenuation stays local to the offending transient.  A bounded makeup
stage then restores integrated loudness toward -23 LUFS without breaching the
true-peak ceiling, followed by a deterministic safety guard.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .loudness import (
    measure_integrated_loudness,
    oversample_4x,
    true_peak_db,
)

LIMIT_DBTP = -1.0
TRIGGER_THRESHOLD_DB = 0.1
MAX_GUARD_ITERATIONS = 8
MAX_MAKEUP_ITERATIONS = 8
LOOKAHEAD_MS = 0.5
INTERNAL_HEADROOM_DB = 0.3


def _peak_limiter_pass(signal: np.ndarray, fs: int, ceiling_dbtp: float,
                       lookahead_ms: float):
    """Single tight look-ahead limiting pass.

    Peak detection runs on the 4x oversampled signal.  The required per-sample
    attenuation is mapped back to the native rate (each native sample takes
    the maximum reduction across its four oversampled positions) so the gain
    is applied before the anti-imaging filter and cannot be smeared out of
    alignment with the transient it is protecting.
    """
    ceiling_linear = 10.0 ** (ceiling_dbtp / 20.0)
    over_4x = oversample_4x(signal)
    peak_per_sample = np.max(np.abs(over_4x), axis=1)
    reduction = np.where(
        peak_per_sample > ceiling_linear,
        20.0 * np.log10(
            np.maximum(peak_per_sample, 1e-12) / ceiling_linear
        ),
        0.0,
    )
    trigger_count = int(np.count_nonzero(reduction > TRIGGER_THRESHOLD_DB))
    max_reduction = float(reduction.max())
    lookahead = max(1, int(round(lookahead_ms / 1000.0 * fs * 4)))
    # A sliding maximum over the next `lookahead` samples.  Left-padding keeps
    # the envelope aligned with the sample it attenuates (the window spans the
    # current sample and the upcoming peaks rather than trailing samples).
    padded = np.pad(reduction, (lookahead, 0), constant_values=0.0)
    window = sliding_window_view(padded, lookahead + 1)
    envelope = window.max(axis=1)[-reduction.size:]
    native_count = signal.shape[0]
    grouped = envelope[: native_count * 4].reshape(native_count, 4).max(axis=1)
    gain_native = 10.0 ** (-grouped / 20.0)
    limited = signal * gain_native[:, None]
    return limited, trigger_count, max_reduction


def apply_limiter(signal: np.ndarray, fs: int, ceiling_dbtp: float = LIMIT_DBTP):
    """Limit true peak to `ceiling_dbtp` while preserving target loudness."""
    internal_ceiling = ceiling_dbtp - INTERNAL_HEADROOM_DB
    limited, trigger_count, peak_reduction = _peak_limiter_pass(
        signal, fs, internal_ceiling, LOOKAHEAD_MS
    )

    # Makeup stage: nudge loudness back up, re-limiting each time, but never
    # let the requested gain exceed what the internal-ceiling headroom allows.
    makeup_db = 0.0
    makeup_passes = 0

    for _ in range(MAX_MAKEUP_ITERATIONS):
        current_loudness = measure_integrated_loudness(limited, fs)
        current_tp = true_peak_db(limited)
        loudness_gap = -23.0 - current_loudness
        headroom_db = internal_ceiling - current_tp - 0.02
        step_db = min(loudness_gap, headroom_db)
        if step_db <= 0.01 or loudness_gap <= 0.0:
            break
        limited *= 10.0 ** (step_db / 20.0)
        limited, extra_triggers, extra_reduction = _peak_limiter_pass(
            limited, fs, internal_ceiling, LOOKAHEAD_MS
        )
        trigger_count = max(trigger_count, extra_triggers)
        peak_reduction = max(peak_reduction, extra_reduction)
        makeup_db += step_db
        makeup_passes += 1

    # Deterministic safety guard against residual resampling ripple.  This is
    # the only stage allowed to approach the declared -1.0 dBTP ceiling.
    guard_passes = 0
    guard_reduction = 0.0
    for _ in range(MAX_GUARD_ITERATIONS):
        current_tp = true_peak_db(limited)
        if current_tp <= ceiling_dbtp:
            break
        trim = 10.0 ** ((ceiling_dbtp - current_tp - 0.05) / 20.0)
        limited *= trim
        guard_reduction += float(-20.0 * np.log10(trim))
        guard_passes += 1

    max_reduction = peak_reduction + max(0.0, guard_reduction)
    stats = {
        "limiter_applied": True,
        "trigger_count": trigger_count,
        "max_gain_reduction_db": round(max_reduction, 3),
        "makeup_gain_db": round(makeup_db, 3),
        "makeup_passes": makeup_passes,
        "guard_passes": guard_passes,
        "ceiling_db_tp": ceiling_dbtp,
    }
    return limited, stats
