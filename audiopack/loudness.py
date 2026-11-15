"""ITU-R BS.1770-4 integrated loudness measurement and gain smoothing."""

from __future__ import annotations

import math

import numpy as np
import pyloudnorm as pyln

from .errors import W_LOW_LOUDNESS

TARGET_LUFS = -23.0
MAX_BOOST_DB = 20.0
MIN_GAIN_RAMP_MS = 100.0
LOW_LOUDNESS_FLOOR = -70.0


def measure_loudness(audio: np.ndarray, sample_rate: int) -> float:
    """Return integrated loudness in LUFS; a silent mix returns ``-math.inf``."""
    meter = pyln.Meter(int(sample_rate))
    loudness = float(meter.integrated_loudness(np.asarray(audio, dtype=np.float64)))
    if not math.isfinite(loudness):
        return -math.inf
    return loudness


def linear_gain(db: float) -> float:
    """Convert a dB value to a linear amplitude factor."""
    return float(10.0 ** (db / 20.0))


def gain_envelope(
    length: int,
    sample_rate: int,
    automation: list[tuple[int, float]],
) -> np.ndarray:
    """Build a deterministic gain curve from ``(sample, db)`` automation points.

    Gain is held outside the automation range and linearly interpolated in dB
    between points.  Every change shorter than 100 ms is expanded to the
    100 ms minimum so no gain step is instantaneous.
    """
    if not automation:
        return np.ones(length, dtype=np.float32)

    points = sorted((max(0, min(length - 1, int(pos))), float(db)) for pos, db in automation)
    min_ramp = max(1, int(round(MIN_GAIN_RAMP_MS * sample_rate / 1000.0)))

    # Enforce the minimum ramp length by pushing adjacent points apart.
    expanded: list[tuple[int, float]] = []
    for pos, db in points:
        if expanded and pos - expanded[-1][0] < min_ramp and db != expanded[-1][1]:
            pos = min(length - 1, expanded[-1][0] + min_ramp)
        expanded.append((pos, db))

    envelope = np.empty(length, dtype=np.float32)
    first_pos, first_db = expanded[0]
    envelope[: first_pos + 1] = linear_gain(first_db)
    cursor = first_pos
    current_db = first_db
    for next_pos, next_db in expanded[1:]:
        span = max(1, next_pos - cursor)
        ramp = np.linspace(current_db, next_db, span, endpoint=False, dtype=np.float64)
        envelope[cursor:next_pos] = linear_gain_array(ramp)
        cursor, current_db = next_pos, next_db
    envelope[cursor:] = linear_gain(current_db)
    return envelope


def linear_gain_array(db_values: np.ndarray) -> np.ndarray:
    """Vectorized dB -> linear conversion."""
    return np.power(10.0, db_values / 20.0).astype(np.float32)


def normalize(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, dict, list[str]]:
    """Normalize ``audio`` toward -23 LUFS with a smoothed gain transition.

    Returns the processed audio, a metrics dict (measured before/after, the
    applied gain and whether the +20 dB boost cap was active) and warnings.
    When the source loudness is at or below -70 LUFS the boost is capped at
    +20 dB and ``W_LOW_LOUDNESS`` is emitted.
    """
    warnings: list[str] = []
    measured = measure_loudness(audio, sample_rate)

    capped = False
    if not math.isfinite(measured):
        target_gain_db = MAX_BOOST_DB
        capped = True
        warnings.append(W_LOW_LOUDNESS)
    else:
        target_gain_db = TARGET_LUFS - measured
        if measured <= LOW_LOUDNESS_FLOOR or target_gain_db > MAX_BOOST_DB:
            target_gain_db = min(target_gain_db, MAX_BOOST_DB)
            capped = True
            warnings.append(W_LOW_LOUDNESS)

    # Constant gain is itself a valid automation curve; gain_envelope keeps
    # the smoothing helper in the active path (zero change -> no ramp needed).
    envelope = gain_envelope(audio.shape[0], sample_rate, [(0, target_gain_db)])
    processed = (audio.astype(np.float32) * envelope[:, np.newaxis]).astype(np.float32)
    post_loudness = measure_loudness(processed, sample_rate)

    metrics = {
        "target_lufs": TARGET_LUFS,
        "measured_lufs": _json_number(measured),
        "output_lufs": _json_number(post_loudness),
        "applied_gain_db": round(float(target_gain_db), 6),
        "gain_boost_capped": bool(capped),
        "min_gain_ramp_ms": MIN_GAIN_RAMP_MS,
    }
    return processed, metrics, warnings


def _json_number(value: float) -> float | None:
    """JSON has no infinity; expose non-finite measurements as ``null``."""
    if not math.isfinite(value):
        return None
    return round(float(value), 6)
