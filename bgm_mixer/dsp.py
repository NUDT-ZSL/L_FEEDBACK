"""Band-limited resampling, fade envelopes, and drift-free loop scheduling."""

from __future__ import annotations

import math
from math import gcd
from typing import Optional, Tuple

import numpy as np
from scipy.signal import resample_poly

from .errors import E_INVALID_INPUT, E_RESAMPLE, MixerError

# Kaiser beta 7 gives more than 70 dB image/alias rejection across the
# 44.1/48 kHz conversions used for the 20 kHz edge test, while retaining useful
# energy near 20 kHz.  (Larger betas widen the transition band in scipy's
# fixed-length resample_poly filter and can reduce that edge rejection.)
_KAISER_BETA = 7.0


def round_ratio(numerator: int, denominator: int) -> int:
    """Round a positive rational number to nearest, with ties rounded up."""

    if numerator < 0 or denominator <= 0:
        raise ValueError("round_ratio requires a non-negative numerator and positive denominator")
    return (2 * numerator + denominator) // (2 * denominator)


def align_loop_indices(
    start_seconds: Optional[float],
    end_seconds: Optional[float],
    source_rate: int,
    source_frames: int,
) -> Tuple[int, int]:
    """Convert loop seconds to nearest source-frame indices, end-exclusive."""

    start = 0.0 if start_seconds is None else start_seconds
    end = source_frames / float(source_rate) if end_seconds is None else end_seconds
    if not math.isfinite(start) or not math.isfinite(end):
        raise MixerError(E_INVALID_INPUT, "loop times must be finite")
    if start < 0.0:
        raise MixerError(E_INVALID_INPUT, "loop start must not be negative")
    if end <= start:
        raise MixerError(E_INVALID_INPUT, "loop end must be greater than loop start")

    start_sample = int(math.floor(start * source_rate + 0.5))
    end_sample = int(math.floor(end * source_rate + 0.5))
    if start_sample < 0 or end_sample > source_frames:
        raise MixerError(
            E_INVALID_INPUT,
            "loop range lies outside the source WAV",
        )
    if end_sample <= start_sample:
        raise MixerError(E_INVALID_INPUT, "loop range contains no source samples")
    return start_sample, end_sample


def resample_segment(
    segment: np.ndarray,
    source_rate: int,
    target_rate: int,
    expected_frames: int,
) -> np.ndarray:
    """Resample a loop segment and force the nearest-integer theoretical length."""

    if segment.ndim != 2:
        raise MixerError(E_RESAMPLE, "resampler input must be a 2-D sample array")
    if segment.shape[0] < 2:
        raise MixerError(E_RESAMPLE, "loop segment must contain at least two samples")
    if expected_frames < 2:
        raise MixerError(E_RESAMPLE, "resampled loop is shorter than two samples")
    if not np.all(np.isfinite(segment)):
        raise MixerError(E_RESAMPLE, "source samples contain NaN or infinity")

    if source_rate == target_rate:
        output = np.array(segment, dtype=np.float64, copy=True)
    else:
        common = gcd(int(source_rate), int(target_rate))
        up = int(target_rate) // common
        down = int(source_rate) // common
        try:
            output = resample_poly(
                segment.astype(np.float64, copy=False),
                up,
                down,
                axis=0,
                window=("kaiser", _KAISER_BETA),
                padtype="constant",
            )
        except (MemoryError, ValueError, RuntimeError, OverflowError) as exc:
            raise MixerError(E_RESAMPLE, f"resampling failed: {exc}") from exc

    if not np.all(np.isfinite(output)):
        raise MixerError(E_RESAMPLE, "resampler produced NaN or infinity")

    # scipy returns ceil(n * up / down); loop lengths are defined by nearest
    # integer frames so that duration error never exceeds half an output frame.
    if output.shape[0] > expected_frames:
        output = output[:expected_frames]
    elif output.shape[0] < expected_frames:
        padding = np.zeros(
            (expected_frames - output.shape[0], output.shape[1]),
            dtype=np.float64,
        )
        output = np.concatenate((output, padding), axis=0)
    return np.ascontiguousarray(output, dtype=np.float64)


def fade_envelope(
    frames: int,
    fade_in_frames: int,
    fade_out_frames: int,
    curve: str,
) -> np.ndarray:
    """Build a repeatable per-loop gain envelope with exact-zero endpoints."""

    if frames < 1:
        raise MixerError(E_RESAMPLE, "cannot build an envelope for an empty loop")
    if fade_in_frames < 0 or fade_out_frames < 0:
        raise MixerError(E_INVALID_INPUT, "fade lengths must not be negative")
    if fade_in_frames + fade_out_frames > frames:
        raise MixerError(
            E_INVALID_INPUT,
            "fade-in plus fade-out is longer than the resampled loop",
        )
    if curve not in ("equal_power", "linear"):
        raise MixerError(E_INVALID_INPUT, "fade curve must be equal_power or linear")

    envelope = np.ones(frames, dtype=np.float64)

    if fade_in_frames:
        index = np.arange(fade_in_frames, dtype=np.float64)
        if fade_in_frames == 1:
            gain = np.zeros(1, dtype=np.float64)
        else:
            unit = index / float(fade_in_frames - 1)
            if curve == "linear":
                gain = unit
            else:
                gain = np.sin(0.5 * math.pi * unit)
        gain[0] = 0.0
        if fade_in_frames > 1:
            gain[-1] = 1.0
        envelope[:fade_in_frames] *= gain

    if fade_out_frames:
        index = np.arange(fade_out_frames, dtype=np.float64)
        if fade_out_frames == 1:
            gain = np.zeros(1, dtype=np.float64)
        else:
            unit = index / float(fade_out_frames - 1)
            if curve == "linear":
                gain = 1.0 - unit
            else:
                gain = np.cos(0.5 * math.pi * unit)
        if fade_out_frames > 1:
            gain[0] = 1.0
        gain[-1] = 0.0
        envelope[-fade_out_frames:] *= gain

    return envelope


def loop_period_ratio(source_frames: int, source_rate: int, target_rate: int) -> Tuple[int, int]:
    """Return the ideal loop period as a reduced target-frame ratio."""

    if source_frames <= 0 or source_rate <= 0 or target_rate <= 0:
        raise MixerError(E_RESAMPLE, "loop period ratio requires positive frame counts and rates")
    numerator = int(source_frames) * int(target_rate)
    denominator = int(source_rate)
    common = gcd(numerator, denominator)
    return numerator // common, denominator // common


def scheduled_period_start(k: int, numerator: int, denominator: int) -> int:
    """Start frame of loop ``k`` for an ideal fractional period.

    Rounding every absolute start (instead of accumulating a rounded period)
    keeps the phase error below half a sample forever.
    """

    if k < 0:
        raise ValueError("loop index must not be negative")
    return round_ratio(int(k) * int(numerator), int(denominator))


def scheduled_loop_count(
    total_frames: int,
    offset_frames: int,
    numerator: int,
    denominator: int,
) -> int:
    """Count loop starts that fall within ``[0, total_frames)``."""

    available = int(total_frames) - int(offset_frames)
    if available <= 0:
        return 0
    max_index = available - 1
    # Largest k satisfying round(k*numerator/denominator) <= max_index.
    return (int(denominator) * (2 * max_index + 1) - 1) // (
        2 * int(numerator)
    ) + 1


def resampled_loop_frames(source_frames: int, source_rate: int, target_rate: int) -> int:
    """Nearest integer number of target frames in one aligned loop cycle."""

    if source_frames <= 0:
        raise MixerError(E_RESAMPLE, "source loop must not be empty")
    return round_ratio(int(source_frames) * int(target_rate), int(source_rate))


def milliseconds_to_frames(milliseconds: float, sample_rate: int) -> int:
    if milliseconds < 0.0 or not math.isfinite(milliseconds):
        raise MixerError(E_INVALID_INPUT, "fade duration must be finite and non-negative")
    return int(math.floor(milliseconds * sample_rate / 1000.0 + 0.5))
