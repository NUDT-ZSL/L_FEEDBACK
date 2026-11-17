"""Deterministic DSP primitives: resampling, channel conversion, fades.

All operations are pure IEEE-754 float computations with a fixed order of
operations, so results are bit-exact reproducible across runs and machines
with the same NumPy major version.
"""

import math

import numpy as np

from .errors import BusinessError, ErrorCode

CURVES = ("linear", "equal_power")


def ms_to_frames(ms, rate):
    """Convert milliseconds to a frame count (round-half-even, deterministic)."""
    return int(round(ms * rate / 1000.0))


def resample_length(in_frames, in_rate, out_rate):
    """Output frame count for resampling, integer round-half-up."""
    return (in_frames * out_rate + in_rate // 2) // in_rate


def resample_linear(data, in_rate, out_rate):
    """Deterministic linear-interpolation resampler.

    Output frame j maps to input position j * in_rate / out_rate (float64).
    No library-default or random behaviour is involved.
    """
    if in_rate == out_rate:
        return data.astype(np.float32, copy=True)
    n = data.shape[0]
    out_n = resample_length(n, in_rate, out_rate)
    if n == 0 or out_n == 0:
        return np.zeros((out_n, data.shape[1]), dtype=np.float32)
    pos = np.arange(out_n, dtype=np.float64) * (float(in_rate) / float(out_rate))
    pos = np.minimum(pos, n - 1)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    frac = (pos - i0).astype(np.float32)
    out = data[i0] * (1.0 - frac)[:, None] + data[i1] * frac[:, None]
    return out.astype(np.float32)


def convert_channels(data, channels):
    """Convert mono<->stereo. Mono->stereo duplicates; stereo->mono averages."""
    current = data.shape[1]
    if current == channels:
        return data
    if current == 1 and channels == 2:
        return np.repeat(data, 2, axis=1)
    if current == 2 and channels == 1:
        mono = (data[:, 0] + data[:, 1]) * 0.5
        return mono.astype(np.float32).reshape(-1, 1)
    raise BusinessError(
        ErrorCode.E_UNSUPPORTED_FORMAT,
        f"cannot convert {current} channels to {channels}",
    )


def fade_gains(curve, n):
    """Fade-in gain envelope of length n as float64.

    Boundary samples are explicitly defined: gains[0] == 0.0 and
    gains[-1] == 1.0 (t = i / (n - 1)). For n == 1 the gain is 1.0.
    Fade-out envelopes are the same array reversed.
    """
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    if curve == "linear":
        return t
    if curve == "equal_power":
        return np.sin(t * (math.pi / 2.0))
    raise BusinessError(ErrorCode.E_MANIFEST_INVALID, f"unknown fade curve: {curve}")


def apply_fades(data, fade_in_frames, fade_in_curve, fade_out_frames, fade_out_curve):
    """Apply fade-in at the head and fade-out at the tail, in float32."""
    out = data.astype(np.float32, copy=True)
    if fade_in_frames > 0:
        g = fade_gains(fade_in_curve, fade_in_frames).astype(np.float32)
        out[:fade_in_frames] *= g[:, None]
    if fade_out_frames > 0:
        g = fade_gains(fade_out_curve, fade_out_frames)[::-1].astype(np.float32)
        out[out.shape[0] - fade_out_frames:] *= g[:, None]
    return out
