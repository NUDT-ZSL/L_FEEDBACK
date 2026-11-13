"""Deterministic linear-phase rational resampler.

General sample-rate conversion factor is out_rate / in_rate.  The underlying
filter is a zero-phase (centered) windowed-sinc low-pass prototype evaluated on
the fine (L * in_rate) grid, then organized into polyphase branches.  No RNG and
no platform-specific math beyond the deterministic CPython ``math`` module is
used, so identical inputs always produce identical integer sample outputs.
"""

import math

from .errors import AudioPipeError, EXIT_VALIDATION

TAPS_PER_PHASE = 16  # even number so the center sits between two taps


class ResampleError(AudioPipeError):
    code = "E_RESAMPLE"
    exit_code = EXIT_VALIDATION


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _design_phases(in_rate, out_rate):
    """Return (L, M, phase_coeffs, prototype_delay_in_fine_samples)."""
    g = _gcd(in_rate, out_rate)
    up = out_rate // g
    down = in_rate // g
    cutoff = min(in_rate, out_rate) / 2.0
    fine_rate = in_rate * up  # = out_rate * down
    # Normalized prototype cutoff relative to the fine-grid Nyquist.
    omega_c = 2.0 * math.pi * cutoff / fine_rate
    half = TAPS_PER_PHASE // 2
    # Prototype tap offsets span -half..half-1 around the sinc center on the
    # fine grid; branch 0 is the zero-phase branch.
    phases = []
    for phase in range(up):
        coeffs = []
        for k in range(-half, half):
            # Fine-grid sample index relative to the sinc center.
            n = k * up - phase
            if n == 0:
                h = omega_c / math.pi
            else:
                x = omega_c * n
                h = math.sin(x) / (math.pi * n)
            # Hann window on the fine-grid prototype, symmetric about n = 0.
            window_arg = (n + half * up) / (TAPS_PER_PHASE * up - 1)
            window = 0.5 - 0.5 * math.cos(2.0 * math.pi * window_arg)
            coeffs.append(h * window)
        phases.append(coeffs)
    # Normalize every polyphase branch to unity DC gain individually. This
    # removes window-truncation gain ripple across fractional phases, so a
    # constant input is reproduced at machine precision regardless of phase.
    normalized = []
    for branch in phases:
        branch_sum = sum(branch)
        if branch_sum <= 0:
            raise ResampleError("invalid resampler normalization")
        normalized.append([c / branch_sum for c in branch])
    phases = normalized
    return up, down, phases


def expected_output_len(in_len, in_rate, out_rate):
    """Theoretical output length from exact duration mapping (round-half-up)."""
    if in_len <= 0:
        return 0
    duration_seconds = in_len / in_rate
    exact = duration_seconds * out_rate
    return int(math.floor(exact + 0.5))


def resample(samples, in_rate, out_rate):
    """Resample integer PCM samples, returning (out_samples_float, meta).

    Output is float in PCM16 amplitude units; callers decide final quantization
    so gain/clipping decisions happen in one place.
    """
    if in_rate == out_rate:
        return [float(s) for s in samples], {
            "source_rate": in_rate,
            "target_rate": out_rate,
            "source_samples": len(samples),
            "output_samples": len(samples),
            "length_delta": 0,
        }
    if in_rate <= 0 or out_rate <= 0:
        raise ResampleError("illegal sample rate")

    up, down, phases = _design_phases(in_rate, out_rate)
    in_len = len(samples)
    out_len = expected_output_len(in_len, in_rate, out_rate)

    half = TAPS_PER_PHASE // 2
    # Input indices needed run slightly negative/beyond; treat outside as 0.
    out = [0.0] * out_len
    src = samples
    for m in range(out_len):
        # Mapping: fine index for output m is m*down; base input and phase.
        fine = m * down
        base = fine // up
        phase = fine - base * up
        coeffs = phases[phase]
        acc = 0.0
        start = base - half + 1
        for j, c in enumerate(coeffs):
            idx = start + j
            if 0 <= idx < in_len:
                acc += c * src[idx]
        out[m] = acc

    theoretical = out_len
    meta = {
        "source_rate": in_rate,
        "target_rate": out_rate,
        "source_samples": in_len,
        "output_samples": len(out),
        "length_delta": len(out) - theoretical,
    }
    return out, meta
