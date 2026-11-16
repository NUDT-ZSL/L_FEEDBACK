"""Pitch-preserving time stretching via WSOLA (pure numpy).

WSOLA (Waveform Similarity Overlap-Add, Verhelst & Roelands 1993) changes
duration without resampling, so the spectral content / pitch is preserved.
Only numpy is used; all windows are deterministic.
"""

from __future__ import annotations

import numpy as np

# Analysis window length (ms). 20 ms windows + 5 ms tolerance are standard
# speech-grade WSOLA parameters at 48 kHz.
WINDOW_MS = 20.0
TOLERANCE_MS = 5.0
MIN_WINDOW = 256
EPS = 1e-12


def _linear_resample(signal: np.ndarray, target_len: int) -> np.ndarray:
    """Plain linear interpolation fallback (very short clips)."""
    n = signal.shape[0]
    if target_len <= 0:
        return np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.full(target_len, float(signal[0]), dtype=np.float64)
    x = np.linspace(0.0, n - 1, num=target_len, dtype=np.float64)
    idx = np.floor(x).astype(np.int64)
    idx_next = np.minimum(idx + 1, n - 1)
    frac = (x - idx).astype(np.float64)
    return signal[idx] * (1.0 - frac) + signal[idx_next] * frac


def time_stretch(signal: np.ndarray, target_len: int, sample_rate: int = 48000) -> np.ndarray:
    """Return *signal* stretched/shortened to exactly *target_len* samples.

    - target_len == n  -> copy
    - Pitch is preserved (no resampling in the WSOLA core); a sub-sample
      linear correction is applied only at the very end to hit the exact
      frame count required by the cutscene timeline.
    """
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        x = x.reshape(-1)
    n = x.shape[0]
    if target_len < 0:
        raise ValueError("target_len must be non-negative")
    if target_len == 0:
        return np.zeros(0, dtype=np.float64)
    if n == 0:
        return np.zeros(target_len, dtype=np.float64)
    if target_len == n:
        return x.copy()

    win = max(MIN_WINDOW, int(round(WINDOW_MS / 1000.0 * sample_rate)))
    win = min(win, n)
    if win < 32 or n < win * 2:
        # Too little material for WSOLA: linear fallback.
        return _linear_resample(x, target_len)

    half = win // 2
    window = np.hanning(win)
    tolerance = max(8, int(round(TOLERANCE_MS / 1000.0 * sample_rate)))
    # WSOLA (Verhelst & Roelands): the analysis anchor advances by
    # synthesis_hop / alpha every output frame, while each concrete segment
    # may deviate only within +/- tolerance for waveform similarity. Keeping
    # the anchor locked to the continuous trajectory is what preserves pitch.
    alpha = target_len / float(n)
    analysis_hop = half / alpha

    out_len = target_len + win
    output = np.zeros(out_len, dtype=np.float64)
    weight = np.zeros(out_len, dtype=np.float64)

    # Continuous (float) input anchor; output positions stay on the sample grid.
    anchor = 0.0
    frame_index = 0
    prev_chosen = 0
    while True:
        syn_pos = frame_index * half
        if syn_pos + win > out_len:
            break
        if frame_index == 0:
            chosen_start = 0
        else:
            nominal = int(round(anchor))
            # Reference segment sits half a window ahead of the nominal anchor,
            # i.e. half a window after the *previously chosen* segment; the
            # search finds the best-matching continuation near that point.
            ref_start = prev_chosen + half
            if ref_start < 0:
                ref_start = 0
            ref_end = min(n, ref_start + win)
            if ref_end - ref_start == win:
                reference = x[ref_start:ref_end]
                lo = max(0, nominal - tolerance)
                hi = min(n - win, nominal + tolerance)
                if hi >= lo:
                    candidates = np.lib.stride_tricks.sliding_window_view(x, win)[lo : hi + 1]
                    ref_norm = reference - np.mean(reference)
                    cand_norm = candidates - np.mean(candidates, axis=1, keepdims=True)
                    num = cand_norm @ ref_norm
                    den = np.sqrt(
                        np.sum(cand_norm * cand_norm, axis=1) * np.sum(ref_norm * ref_norm)
                    )
                    corr = num / np.maximum(den, EPS)
                    chosen_start = lo + int(np.argmax(corr))
                else:
                    chosen_start = min(max(0, nominal), n - win)
            else:
                chosen_start = nominal
            if chosen_start < 0 or chosen_start + win > n:
                break

        prev_chosen = chosen_start
        end = min(n, chosen_start + win)
        length = end - chosen_start
        seg = x[chosen_start:end] * window[:length]
        o_end = min(out_len, syn_pos + length)
        output[syn_pos:o_end] += seg[: o_end - syn_pos]
        weight[syn_pos:o_end] += window[: o_end - syn_pos]

        # Advance the continuous anchor by the fixed analysis hop regardless
        # of the local similarity offset (pitch-lock invariant).
        anchor += analysis_hop
        frame_index += 1

    nonzero = weight > EPS
    output[nonzero] /= weight[nonzero]
    last = int(np.max(np.nonzero(nonzero)[0])) + 1 if np.any(nonzero) else 0
    if last <= 1:
        return _linear_resample(x, target_len)
    raw = output[:last]

    # Sub-sample correction so the returned clip occupies exactly target_len
    # frames (the WSOLA core itself introduces no pitch shift).
    if raw.shape[0] != target_len:
        raw = _linear_resample(raw, target_len)

    # Crossfade the first few samples against the first input sample to avoid
    # a boundary click when this clip is spliced into the voice track.
    fade = min(64, target_len)
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float64)
        raw[:fade] = raw[:fade] * ramp + float(x[0]) * (1.0 - ramp)
    return raw


def fit_duration(
    signal: np.ndarray,
    target_len: int,
    stretch_min: float,
    stretch_max: float,
) -> tuple[np.ndarray, float, str | None]:
    """Fit *signal* into exactly *target_len* frames under ratio limits.

    Returns (fitted_samples, applied_ratio, event_kind):
      event_kind None  -> pure WSOLA within limits
      event_kind "trim" -> ratio below stretch_min, tail trimmed to fit
      event_kind "pad"  -> ratio above stretch_max, silence padded to fit
    """
    n = signal.shape[0]
    if target_len <= 0:
        return np.zeros(0, dtype=np.float64), 1.0, "pad"
    ratio = target_len / float(n) if n > 0 else float("inf")

    if n == 0 or ratio > stretch_max:
        fitted = time_stretch(signal, min(target_len, max(1, int(round(n * stretch_max)))))
        out = np.zeros(target_len, dtype=np.float64)
        out[: fitted.shape[0]] = fitted
        return out, min(ratio, stretch_max) if n else stretch_max, "pad"
    if ratio < stretch_min:
        # The dialect is too long even at the slowest permitted rate. Keep
        # the pitch unchanged by dropping the tail: time-stretch only the
        # prefix that, at stretch_min, fills the slot exactly.
        keep = max(1, int(round(target_len / stretch_min)))
        fitted = time_stretch(signal[:keep], target_len)
        return fitted, stretch_min, "trim"
    return time_stretch(signal, target_len), ratio, None
