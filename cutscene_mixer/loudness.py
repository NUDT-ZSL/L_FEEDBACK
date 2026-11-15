"""Simplified ITU-R BS.1770 K-weighted loudness measurement.

Simplifications: the two standard 48 kHz K-weighting biquad stages are
applied at every sample rate, and integrated loudness is computed over the
whole signal without block gating.
"""

import numpy as np

# BS.1770 stage 1 (high shelf) and stage 2 (high pass), 48 kHz coefficients.
_STAGE1_B = (1.53512485958697, -2.69169618940638, 1.19839281085285)
_STAGE1_A = (1.0, -1.69065929318241, 0.73248077421585)
_STAGE2_B = (1.0, -2.0, 1.0)
_STAGE2_A = (1.0, -1.99004745483398, 0.99007225036621)

_ABSOLUTE_SILENCE_LUFS = -120.0


def _biquad(signal, b, a):
    """Direct-form-II biquad filter along axis 0 (no scipy dependency)."""
    b0, b1, b2 = b
    _, a1, a2 = a
    out = np.empty_like(signal)
    for channel in range(signal.shape[1]):
        x = signal[:, channel]
        y = np.empty_like(x)
        z1 = 0.0
        z2 = 0.0
        for i in range(x.shape[0]):
            acc = b0 * x[i] + z1
            z1 = b1 * x[i] - a1 * acc + z2
            z2 = b2 * x[i] - a2 * acc
            y[i] = acc
        out[:, channel] = y
    return out


def k_weighted(signal):
    """Apply the two K-weighting filter stages to (samples, channels)."""
    filtered = _biquad(signal, _STAGE1_B, _STAGE1_A)
    return _biquad(filtered, _STAGE2_B, _STAGE2_A)


def integrated_loudness(signal):
    """Integrated loudness in LUFS for (samples, channels) float audio."""
    if signal.size == 0 or signal.shape[0] == 0:
        return _ABSOLUTE_SILENCE_LUFS
    filtered = k_weighted(signal)
    mean_square = np.mean(np.square(filtered), axis=0)
    total = float(np.sum(mean_square))
    if total <= 0.0:
        return _ABSOLUTE_SILENCE_LUFS
    return -0.691 + 10.0 * np.log10(total)
