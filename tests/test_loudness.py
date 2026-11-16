import math

import numpy as np

from cinemixer.loudness import integrated_loudness, linear_gain_for_target


def _sine(seconds: float, rate: int, frequency: float, peak: float, channels: int = 2):
    length = int(seconds * rate)
    timeline = np.arange(length) / rate
    tone = peak * np.sin(2 * np.pi * frequency * timeline)
    if channels == 1:
        return tone[:, np.newaxis]
    return np.column_stack([tone, tone])


def test_silence_returns_negative_infinity():
    silence = np.zeros((48000, 2))
    assert integrated_loudness(silence, 48000) == float("-inf")


def test_known_sine_loudness_close_to_analytic_value():
    # -20 dBFS stereo sine. At 1 kHz the K-weighting chain measures about
    # -0.25 dB relative to input (shelf +2.8 dB, HP -3.1 dB), giving an
    # integrated loudness near -20.2 LUFS.
    audio = _sine(2.0, 48000, 1000.0, 10 ** (-20 / 20))
    measured = integrated_loudness(audio, 48000)
    assert -21.0 < measured < -19.5


def test_monotonic_with_level():
    quiet = _sine(1.5, 48000, 500.0, 0.1)
    loud = _sine(1.5, 48000, 500.0, 0.4)
    assert integrated_loudness(loud, 48000) > integrated_loudness(quiet, 48000)


def test_mono_and_stereo_equivalent_for_duplicated_channel():
    mono = _sine(1.5, 48000, 500.0, 0.2, channels=1)
    stereo = np.repeat(mono, 2, axis=1)
    # Duplicated channels add +3 LU by design (two equally weighted channels).
    delta = integrated_loudness(stereo, 48000) - integrated_loudness(mono, 48000)
    assert abs(delta - 3.0) < 0.2


def test_gain_correction_hits_target():
    audio = _sine(2.0, 48000, 1000.0, 0.2)
    measured = integrated_loudness(audio, 48000)
    target = -24.0
    gain = linear_gain_for_target(measured, target)
    corrected = audio * gain
    new_measured = integrated_loudness(corrected, 48000)
    assert abs(new_measured - target) < 0.1
    assert math.isfinite(gain) and gain > 0
