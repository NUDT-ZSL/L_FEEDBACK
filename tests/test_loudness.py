from __future__ import annotations

import numpy as np

from rhythmforge.loudness import integrated_lufs, loudness_gain


def _tone(frequency=440.0, seconds=2.0, rate=48000, amplitude=0.2):
    t = np.arange(int(seconds * rate)) / rate
    mono = amplitude * np.sin(2 * np.pi * frequency * t)
    return np.stack([mono, mono], axis=1).astype(np.float32)


def test_loudness_measurement_finite_and_deterministic():
    audio = _tone()
    first = integrated_lufs(audio, 48000)
    second = integrated_lufs(audio, 48000)
    assert np.isfinite(first)
    assert first == second


def test_gain_reaches_target_without_peak_breach():
    audio = _tone(amplitude=0.05)
    result = loudness_gain(audio, 48000, target_lufs=-23.0,
                           true_peak_dbfs=-1.0)
    output = audio.astype(np.float64) * result.applied_gain_linear
    assert np.max(np.abs(output)) <= result.peak_limit_linear + 1e-9
    assert abs(result.output_lufs - (-23.0)) < 0.2
    # Peak protection was unnecessary for this low-crest tone.
    assert abs(result.peak_gain_db) < 1e-6


def test_peak_protection_engages_for_aggressive_target():
    # Asking to push an already-loud tone toward 0 LUFS must trip the peak
    # limiter instead of allowing samples past the -1 dBFS ceiling.
    audio = _tone(amplitude=0.9)
    result = loudness_gain(audio, 48000, target_lufs=0.0,
                           true_peak_dbfs=-1.0)
    output = audio.astype(np.float64) * result.applied_gain_linear
    assert result.peak_gain_linear < 1.0
    assert np.max(np.abs(output)) <= result.peak_limit_linear + 1e-9


def test_gain_is_reusable_and_linear():
    audio = _tone(amplitude=0.15)
    result = loudness_gain(audio, 48000, -23.0, -1.0)
    scaled_a = audio.astype(np.float64) * result.applied_gain_linear
    scaled_b = (audio * 2).astype(np.float64) * result.applied_gain_linear
    assert np.allclose(scaled_b, scaled_a * 2, atol=1e-6)
