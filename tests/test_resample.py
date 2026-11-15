"""Sinc resampler: length rule, timing alignment and signal fidelity."""

from __future__ import annotations

import numpy as np

from audiopack.resample import resample


def test_output_length_uses_rounding_rule():
    sr_in = 44100
    sr_out = 48000
    signal = np.zeros((1000, 1), dtype=np.float32)
    out = resample(signal, sr_in, sr_out)
    expected = int(np.floor(round(1000 * sr_out / sr_in)))
    assert out.shape == (expected, 1)
    assert out.dtype == np.float32


def test_tone_pitch_and_timing_within_one_sample():
    sr_in, sr_out = 44100, 48000
    frames_in = 4410  # 100 ms
    t = np.arange(frames_in) / sr_in
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)[:, np.newaxis]

    out = resample(tone, sr_in, sr_out)
    out_mono = out[:, 0]

    # Timing: output impulse alignment must stay within +/- one output sample.
    impulse = np.zeros(2205, dtype=np.float32)
    impulse[1000] = 1.0
    resampled_impulse = resample(impulse[:, np.newaxis], sr_in, sr_out)[:, 0]
    peak = int(np.argmax(np.abs(resampled_impulse)))
    expected_peak = int(np.floor(round(1000 * sr_out / sr_in)))
    assert abs(peak - expected_peak) <= 1

    # Fidelity: resampled tone stays phase-aligned with the analytic target.
    t_out = np.arange(out.shape[0]) / sr_out
    target = 0.5 * np.sin(2 * np.pi * 1000 * t_out)
    error = np.sqrt(np.mean((out_mono - target) ** 2))
    assert error < 0.02


def test_upsample_then_downsample_roundtrip():
    sr = 32000
    rng = np.random.default_rng(7)
    original = rng.standard_normal((8000, 2)).astype(np.float32) * 0.3
    up = resample(original, sr, 48000)
    back = resample(up, 48000, sr)
    # Compare the shared interior (edges use clamped indices).
    interior = slice(50, -50)
    corr = np.corrcoef(back[interior, 0], original[interior, 0])[0, 1]
    assert corr > 0.99


def test_identical_rate_is_copy():
    signal = np.ones((128, 2), dtype=np.float32)
    out = resample(signal, 48000, 48000)
    assert np.array_equal(out, signal)
    out[:] = 0
    assert signal[0, 0] == 1.0
