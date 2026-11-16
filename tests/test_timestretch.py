import numpy as np
import pytest

from dub_replace.timestretch import fit_duration, time_stretch


def dominant_freq(signal: np.ndarray, sr: int) -> float:
    windowed = signal * np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    k = int(np.argmax(spectrum[1:]) + 1)
    # Parabolic interpolation for sub-bin accuracy.
    alpha = spectrum[k - 1]
    beta = spectrum[k]
    gamma = spectrum[k + 1] if k + 1 < spectrum.size else beta
    denom = alpha - 2 * beta + gamma
    shift = 0.5 * (alpha - gamma) / denom if denom != 0 else 0.0
    return (k + shift) * sr / signal.size


def test_time_stretch_exact_length():
    sr = 48000
    x = np.random.default_rng(0).normal(0, 0.3, sr // 2)
    for factor in (0.6, 0.8, 1.25, 1.8):
        target = int(x.size * factor)
        y = time_stretch(x, target)
        assert y.shape == (target,)
        assert np.all(np.isfinite(y))


def test_time_stretch_preserves_pitch():
    sr = 48000
    n = 2 * sr
    t = np.arange(n) / sr
    x = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    for factor in (0.6, 1.5):
        y = time_stretch(x, int(n * factor))
        f_in = dominant_freq(x, sr)
        f_out = dominant_freq(y, sr)
        assert abs(f_out - f_in) < 6.0  # Hz; WSOLA keeps the carrier pitch


def test_identity_and_edge_cases():
    x = np.zeros(1000)
    assert np.array_equal(time_stretch(x, 1000), x)
    assert time_stretch(x, 500).shape == (500,)
    assert time_stretch(np.zeros(0), 10).shape == (10,)
    with pytest.raises(ValueError):
        time_stretch(x, -1)


def test_fit_duration_clamps_and_marks_events():
    sr = 48000
    slot = int(0.4 * sr)
    # 2 s dialect into a 0.4 s slot -> ratio 0.2 < min 0.5 -> trim event.
    long_clip = np.random.default_rng(1).normal(0, 0.3, 2 * sr)
    fitted, ratio, event = fit_duration(long_clip, slot, 0.5, 2.0)
    assert fitted.shape == (slot,)
    assert event == "trim"
    # Ratio is clamped at the declared minimum and the tail is discarded.
    assert ratio == pytest.approx(0.5)
    assert np.any(fitted != 0.0)

    # 0.1 s dialect into the same slot -> ratio 4.0 > max 2.0 -> pad event.
    short_clip = np.random.default_rng(2).normal(0, 0.3, int(0.1 * sr))
    fitted, ratio, event = fit_duration(short_clip, slot, 0.5, 2.0)
    assert fitted.shape == (slot,)
    assert event == "pad"
    assert ratio == pytest.approx(2.0)
    # Tail must be silence padding.
    voiced = int(0.1 * sr * 2.0)
    assert np.all(fitted[voiced + 10:] == 0.0)

    # Within limits -> no event, exact fit.
    mid_clip = np.random.default_rng(3).normal(0, 0.3, int(0.3 * sr))
    fitted, ratio, event = fit_duration(mid_clip, slot, 0.5, 2.0)
    assert event is None
    assert fitted.shape == (slot,)
    assert ratio == pytest.approx(slot / mid_clip.size)
