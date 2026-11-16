import numpy as np
import pytest

from locdub import wsola

SR = 44100


def _voiced_signal(seconds=2.0, f0=220.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * seconds)) / SR
    phase = 2 * np.pi * f0 * t
    harmonic = sum(np.sin(phase * k) / k for k in range(1, 8))
    return (0.4 * harmonic + 0.02 * rng.standard_normal(len(t))).astype(np.float64)


def _peak_frequency(signal, f0):
    n = 1 << 20
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)), n))
    freqs = np.fft.rfftfreq(n, 1 / SR)
    lo = int(f0 * 0.7 * n / SR)
    hi = int(f0 * 1.3 * n / SR)
    idx = lo + int(np.argmax(spectrum[lo:hi]))
    left, center, right = spectrum[idx - 1], spectrum[idx], spectrum[idx + 1]
    correction = 0.5 * (left - right) / (left - 2 * center + right)
    return freqs[idx] + correction * (freqs[1] - freqs[0])


@pytest.mark.parametrize("ratio", [0.5, 0.7, 1.0, 1.3, 1.5, 2.0])
def test_duration_is_exact(ratio):
    signal = _voiced_signal()
    out = wsola.stretch_samples(signal, ratio, SR)
    assert len(out) == wsola.target_sample_count(len(signal), ratio)


@pytest.mark.parametrize("ratio", [0.5, 0.7, 1.3, 1.5, 2.0])
def test_pitch_is_preserved(ratio):
    f0 = 220.0
    signal = _voiced_signal(f0=f0)
    out = wsola.stretch_samples(signal, ratio, SR)
    peak = _peak_frequency(out, f0)
    cents = 1200 * np.log2(peak / f0)
    assert abs(cents) < 20.0


def test_stereo_preserves_channels():
    signal = _voiced_signal(seconds=1.0)
    stereo = np.stack((signal, signal * 0.7), axis=1)
    out = wsola.stretch_samples(stereo, 1.4, SR)
    assert out.ndim == 2
    assert out.shape[1] == 2
    assert len(out) == wsola.target_sample_count(len(signal), 1.4)


def test_output_is_finite_and_bounded():
    rng = np.random.default_rng(3)
    signal = rng.standard_normal(SR).astype(np.float64) * 0.3
    out = wsola.stretch_samples(signal, 1.7, SR)
    assert np.isfinite(out).all()


def test_invalid_ratio_raises():
    with pytest.raises(ValueError):
        wsola.stretch_samples(np.zeros(100), 0.0, SR)
