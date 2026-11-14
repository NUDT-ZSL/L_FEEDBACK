"""Unit tests for the BS.1770-4 measurement and true-peak machinery."""
from __future__ import annotations

import unittest

import numpy as np

from audio_norm.loudness import (
    find_silence_segments,
    measure_integrated_loudness,
    true_peak_db,
)
from audio_norm.limiter import apply_limiter


def _tone(fs: int, seconds: float, freq: float = 440.0,
          amplitude: float = 0.5, channels: int = 2) -> np.ndarray:
    t = np.arange(int(fs * seconds)) / fs
    mono = amplitude * np.sin(2 * np.pi * freq * t)
    return np.repeat(mono.reshape(-1, 1), channels, axis=1)


class LoudnessTests(unittest.TestCase):
    def test_gain_scaling_shifts_loudness_by_gain_db(self):
        signal = _tone(48000, 3.0, amplitude=0.4)
        baseline = measure_integrated_loudness(signal, 48000)
        boosted = signal * (10.0 ** (6.0 / 20.0))
        shifted = measure_integrated_loudness(boosted, 48000)
        self.assertAlmostEqual(shifted - baseline, 6.0, places=1)

    def test_supported_layouts_measure_finite(self):
        for channels in (1, 2, 6):
            signal = _tone(48000, 2.0, channels=channels)
            loudness = measure_integrated_loudness(signal, 48000)
            self.assertTrue(np.isfinite(loudness))

    def test_unsupported_layout_rejected(self):
        signal = _tone(48000, 2.0, channels=4)
        with self.assertRaises(ValueError):
            measure_integrated_loudness(signal, 48000)

    def test_silence_detection_threshold_and_duration(self):
        fs = 48000
        rng = np.random.default_rng(7)
        total = fs * 4
        signal = np.zeros((total, 1))
        speech = 0.4 * np.sin(2 * np.pi * 180 * np.arange(fs) / fs)
        signal[fs: 2 * fs, 0] = speech
        signal[3 * fs: 4 * fs, 0] = speech
        signal += (10.0 ** (-72 / 20.0)) * rng.standard_normal((total, 1))
        segments = find_silence_segments(signal, fs)
        self.assertTrue(segments)
        # Leading silence should be detected and start at zero.
        self.assertLessEqual(segments[0][0], 0.1)
        self.assertTrue(all(end - start >= 0.199 for start, end in segments))

    def test_true_peak_detects_intersample_peak(self):
        fs = 48000
        signal = 0.7 * _tone(fs, 1.0, freq=8000.0, channels=1)
        tp = true_peak_db(signal)
        self.assertTrue(np.isfinite(tp))


class LimiterTests(unittest.TestCase):
    def test_limiter_holds_ceiling(self):
        fs = 48000
        rng = np.random.default_rng(3)
        n = fs * 3
        t = np.arange(n) / fs
        signal = (0.05 * np.sin(2 * np.pi * 220 * t)
                  + 0.01 * rng.standard_normal(n)).reshape(-1, 1)
        for position in (0.5, 1.5, 2.5):
            idx = int(position * fs)
            signal[idx:idx + 16] = 0.999
        limited, stats = apply_limiter(signal, fs)
        self.assertTrue(stats["limiter_applied"])
        self.assertGreater(stats["trigger_count"], 0)
        self.assertLessEqual(true_peak_db(limited), -1.0)

    def test_limiter_preserves_loudness_when_physically_possible(self):
        fs = 48000
        rng = np.random.default_rng(11)
        n = fs * 4
        t = np.arange(n) / fs
        base = (0.12 * np.sin(2 * np.pi * 200 * t)
                + 0.02 * rng.standard_normal(n)).reshape(-1, 1)
        for position in (1.0, 2.0, 3.0):
            idx = int(position * fs)
            base[idx:idx + 8] = 0.99
        current = measure_integrated_loudness(base, fs)
        normalized = base * 10.0 ** ((-23.0 - current) / 20.0)
        limited, _ = apply_limiter(normalized, fs)
        self.assertLessEqual(true_peak_db(limited), -1.0)
        self.assertLessEqual(
            abs(measure_integrated_loudness(limited, fs) + 23.0), 0.5
        )


if __name__ == "__main__":
    unittest.main()
