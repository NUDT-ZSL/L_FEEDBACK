import unittest

import numpy as np

from lipsync_tool.wsola import clamp_stretch_factor, wsola_time_stretch


class WsolaTests(unittest.TestCase):
    def test_unity_factor_is_exact_copy(self):
        rng = np.random.default_rng(3)
        signal = rng.normal(0, 0.3, size=32000)
        output = wsola_time_stretch(signal, 1.0, 16000)
        np.testing.assert_array_equal(output, signal)

    def test_output_length_matches_stretch_factor(self):
        rng = np.random.default_rng(4)
        duration = 2.0
        rate = 16000
        time_axis = np.arange(int(rate * duration)) / rate
        signal = 0.4 * np.sin(2 * np.pi * 220 * time_axis)
        for factor in (0.75, 0.9, 1.1, 1.25):
            output = wsola_time_stretch(signal, factor, rate)
            expected = int(round(signal.size * factor))
            self.assertEqual(output.size, expected)
            self.assertEqual(output.size / rate, factor * duration)

    def test_pitch_energy_is_preserved(self):
        rate = 16000
        time_axis = np.arange(rate * 4) / rate
        signal = 0.5 * np.sin(2 * np.pi * 300 * time_axis)
        # A search tolerance below one pitch period avoids period-skipping on
        # this purely sinusoidal test signal.
        stretched = wsola_time_stretch(signal, 1.2, rate, tolerance_ms=4.0)
        # Zero-padded FFT gives a much finer frequency estimate.
        windowed = stretched * np.hanning(stretched.size)
        spectrum = np.abs(np.fft.rfft(windowed, n=stretched.size * 8))
        peak_bin = int(np.argmax(spectrum))
        peak_frequency = peak_bin * rate / (stretched.size * 8)
        self.assertAlmostEqual(peak_frequency, 300, delta=8)

    def test_clamp_stretch_factor(self):
        factor, warning = clamp_stretch_factor(1.4, 0.75, 1.25)
        self.assertEqual(factor, 1.25)
        self.assertIn("clamped", warning)
        factor, warning = clamp_stretch_factor(0.6, 0.75, 1.25)
        self.assertEqual(factor, 0.75)
        self.assertIn("clamped", warning)
        factor, warning = clamp_stretch_factor(1.0, 0.75, 1.25)
        self.assertEqual(factor, 1.0)
        self.assertIsNone(warning)


if __name__ == "__main__":
    unittest.main()
