from __future__ import annotations

import unittest

import numpy as np

from lipalign.stretch import StretchSettings, time_stretch


class StretchTests(unittest.TestCase):
    def test_wsola_and_phase_vocoder_preserve_pitch_and_exact_length(self) -> None:
        sample_rate = 48000
        source_count = sample_rate
        time = np.arange(source_count, dtype=np.float64) / float(sample_rate)
        source = 0.5 * np.sin(2.0 * np.pi * 440.0 * time)
        for algorithm in ("wsola", "phase_vocoder"):
            for target_count in (24000, 72000):
                with self.subTest(algorithm=algorithm, target_count=target_count):
                    output = time_stretch(
                        source,
                        target_count,
                        sample_rate,
                        StretchSettings(algorithm=algorithm),
                    )
                    self.assertEqual(len(output), target_count)
                    region = output[2000:-2000]
                    frequencies = np.fft.rfftfreq(len(region), 1.0 / sample_rate)
                    peak = float(
                        frequencies[int(np.argmax(np.abs(np.fft.rfft(region))))]
                    )
                    self.assertAlmostEqual(peak, 440.0, delta=2.0)
                    self.assertGreater(float(np.sqrt(np.mean(region * region))), 0.1)

    def test_time_stretch_handles_tiny_fragments(self) -> None:
        source = np.linspace(-0.2, 0.2, 100, dtype=np.float64)
        for algorithm in ("wsola", "phase_vocoder"):
            output = time_stretch(
                source, 173, 48000, StretchSettings(algorithm=algorithm)
            )
            self.assertEqual(len(output), 173)
            self.assertTrue(np.all(np.isfinite(output)))


if __name__ == "__main__":
    unittest.main()

