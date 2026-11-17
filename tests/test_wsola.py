from __future__ import annotations

import unittest

import numpy as np

from dialogue_mixer.wsola import WSOLAParams, wsola_stretch


SAMPLE_RATE = 48000


def dominant_frequency(samples: np.ndarray) -> float:
    trimmed = samples[4800:-4800]
    windowed = trimmed * np.hanning(trimmed.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(trimmed.size, d=1.0 / SAMPLE_RATE)
    return float(frequencies[int(np.argmax(spectrum))])


class WSOLATests(unittest.TestCase):
    def setUp(self) -> None:
        time = np.arange(SAMPLE_RATE, dtype=np.float64) / SAMPLE_RATE
        self.tone = 0.5 * np.sin(2.0 * np.pi * 440.0 * time)
        self.params = WSOLAParams(
            frame_size=2048,
            synthesis_hop=512,
            search_radius=480,
        )

    def test_stretch_has_exact_length_and_preserves_pitch(self) -> None:
        stretched = wsola_stretch(
            self.tone,
            int(1.5 * SAMPLE_RATE),
            SAMPLE_RATE,
            params=self.params,
        )
        self.assertEqual(stretched.size, int(1.5 * SAMPLE_RATE))
        self.assertTrue(np.all(np.isfinite(stretched)))
        self.assertAlmostEqual(dominant_frequency(stretched), 440.0, delta=2.0)
        self.assertGreater(float(np.max(np.abs(stretched))), 0.4)

    def test_compression_has_exact_length_and_preserves_pitch(self) -> None:
        compressed = wsola_stretch(
            self.tone,
            int(0.72 * SAMPLE_RATE),
            SAMPLE_RATE,
            params=self.params,
        )
        self.assertEqual(compressed.size, int(0.72 * SAMPLE_RATE))
        self.assertAlmostEqual(dominant_frequency(compressed), 440.0, delta=3.0)

    def test_zero_target_returns_empty(self) -> None:
        self.assertEqual(wsola_stretch(self.tone, 0, SAMPLE_RATE).size, 0)


if __name__ == "__main__":
    unittest.main()

