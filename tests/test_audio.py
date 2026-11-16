from __future__ import annotations

import math
import unittest

import numpy as np

from dialogue_localizer.audio import (
    equal_power_fades,
    trim_leading_trailing_silence,
)
from dialogue_localizer.loudness import (
    integrated_loudness,
    normalize_segment,
)


class AudioPrimitiveTests(unittest.TestCase):
    def test_equal_power_fade_preserves_energy(self) -> None:
        for length in (1, 2, 17, 480):
            fade_out, fade_in = equal_power_fades(length)
            np.testing.assert_allclose(
                fade_out * fade_out + fade_in * fade_in,
                1.0,
                atol=1e-12,
            )

    def test_silence_trim_removes_padding_only(self) -> None:
        sample_rate = 48_000
        active = np.sin(
            2.0 * math.pi * 440.0 * np.arange(4_800) / sample_rate
        )
        padded = np.concatenate(
            (np.zeros(4_800), active, np.zeros(4_800))
        )
        stereo = np.column_stack((padded, padded))
        trimmed = trim_leading_trailing_silence(stereo, sample_rate)
        self.assertGreaterEqual(trimmed.shape[0], active.shape[0])
        self.assertLess(trimmed.shape[0], stereo.shape[0] // 2)
        self.assertGreater(float(np.max(np.abs(trimmed))), 0.9)

    def test_normalization_reaches_loudness_and_peak_targets(self) -> None:
        sample_rate = 48_000
        time = np.arange(sample_rate * 3) / sample_rate
        signal = 0.002 * np.sin(2.0 * math.pi * 220.0 * time)
        signal[100:: sample_rate // 3] += 0.8
        stereo = np.column_stack((signal, signal))
        normalized, returned_loudness = normalize_segment(
            stereo, target_lufs=-23.0, peak_dbfs=-1.0
        )
        measured = integrated_loudness(normalized)
        peak_dbfs = 20.0 * math.log10(float(np.max(np.abs(normalized))))
        self.assertAlmostEqual(returned_loudness, measured, places=8)
        self.assertLessEqual(abs(measured + 23.0), 1.0)
        self.assertLessEqual(peak_dbfs, -1.0)

    def test_all_silence_is_not_accepted_for_normalization(self) -> None:
        silence = np.zeros((48_000, 2), dtype=np.float64)
        with self.assertRaises(ValueError):
            normalize_segment(silence)


if __name__ == "__main__":
    unittest.main()

