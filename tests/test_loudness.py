import math
import unittest

import numpy as np

from audiomixer.dsp import equal_power_pan
from audiomixer.errors import MixerError
from audiomixer.loudness import (
    apply_k_weighting,
    measure_loudness,
    normalise_loudness,
    sample_peak_dbfs,
    true_peak_dbfs,
)


def tone(frequency=1000.0, amplitude=1.0, seconds=2.0, sr=48000, channels=1, pan=0.0):
    frames = int(seconds * sr)
    phase = 2.0 * np.pi * frequency * np.arange(frames) / sr
    mono = amplitude * np.sin(phase)
    if channels == 1:
        return mono.reshape(-1, 1)
    left, right = equal_power_pan(pan)
    return np.stack((mono * left, mono * right), axis=1)


class LoudnessMeasurementTests(unittest.TestCase):
    def test_full_scale_1khz_mono(self):
        # FFmpeg's ebur128 reports a 1 kHz, 0 dBFS sine at -3.0 LUFS; the
        # official BS.1770-4 48 kHz coefficients must reproduce that closely.
        result = measure_loudness(tone(amplitude=1.0), 48000)
        self.assertFalse(result["silence"])
        self.assertAlmostEqual(result["integrated_lufs"], -3.0, delta=0.1)

    def test_known_amplitude_step(self):
        # Halving amplitude is -6.0206 dB and must move LUFS by the same amount.
        loud = measure_loudness(tone(amplitude=0.8), 48000)["integrated_lufs"]
        quiet = measure_loudness(tone(amplitude=0.4), 48000)["integrated_lufs"]
        self.assertAlmostEqual(loud - quiet, 20.0 * math.log10(2.0), delta=0.05)

    def test_stereo_equals_two_weighted_copies(self):
        # Two identical full-weight copies (not constant-power pan) are +3 LUFS.
        mono_signal = tone(amplitude=0.7, channels=1)
        stereo = np.repeat(mono_signal, 2, axis=1)
        stereo_loudness = measure_loudness(stereo, 48000)["integrated_lufs"]
        mono_loudness = measure_loudness(mono_signal, 48000)["integrated_lufs"]
        # Two independent copies at half power: +3.01 LUFS.
        self.assertAlmostEqual(stereo_loudness - mono_loudness, 3.0103, delta=0.05)

    def test_silence_is_gated_out(self):
        silence = np.zeros((48000, 2))
        result = measure_loudness(silence, 48000)
        self.assertTrue(result["silence"])
        self.assertIsNone(result["integrated_lufs"])

    def test_short_signal_is_padded_to_one_block(self):
        short = tone(seconds=0.1, amplitude=0.5)
        result = measure_loudness(short, 48000)
        self.assertGreaterEqual(result["total_blocks"], 1)
        self.assertIsNotNone(result["integrated_lufs"])

    def test_k_weighting_dc_removed(self):
        dc = np.full((48000, 1), 0.5)
        filtered = apply_k_weighting(dc, 48000)
        # The RLB high-pass must suppress a constant component after settling.
        self.assertLess(np.max(np.abs(filtered[24000:])), 1e-6)


class PeakTests(unittest.TestCase):
    def test_sample_peak_dbfs(self):
        signal = tone(amplitude=0.5, seconds=0.5)
        self.assertAlmostEqual(sample_peak_dbfs(signal), 20 * math.log10(0.5), delta=0.01)

    def test_true_peak_at_least_sample_peak(self):
        # A high frequency near Nyquist can hide energy between samples.
        signal = tone(frequency=12000.0, amplitude=0.6, seconds=1.0)
        self.assertGreaterEqual(true_peak_dbfs(signal, 48000), sample_peak_dbfs(signal) - 0.05)


class NormalisationTests(unittest.TestCase):
    def test_hits_target_and_peak_ceiling(self):
        signal = tone(frequency=300.0, amplitude=0.9, seconds=3.0, channels=2)
        adjusted, stats = normalise_loudness(signal, 48000, -23.0, -1.0, 0.5)
        self.assertAlmostEqual(stats["loudness_after_lufs"], -23.0, delta=0.05)
        self.assertLessEqual(stats["peak_after_dbfs"], -1.0 + 1e-6)
        self.assertFalse(stats["peak_limited"])
        # The returned audio must match the reported gain.
        expected = signal * 10 ** (stats["applied_gain_db"] / 20)
        np.testing.assert_allclose(adjusted, expected, atol=1e-12)

    def test_peak_ceiling_triggers_error_when_shortfall_too_large(self):
        # A very hot tone targeting -16 LUFS with a -20 dBFS ceiling cannot be
        # satisfied within a tight tolerance.
        signal = tone(frequency=300.0, amplitude=1.0, seconds=2.0)
        with self.assertRaises(MixerError) as context:
            normalise_loudness(signal, 48000, -16.0, -20.0, 0.1)
        self.assertEqual(context.exception.code, "E_PEAK_LIMIT")
        self.assertIn("stats", context.exception.details)

    def test_peak_ceiling_accepted_within_tolerance(self):
        # A quiet tone targeting a loud level: a tight peak ceiling caps the
        # gain below the target gain, leaving a shortfall inside tolerance.
        signal = tone(frequency=300.0, amplitude=0.2, seconds=2.0)
        _, stats = normalise_loudness(signal, 48000, -10.0, -8.0, 2.0)
        self.assertTrue(stats["peak_limited"])
        self.assertLessEqual(stats["peak_after_dbfs"], -8.0 + 1e-6)
        self.assertGreater(stats["loudness_shortfall_lu"], 0.0)

    def test_silence_cannot_be_normalised(self):
        with self.assertRaises(MixerError) as context:
            normalise_loudness(np.zeros((48000, 1)), 48000, -23.0, -1.0, 0.5)
        self.assertEqual(context.exception.code, "E_PEAK_LIMIT")


if __name__ == "__main__":
    unittest.main()
