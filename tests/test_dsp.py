from __future__ import annotations

import math
import unittest

import numpy as np

from audio_event_mixer.dsp import (
    apply_loop_cleanup,
    crossfade_sample_count,
    pan_mono,
    panning_gains,
    resample_windowed_sinc,
)


class DspTest(unittest.TestCase):
    def test_equal_power_panning_and_depth_zero(self) -> None:
        left, right = panning_gains(2000, 48000, 1.7, 0.9)
        np.testing.assert_allclose(left**2 + right**2, 1.0, atol=1e-14)
        self.assertGreater(float(np.max(np.abs(left - right))), 0.1)

        center_left, center_right = panning_gains(
            2000, 48000, 3.0, 0.0, start_frame=1234
        )
        np.testing.assert_array_equal(center_left, center_right)
        np.testing.assert_allclose(center_left, math.sqrt(0.5))

    def test_pan_mono_preserves_frame_count_and_energy(self) -> None:
        mono = np.sin(2.0 * math.pi * 440.0 * np.arange(4800) / 48000.0)
        stereo = pan_mono(mono, 48000, 0.75, 0.8)
        self.assertEqual(stereo.shape, (4800, 2))
        np.testing.assert_allclose(
            np.sum(stereo**2, axis=1), mono**2, atol=1e-12
        )

    def test_crossfade_sample_count_floor_and_cap(self) -> None:
        self.assertEqual(crossfade_sample_count(600, 48000), 60)
        self.assertEqual(crossfade_sample_count(48000, 48000), 240)
        self.assertEqual(crossfade_sample_count(10, 48000), 1)
        self.assertEqual(crossfade_sample_count(3, 48000), 0)

    def test_loop_cleanup_preserves_length_and_removes_boundary_jump(self) -> None:
        rng = np.random.default_rng(1234)
        source = rng.normal(0.0, 0.2, 1000)
        source[100:700] += np.sin(np.linspace(0.0, 12.0, 600))
        before_jump = abs(source[699] - source[100])
        self.assertGreater(before_jump, 1e-8)
        result = apply_loop_cleanup(source, 48000, 100, 700)
        self.assertEqual(result.samples.shape, source.shape)
        self.assertEqual(result.crossfade_samples, 60)
        self.assertAlmostEqual(
            float(abs(result.samples[699] - result.samples[100])), 0.0, places=12
        )
        np.testing.assert_array_equal(result.samples[:640], source[:640])

    def test_resampling_duration_edge_dc_and_frequency(self) -> None:
        source_rate = 44100
        target_rate = 48000
        frames = source_rate
        t = np.arange(frames) / float(source_rate)
        tone = 0.5 * np.sin(2.0 * math.pi * 440.0 * t)
        resampled = resample_windowed_sinc(tone, source_rate, target_rate)
        self.assertEqual(resampled.shape, (48000,))
        duration_error = abs(resampled.size / target_rate - frames / source_rate)
        self.assertLessEqual(duration_error, 0.001)

        constant = resample_windowed_sinc(
            np.ones(1000), 32000, 48000
        )
        self.assertLess(float(np.max(np.abs(constant - 1.0))), 1e-10)

        spectrum = np.abs(np.fft.rfft(resampled[4000:-4000]))
        frequencies = np.fft.rfftfreq(resampled[4000:-4000].size, 1.0 / target_rate)
        peak_frequency = float(frequencies[int(np.argmax(spectrum))])
        self.assertLess(abs(peak_frequency - 440.0), 2.0)
        self.assertGreater(
            float(np.sqrt(np.mean(resampled**2))),
            0.30,
        )

    def test_downsampling_duration_and_dc(self) -> None:
        frames = 9600
        t = np.arange(frames) / 96000.0
        source = 0.4 * np.sin(2.0 * math.pi * 300.0 * t)
        resampled = resample_windowed_sinc(source, 96000, 48000)
        self.assertEqual(resampled.shape, (4800,))
        self.assertLess(abs(float(np.mean(resampled))), 1e-5)


if __name__ == "__main__":
    unittest.main()
