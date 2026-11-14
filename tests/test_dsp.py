from __future__ import annotations

import math
import unittest

import numpy as np

from cinemix.dsp import (
    crossfade_gains,
    integrated_loudness,
    interval_envelope,
    resample_signal,
    true_peak,
)


class DspTests(unittest.TestCase):
    def test_bs1770_reference_sine(self) -> None:
        sample_rate = 48000
        time = np.arange(sample_rate * 2) / sample_rate
        stereo = np.stack(
            (np.sin(2.0 * math.pi * 1000.0 * time) * 0.1,) * 2,
            axis=1,
        )
        measured = integrated_loudness(stereo, sample_rate)
        self.assertAlmostEqual(measured.lufs, -20.0, delta=0.05)
        mono = integrated_loudness(stereo[:, 0], sample_rate)
        self.assertAlmostEqual(mono.lufs, -23.0, delta=0.05)

    def test_true_peak_detects_intersample_peak(self) -> None:
        sample_rate = 48000
        time = np.arange(sample_rate * 2) / sample_rate
        signal = np.sin(
            2.0 * math.pi * (sample_rate / 4.0) * time + math.pi / 4.0
        )
        sample_peak = 20.0 * math.log10(float(np.max(np.abs(signal))))
        measured = true_peak(signal[:, None], sample_rate)
        self.assertLess(sample_peak, -3.0)
        self.assertAlmostEqual(measured.dbtp, 0.0, delta=0.02)

    def test_resampler_rates_and_timeline(self) -> None:
        pairs = ((44100, 48000), (48000, 96000), (96000, 44100))
        for input_rate, output_rate in pairs:
            with self.subTest(input_rate=input_rate, output_rate=output_rate):
                frames = input_rate * 2
                time = np.arange(frames) / input_rate
                source = np.sin(2.0 * math.pi * 997.0 * time + 0.3)
                converted = resample_signal(source, input_rate, output_rate)
                self.assertEqual(
                    converted.size,
                    round(frames * output_rate / input_rate),
                )
                output_time = np.arange(converted.size) / output_rate
                expected = np.sin(2.0 * math.pi * 997.0 * output_time + 0.3)
                edge = int(0.05 * output_rate)
                error = converted[edge:-edge] - expected[edge:-edge]
                rms_error = math.sqrt(float(np.mean(error * error)))
                self.assertLess(rms_error, 1e-4)

                impulse = np.zeros(input_rate)
                impulse[input_rate // 2] = 1.0
                converted_impulse = resample_signal(
                    impulse, input_rate, output_rate
                )
                peak_index = int(np.argmax(converted_impulse))
                expected_index = (input_rate // 2) * output_rate / input_rate
                self.assertLessEqual(
                    abs(peak_index - expected_index), 1.0
                )

    def test_crossfade_curve_endpoints_and_energy(self) -> None:
        progress = np.linspace(0.0, 1.0, 101)
        for curve in ("linear", "equal_power", "s_curve"):
            outgoing, incoming = crossfade_gains(progress, curve)
            self.assertAlmostEqual(float(outgoing[0]), 1.0, places=12)
            self.assertAlmostEqual(float(incoming[0]), 0.0, places=12)
            self.assertAlmostEqual(float(outgoing[-1]), 0.0, places=12)
            self.assertAlmostEqual(float(incoming[-1]), 1.0, places=12)
            np.testing.assert_allclose(
                np.asarray(outgoing) ** 2 + np.asarray(incoming) ** 2,
                1.0,
                atol=1e-12,
            )

    def test_fade_boundary_is_sample_exact(self) -> None:
        sample_rate = 48000
        frames = 48000
        start = 0.1234
        end = 0.4567
        track_start = 0.05
        envelope = interval_envelope(
            frames,
            sample_rate,
            track_start,
            start,
            end,
            "s_curve",
            "in",
        )
        track_offset = round(track_start * sample_rate)
        start_index = round(start * sample_rate) - track_offset
        end_index = round(end * sample_rate) - track_offset
        self.assertEqual(float(envelope[start_index - 1]), 0.0)
        self.assertEqual(float(envelope[start_index]), 0.0)
        self.assertAlmostEqual(float(envelope[end_index - 1]), 1.0, places=6)
        self.assertEqual(float(envelope[end_index]), 1.0)
        worst_error_ms = 0.5 / sample_rate * 1000.0
        self.assertLess(worst_error_ms, 1.0)


if __name__ == "__main__":
    unittest.main()

