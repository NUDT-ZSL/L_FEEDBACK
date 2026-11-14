import unittest

import numpy as np

from audiomixer.resampler import (
    _resample_numpy,
    engine_description,
    expected_output_length,
    resample_mono,
)


class ResamplerTests(unittest.TestCase):
    def test_length_relationship(self):
        self.assertEqual(expected_output_length(44100, 44100, 48000), 48000)
        self.assertEqual(expected_output_length(48000, 48000, 44100), 44100)
        self.assertEqual(expected_output_length(1000, 8000, 12000), 1500)

    def test_passthrough_when_rates_match(self):
        signal = np.linspace(-0.5, 0.5, 1000)
        out, info = resample_mono(signal, 48000, 48000)
        np.testing.assert_array_equal(signal, out)
        self.assertEqual(info["engine"], "passthrough")
        self.assertEqual(info["length_residual_samples"], 0)

    def test_upsample_preserves_tone_frequency_and_length(self):
        sr_in, sr_out = 44100, 48000
        frames = sr_in  # one second
        time_axis = np.arange(frames) / sr_in
        signal = 0.6 * np.sin(2 * np.pi * 1000.0 * time_axis)
        out, info = resample_mono(signal, sr_in, sr_out)
        self.assertEqual(out.shape[0], 48000)
        self.assertEqual(info["final_output_frames"], 48000)
        # A 1 kHz tone must remain ~1 kHz: measure spectral peak after settling.
        steady = out[sr_out // 4 :]
        spectrum = np.abs(np.fft.rfft(steady * np.hanning(steady.size)))
        freqs = np.fft.rfftfreq(steady.size, 1 / sr_out)
        peak_frequency = freqs[np.argmax(spectrum)]
        self.assertAlmostEqual(peak_frequency, 1000.0, delta=4.0)
        # Amplitude is preserved by a quality resampler within a small margin.
        self.assertLess(abs(np.max(np.abs(steady)) - 0.6), 0.03)

    def test_downsample_length_and_residual_reporting(self):
        signal = np.random.default_rng(42).standard_normal(48000)
        out, info = resample_mono(signal, 48000, 44100)
        self.assertEqual(out.shape[0], 44100)
        self.assertEqual(
            info["raw_output_frames"] - info["trimmed_samples"]
            + info["padded_samples"],
            44100,
        )
        self.assertIn(info["engine"], {"ffmpeg_soxr", "ffmpeg_swr", "numpy_windowed_sinc"})

    def test_numpy_fallback_is_self_consistent(self):
        sr_in, sr_out = 32000, 48000
        time_axis = np.arange(sr_in) / sr_in
        signal = 0.4 * np.sin(2 * np.pi * 440.0 * time_axis)
        out = _resample_numpy(signal, sr_in, sr_out)
        self.assertEqual(out.shape[0], expected_output_length(signal.size, sr_in, sr_out))
        steady = out[4800:]
        self.assertLess(abs(np.max(np.abs(steady)) - 0.4), 0.02)

    def test_engine_is_known(self):
        self.assertIn(
            engine_description(),
            {"ffmpeg_soxr", "ffmpeg_swr", "numpy_windowed_sinc"},
        )


if __name__ == "__main__":
    unittest.main()
