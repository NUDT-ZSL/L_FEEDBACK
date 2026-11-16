import unittest

import numpy as np

from lipsync_tool import wavio


class WavIoTests(unittest.TestCase):
    def test_round_trip_mono_and_stereo_16bit(self):
        import tempfile
        from pathlib import Path

        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as directory:
            mono_path = Path(directory) / "mono.wav"
            stereo_path = Path(directory) / "stereo.wav"
            mono = rng.uniform(-0.8, 0.8, size=8000)
            stereo = np.column_stack([mono, mono * 0.5])
            wavio.write_wav(mono_path, mono, 16000, 2)
            wavio.write_wav(stereo_path, stereo, 16000, 2)

            mono_wav = wavio.read_wav(mono_path)
            stereo_wav = wavio.read_wav(stereo_path)
            self.assertEqual(mono_wav.channels, 1)
            self.assertEqual(stereo_wav.channels, 2)
            self.assertEqual(mono_wav.sample_width, 2)
            # 16-bit PCM quantization step is ~3.05e-5
            self.assertTrue(np.allclose(mono_wav.mono(), mono, atol=4e-5))
            self.assertTrue(np.allclose(stereo_wav.mono(), (mono + mono * 0.5) / 2, atol=4e-5))

    def test_resample_backends_preserve_duration(self):
        rng = np.random.default_rng(1)
        signal = rng.normal(0, 0.2, size=12000)
        for backend in (wavio.resample, wavio.resample_linear):
            up = backend(signal, 24000, 48000)
            down = backend(signal, 48000, 16000)
            self.assertAlmostEqual(up.shape[0] / 48000, signal.shape[0] / 24000, delta=0.01)
            self.assertAlmostEqual(down.shape[0] / 16000, signal.shape[0] / 48000, delta=0.01)

    def test_normalize_to_mono_resamples_and_downmixes(self):
        rng = np.random.default_rng(2)
        stereo = rng.normal(0, 0.2, size=(24000, 2))
        wav = wavio.WavData(stereo, 24000, 2)
        mono, rate, notes = wavio.normalize_to_mono(wav, 32000)
        self.assertEqual(rate, 32000)
        self.assertEqual(mono.ndim, 1)
        self.assertAlmostEqual(mono.shape[0] / 32000, 1.0, delta=0.01)
        self.assertTrue(any("downmixed" in note for note in notes))
        self.assertTrue(any("resampled" in note for note in notes))

    def test_pcm_widths_8_24_32_bit_round_trip(self):
        import tempfile
        from pathlib import Path

        rng = np.random.default_rng(8)
        samples = rng.uniform(-0.5, 0.5, size=4000)
        tolerances = {1: 0.02, 2: 4e-5, 3: 2e-7, 4: 1e-9}
        with tempfile.TemporaryDirectory() as directory:
            for width, tolerance in tolerances.items():
                path = Path(directory) / f"width_{width}.wav"
                wavio.write_wav(path, samples, 8000, width)
                loaded = wavio.read_wav(path)
                self.assertEqual(loaded.sample_width, width)
                self.assertTrue(np.allclose(loaded.mono(), samples, atol=tolerance))


if __name__ == "__main__":
    unittest.main()
