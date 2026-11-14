import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audiomixer.errors import MixerError
from audiomixer.wavio import quantize_pcm, read_wav, write_wav_atomic


class WavRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="audiomixer_wav_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_all_supported_widths_round_trip(self):
        source = (np.indices((480, 2)).sum(axis=0) % 200 - 100) / 256.0
        for width in (2, 3, 4):
            target = self.tmp / f"tone_{width * 8}.wav"
            write_wav_atomic(target, source, 48000, width)
            samples, sample_rate, read_width = read_wav(target)
            self.assertEqual(sample_rate, 48000)
            self.assertEqual(read_width, width)
            self.assertEqual(samples.shape, (480, 2))
            max_error = 1.0 / (1 << (8 * width - 1))
            np.testing.assert_allclose(samples, source, atol=max_error + 1e-12)

    def test_quantisation_clips_and_centers(self):
        over = np.array([[1.5], [-1.5], [0.0]])
        quantised = quantize_pcm(over, 2)
        self.assertEqual(quantised[0, 0], 32767)
        self.assertEqual(quantised[1, 0], -32768)
        self.assertEqual(quantised[2, 0], 0)

    def test_missing_file_raises_business_error(self):
        with self.assertRaises(MixerError) as context:
            read_wav(self.tmp / "nope.wav")
        self.assertEqual(context.exception.code, "E_MISSING_ASSET")

    def test_corrupt_file_raises_unsupported(self):
        target = self.tmp / "junk.wav"
        target.write_bytes(b"RIFFxxxxWAVEnotreally")
        with self.assertRaises(MixerError) as context:
            read_wav(target)
        self.assertEqual(context.exception.code, "E_UNSUPPORTED_FORMAT")

    def test_atomic_write_replaces_without_temp_leftovers(self):
        target = self.tmp / "atomic.wav"
        signal = np.zeros((100, 2))
        write_wav_atomic(target, signal, 48000, 4)
        first = target.read_bytes()
        write_wav_atomic(target, signal + 0.1, 48000, 4)
        self.assertNotEqual(first, target.read_bytes())
        leftovers = [p for p in self.tmp.iterdir() if p.name.startswith(".atomic.wav.")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
