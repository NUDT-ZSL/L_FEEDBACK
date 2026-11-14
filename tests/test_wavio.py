import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adaptive_mixer.wavio import (
    read_wav, read_wav_chunks, write_wav_24bit,
)


class WavIOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_24bit_mono(self):
        rng = np.random.default_rng(11)
        samples = rng.standard_normal(24000) * 0.25
        path = self.dir / "mix.wav"
        write_wav_24bit(path, samples, 48000)
        loaded, sr = read_wav(path)
        self.assertEqual(sr, 48000)
        self.assertEqual(len(loaded), len(samples))
        np.testing.assert_allclose(loaded, samples, atol=1 / 8388608 * 1.5)

    def test_riff_layout_contains_fact(self):
        path = self.dir / "mix.wav"
        write_wav_24bit(path, np.zeros(1024), 48000)
        info = read_wav_chunks(path)
        ids = [name for name, _size in info["chunks"]]
        self.assertEqual(ids, ["fmt ", "fact", "data"])
        fact_size = dict(info["chunks"])["fact"]
        self.assertEqual(fact_size, 4)
        with path.open("rb") as handle:
            raw = handle.read()
        fact_pos = raw.index(b"fact")
        sample_count = struct.unpack("<I", raw[fact_pos + 8:
                                                fact_pos + 12])[0]
        self.assertEqual(sample_count, 1024)
        self.assertEqual(raw[:4], b"RIFF")
        self.assertEqual(raw[8:12], b"WAVE")

    def test_soundfile_can_open_output(self):
        import soundfile as sf
        path = self.dir / "sf.wav"
        write_wav_24bit(path, np.zeros(4800), 48000)
        info = sf.info(str(path))
        self.assertEqual(info.samplerate, 48000)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.subtype, "PCM_24")


if __name__ == "__main__":
    unittest.main()
