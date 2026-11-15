"""Unit tests for the DSP building blocks. Run: python -m unittest discover tests"""

import math
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiopipe import channelmap, loudness, resample, wavio
from audiopipe.errors import Code, PipelineError
from audiopipe.pipeline import _plan_crossfades


def tone(rate, seconds, freq, amp, channels=2):
    t = np.arange(int(rate * seconds), dtype=np.float64) / rate
    x = amp * np.sin(2 * np.pi * freq * t)
    return np.tile(x[:, None], (1, channels))


class TestWavIo(unittest.TestCase):
    def test_roundtrip_16_24_32(self):
        with tempfile.TemporaryDirectory() as d:
            x = tone(48000, 0.5, 440.0, 0.5)
            for bits in (16, 24, 32):
                p = os.path.join(d, f"t{bits}.wav")
                wavio.write_wav(p, x, 48000, bits)
                y, rate, ch = wavio.read_wav(p)
                self.assertEqual((rate, ch), (48000, 2))
                err = np.max(np.abs(y - x))
                self.assertLess(err, 2.0 / (2 ** bits) + 1e-12)

    def test_roundtrip_6ch_extensible(self):
        with tempfile.TemporaryDirectory() as d:
            x = tone(48000, 0.2, 330.0, 0.3, channels=6)
            p = os.path.join(d, "six.wav")
            wavio.write_wav(p, x, 48000, 24)
            y, rate, ch = wavio.read_wav(p)
            self.assertEqual(ch, 6)
            self.assertLess(np.max(np.abs(y - x)), 1e-6)

    def test_float_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.wav")
            pcm = np.zeros((100, 2), dtype="<f4").tobytes()
            fmt = struct.pack("<HHIIHH", 3, 2, 48000, 48000 * 8, 8, 32)
            riff = b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(pcm)) + b"WAVE"
            with open(p, "wb") as fh:
                fh.write(riff + b"fmt " + struct.pack("<I", len(fmt)) + fmt)
                fh.write(b"data" + struct.pack("<I", len(pcm)) + pcm)
            with self.assertRaises(PipelineError) as ctx:
                wavio.read_wav(p)
            self.assertEqual(ctx.exception.code, Code.E_UNSUPPORTED_FORMAT)

    def test_corrupt_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.wav")
            with open(p, "wb") as fh:
                fh.write(b"not a wav at all")
            with self.assertRaises(PipelineError) as ctx:
                wavio.read_wav(p)
            self.assertEqual(ctx.exception.code, Code.E_DECODE_ERROR)


class TestResample(unittest.TestCase):
    def test_length_and_identity(self):
        x = tone(48000, 1.0, 1000.0, 0.5)
        self.assertIs(resample.resample(x, 48000, 48000), x)
        y = resample.resample(tone(44100, 1.0, 1000.0, 0.5), 44100, 48000)
        self.assertEqual(y.shape[0], 48000)
        y2 = resample.resample(x, 48000, 44100)
        self.assertEqual(y2.shape[0], 44100)

    def test_tone_preserved_no_images(self):
        x = tone(44100, 2.0, 440.0, 0.2, channels=1)
        y = resample.resample(x, 44100, 48000)
        self.assertAlmostEqual(np.sqrt((y ** 2).mean()),
                               np.sqrt((x ** 2).mean()), places=4)
        spec = np.abs(np.fft.rfft(y[:, 0]))
        freqs = np.fft.rfftfreq(len(y), 1 / 48000)
        peak = freqs[np.argmax(spec)]
        self.assertAlmostEqual(peak, 440.0, delta=1.0)
        spec2 = spec.copy()
        spec2[np.abs(freqs - 440) < 200] = 0
        self.assertLess(spec2.max(), spec.max() * 10 ** (-55 / 20))

    def test_deterministic(self):
        x = tone(44100, 0.5, 1000.0, 0.3)
        a = resample.resample(x, 44100, 48000)
        b = resample.resample(x, 44100, 48000)
        self.assertTrue(np.array_equal(a, b))


class TestLoudness(unittest.TestCase):
    def test_sine_reference(self):
        # 1 kHz sine at -20 dBFS reads about -20 LKFS (K-weight ~ +0.7 dB
        # at 1 kHz cancels the -0.691 offset).
        x = tone(48000, 4.0, 1000.0, 0.1)
        loud = loudness.integrated_loudness(x, loudness.CHANNEL_WEIGHTS[2])
        self.assertAlmostEqual(loud, -20.0, delta=0.3)

    def test_silence_gated_out(self):
        x = np.zeros((48000 * 2, 2))
        self.assertEqual(loudness.integrated_loudness(
            x, loudness.CHANNEL_WEIGHTS[2]), float("-inf"))

    def test_surround_weight(self):
        x = tone(48000, 2.0, 500.0, 0.1, channels=6)
        loud6 = loudness.integrated_loudness(x, loudness.CHANNEL_WEIGHTS[6])
        loud2 = loudness.integrated_loudness(x[:, :2], loudness.CHANNEL_WEIGHTS[2])
        # 5.82x weighted power vs 2x -> +10*log10(5.82/2) dB
        self.assertAlmostEqual(loud6 - loud2, 10 * math.log10(5.82 / 2), delta=0.1)


class TestDownmix(unittest.TestCase):
    def test_bs775_coefficients(self):
        x = np.zeros((4, 6))
        x[:, 0] = 1.0   # L
        x[:, 2] = 1.0   # C
        x[:, 4] = 1.0   # Ls
        x[:, 3] = 1.0   # LFE (dropped)
        out = channelmap.to_stereo(x)
        self.assertAlmostEqual(out[0, 0], 1.0 + 2 * channelmap.SQRT1_2)
        self.assertAlmostEqual(out[0, 1], channelmap.SQRT1_2)

    def test_mono_to_stereo(self):
        x = np.ones((4, 1))
        out = channelmap.to_stereo(x)
        self.assertEqual(out.shape, (4, 2))
        self.assertAlmostEqual(out[0, 0], channelmap.SQRT1_2)

    def test_unsupported_layout(self):
        with self.assertRaises(PipelineError):
            channelmap.to_stereo(np.zeros((4, 4)))


class TestCrossfadePlan(unittest.TestCase):
    def test_no_overlap_and_truncation(self):
        entries = [{"warnings": []} for _ in range(3)]
        lengths = [4800, 100, 4800]  # tiny middle segment
        ds = _plan_crossfades(lengths, 960, entries)
        self.assertLessEqual(ds[0] + ds[1], lengths[1])
        self.assertTrue(any("XFADE_TRUNCATED" in e["warnings"] for e in entries))

    def test_normal(self):
        entries = [{"warnings": []} for _ in range(3)]
        ds = _plan_crossfades([48000, 48000, 48000], 960, entries)
        self.assertEqual(ds, [960, 960])
        self.assertFalse(any(e["warnings"] for e in entries))


if __name__ == "__main__":
    unittest.main()
