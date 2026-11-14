"""Unit tests for PCM WAV I/O and SRT parsing/correction."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from audio_norm.srt import (
    SrtFormatError,
    SubtitleEntry,
    adjust_timeline,
    format_srt,
    parse_srt,
)
from audio_norm.wavio import (
    WavFormatError,
    read_wav,
    write_wav_24bit,
)
import wave


class WavTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.mkdtemp()

    def _write_pcm(self, path, samples, rate, width, channels):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(width)
            wf.setframerate(rate)
            scale = 32767 if width == 2 else 8388607
            pcm = np.clip(np.round(samples * scale), -scale, scale)
            if width == 2:
                wf.writeframes(pcm.astype("<i2").tobytes())
            else:
                flat = pcm.astype(np.int32).reshape(-1)
                flat = np.where(flat < 0, flat + 0x1000000, flat)
                packed = np.empty((flat.size, 3), dtype=np.uint8)
                packed[:, 0] = flat & 0xFF
                packed[:, 1] = (flat >> 8) & 0xFF
                packed[:, 2] = (flat >> 16) & 0xFF
                wf.writeframes(packed.tobytes())

    def test_roundtrip_16_and_24_bit(self):
        samples = (0.5 * np.sin(
            2 * np.pi * 440 * np.arange(4800) / 48000
        )).reshape(-1, 1)
        for width in (2, 3):
            path = os.path.join(self.temp, f"w{width}.wav")
            self._write_pcm(path, samples, 48000, width, 1)
            loaded = read_wav(path)
            self.assertEqual(loaded.sample_rate, 48000)
            self.assertEqual(loaded.sample_width, width)
            np.testing.assert_allclose(
                loaded.samples, samples, atol=1 / (2 ** (8 * width - 1))
            )

    def test_write_24bit_output(self):
        stereo = np.column_stack([
            0.3 * np.ones(1000), -0.3 * np.ones(1000)
        ])
        path = os.path.join(self.temp, "out24.wav")
        write_wav_24bit(path, stereo, 48000)
        loaded = read_wav(path)
        self.assertEqual(loaded.sample_width, 3)
        self.assertEqual(loaded.n_channels, 2)
        self.assertEqual(loaded.sample_rate, 48000)

    def test_unsupported_sample_rate_rejected(self):
        path = os.path.join(self.temp, "32k.wav")
        self._write_pcm(path, np.zeros(1000), 32000, 2, 1)
        with self.assertRaises(WavFormatError):
            read_wav(path)

    def test_unsupported_channels_rejected(self):
        path = os.path.join(self.temp, "quad.wav")
        self._write_pcm(path, np.zeros((1000, 4)), 48000, 2, 4)
        with self.assertRaises(WavFormatError):
            read_wav(path)

    def test_corrupt_wav_rejected(self):
        path = os.path.join(self.temp, "broken.wav")
        with open(path, "wb") as handle:
            handle.write(b"RIFF\x10\x00\x00\x00WAVEfmt ")
        with self.assertRaises(WavFormatError):
            read_wav(path)


class SrtTests(unittest.TestCase):
    def test_parse_and_roundtrip(self):
        text = (
            "1\n00:00:01,000 --> 00:00:02,500\nHello world\n\n"
            "2\n00:00:03.000 --> 00:00:04.000\nSecond cue\n"
        )
        entries = parse_srt(text)
        self.assertEqual(len(entries), 2)
        self.assertAlmostEqual(entries[0].start, 1.0)
        self.assertAlmostEqual(entries[0].end, 2.5)
        rendered = format_srt(entries)
        self.assertIn("00:00:01,000 --> 00:00:02,500", rendered)

    def test_malformed_timing_rejected(self):
        with self.assertRaises(SrtFormatError):
            parse_srt("1\nbad timing line\ntext\n")

    def test_adjust_timeline_moves_entries_in_silence(self):
        entries = [
            SubtitleEntry(1, 0.4, 1.4, "in silence"),
            SubtitleEntry(2, 5.0, 6.0, "on speech"),
        ]
        silence = [(0.0, 1.0), (3.0, 3.8)]
        adjustments = adjust_timeline(entries, silence)
        self.assertEqual(len(adjustments), 1)
        self.assertAlmostEqual(entries[0].start, 1.0)
        self.assertAlmostEqual(entries[0].end, 2.0)  # duration preserved
        self.assertAlmostEqual(entries[1].start, 5.0)


if __name__ == "__main__":
    unittest.main()
