from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cinemix.errors import CinemixError
from cinemix.wavio import (
    read_pcm_wav,
    riff_info,
    write_pcm16_wav,
    write_pcm_wav,
)


class WavIoTests(unittest.TestCase):
    def test_24_bit_roundtrip(self) -> None:
        sample_rate = 48000
        time = np.arange(4800) / sample_rate
        signal = np.stack(
            (
                np.sin(2.0 * np.pi * 440.0 * time),
                np.cos(2.0 * np.pi * 660.0 * time) * 0.5,
            ),
            axis=1,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tone.wav"
            write_pcm_wav(path, signal, sample_rate)
            decoded, info = read_pcm_wav(path)
            self.assertEqual(info.bit_depth, 24)
            self.assertEqual(info.channels, 2)
            self.assertEqual(info.sample_rate, sample_rate)
            self.assertLess(float(np.max(np.abs(decoded - signal))), 1.5e-7)
            metadata = riff_info(path)
            self.assertEqual(metadata["format_tag"], 1)
            self.assertEqual(metadata["bit_depth"], 24)

    def test_16_bit_roundtrip(self) -> None:
        sample_rate = 44100
        time = np.arange(4410) / sample_rate
        signal = np.sin(2.0 * np.pi * 220.0 * time)[:, None] * 0.5
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tone16.wav"
            write_pcm16_wav(path, signal, sample_rate)
            decoded, info = read_pcm_wav(path)
            self.assertEqual(info.bit_depth, 16)
            self.assertEqual(info.channels, 1)
            self.assertLess(float(np.max(np.abs(decoded - signal))), 1.0 / 32768.0)

    def test_missing_input_has_stable_code(self) -> None:
        with self.assertRaises(CinemixError) as raised:
            read_pcm_wav("/definitely/not/present.wav")
        self.assertEqual(raised.exception.code, "E_INPUT_MISSING")


if __name__ == "__main__":
    unittest.main()

