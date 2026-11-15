"""Generate deterministic local test WAV samples (no real audio needed).

Covers 44.1/48 kHz, 16/24-bit, 1/2/6 channels, plus an IEEE-float file
and a corrupt file for the error-isolation paths. All signals derive
from fixed seeds, so repeated runs produce identical bytes.

Usage: python tools/gen_samples.py OUTPUT_DIR
"""

from __future__ import annotations

import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audiopipe import wavio  # noqa: E402


def _tone(rate, seconds, freq, amp, channels, phase_offsets=None):
    n = int(round(rate * seconds))
    t = np.arange(n, dtype=np.float64) / rate
    out = np.empty((n, channels), dtype=np.float64)
    for c in range(channels):
        ph = 0.0 if phase_offsets is None else phase_offsets[c % len(phase_offsets)]
        f = freq * (1.0 + 0.07 * c)
        out[:, c] = amp * np.sin(2.0 * np.pi * f * t + ph)
    return out


def _noise(rate, seconds, amp, channels, seed):
    rng = np.random.RandomState(seed)
    n = int(round(rate * seconds))
    return (rng.standard_normal((n, channels)) * amp).astype(np.float64)


def _write_float32_wav(path, x, rate):
    x = np.ascontiguousarray(x, dtype="<f4")
    channels = x.shape[1]
    pcm = x.tobytes()
    fmt = struct.pack("<HHIIHH", 3, channels, rate,
                      rate * channels * 4, channels * 4, 32)
    riff = b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(pcm)) + b"WAVE"
    with open(path, "wb") as fh:
        fh.write(riff + b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        fh.write(b"data" + struct.pack("<I", len(pcm)) + pcm)


def generate(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    made = {}

    def emit(name, x, rate, bits):
        path = os.path.join(out_dir, name)
        wavio.write_wav(path, x, rate, bits)
        made[name] = path

    emit("tone_48k_16_stereo.wav",
         _tone(48000, 6.0, 1000.0, 0.10, 2, (0.0, 0.4)), 48000, 16)
    emit("tone_44k1_16_stereo.wav",
         _tone(44100, 5.0, 440.0, 0.20, 2, (0.0, 0.9)), 44100, 16)
    emit("noise_48k_24_mono.wav",
         _noise(48000, 4.0, 0.15, 1, seed=20261116), 48000, 24)
    emit("tone_44k1_24_6ch.wav",
         _tone(44100, 5.0, 330.0, 0.12, 6), 44100, 24)
    emit("tone_48k_16_6ch.wav",
         _tone(48000, 4.0, 660.0, 0.08, 6), 48000, 16)
    emit("tone_48k_24_stereo_loud.wav",
         _tone(48000, 3.0, 2000.0, 0.90, 2), 48000, 24)

    _write_float32_wav(os.path.join(out_dir, "bad_float32.wav"),
                       _tone(48000, 1.0, 500.0, 0.5, 2), 48000)
    with open(os.path.join(out_dir, "corrupt.wav"), "wb") as fh:
        fh.write(b"RIFF\x00\x00\x00\x00WAVEjunk-not-a-real-wav")
    made["bad_float32.wav"] = os.path.join(out_dir, "bad_float32.wav")
    made["corrupt.wav"] = os.path.join(out_dir, "corrupt.wav")
    return made


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "samples"
    paths = generate(target)
    for name in sorted(paths):
        print(os.path.join(target, name))
