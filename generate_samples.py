"""Deterministically generate WAV dialogue tracks and matching SRT files.

No external assets are used: every waveform is derived from a fixed-seed
NumPy random generator.  Run with the project virtual environment:

    python generate_samples.py --output samples/input
"""
from __future__ import annotations

import argparse
import math
import os
import wave

import numpy as np

from audio_norm.srt import format_srt, SubtitleEntry


def _write_wav(path: str, samples: np.ndarray, sample_rate: int,
               sample_width: int) -> None:
    n_channels = samples.shape[1]
    with wave.open(path, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        if sample_width == 2:
            scale = 32767.0
        elif sample_width == 3:
            scale = 8388607.0
        else:
            raise ValueError("only 16/24-bit output supported")
        pcm = np.clip(np.round(samples * scale), -scale, scale)
        pcm = pcm.reshape(-1).astype(np.int32)
        if sample_width == 2:
            wf.writeframes(pcm.astype("<i2").tobytes())
        else:
            packed = np.empty((pcm.size, 3), dtype=np.uint8)
            flat = np.where(pcm < 0, pcm + 0x1000000, pcm)
            packed[:, 0] = flat & 0xFF
            packed[:, 1] = (flat >> 8) & 0xFF
            packed[:, 2] = (flat >> 16) & 0xFF
            wf.writeframes(packed.tobytes())


def _speech_like(rng: np.random.Generator, duration: float, fs: int,
                  amplitude: float) -> np.ndarray:
    n = int(round(duration * fs))
    t = np.arange(n) / fs
    f0 = 130.0
    # Harmonic series + amplitude modulation gives voiced-speech-like energy.
    signal = np.zeros(n)
    for harmonic, level in enumerate((1.0, 0.6, 0.35, 0.2, 0.12), start=1):
        signal += level * np.sin(2 * math.pi * f0 * harmonic * t)
    signal /= np.max(np.abs(signal))
    syllable = 0.5 + 0.5 * np.sin(2 * math.pi * 3.2 * t)
    shimmer = 10.0 ** (rng.normal(0.0, 0.08, n) / 20.0)
    env = np.clip(syllable, 0.05, 1.0) * shimmer
    shaped = signal * env
    # Gentle high-frequency breath noise.
    shaped += 0.03 * rng.standard_normal(n)
    return amplitude * shaped / np.max(np.abs(shaped))


def _layout(signal: np.ndarray, layout: str) -> np.ndarray:
    if layout == "mono":
        return signal.reshape(-1, 1)
    if layout == "stereo":
        stereo = np.column_stack([signal, np.roll(signal, 37)])
        return stereo
    if layout == "5.1":
        # L, R, C, LFE, Ls, Rs.  Dialogue mostly on center; LFE carries low
        # content; surrounds carry quieter ambience.
        n = signal.shape[0]
        center = signal
        left = 0.7 * signal
        right = 0.7 * np.roll(signal, 53)
        lfe = 0.4 * np.sin(2 * math.pi * 60 * np.arange(n) / 48000.0)
        left_surround = 0.25 * np.roll(signal, 101)
        right_surround = 0.25 * np.roll(signal, 211)
        return np.column_stack(
            [left, right, center, lfe, left_surround, right_surround]
        )
    raise ValueError(f"unknown layout {layout}")


def _with_silence(fs: int, layout: str, speech_regions, rng: np.random.Generator,
                  total_seconds: float):
    total = int(round(total_seconds * fs))
    mono = np.zeros(total)
    for start, duration, amplitude in speech_regions:
        speech = _speech_like(rng, duration, fs, amplitude)
        idx = int(round(start * fs))
        mono[idx:idx + speech.shape[0]] = speech
    # Very faint noise floor, well below -60 LUFS.
    mono += 10.0 ** (-72.0 / 20.0) * rng.standard_normal(total) * 0.02
    return _layout(mono, layout)


def _write_srt(path: str, entries: list[SubtitleEntry]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(format_srt(entries))


def generate(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(20240517)

    # 1) Stereo 48k/24-bit, loud track with a leading silence and two cues
    #    whose starts fall inside silent regions.
    stereo = _with_silence(
        48000, "stereo",
        [(1.0, 2.5, 0.85), (4.5, 2.0, 0.8), (7.5, 1.5, 0.75)],
        rng, 10.0,
    )
    _write_wav(os.path.join(output_dir, "dialogue_stereo.wav"),
               stereo, 48000, 3)
    _write_srt(os.path.join(output_dir, "dialogue_stereo.srt"), [
        SubtitleEntry(1, 0.40, 1.40, "Opening line inside the leading silence"),
        SubtitleEntry(2, 4.20, 5.10, "Second line starts inside a short gap"),
        SubtitleEntry(3, 8.00, 9.00, "Third line already on speech"),
    ])

    # 2) Mono 44.1k/16-bit quiet track requiring large positive gain and
    #    clipping/limiting after normalization.
    mono = _with_silence(
        44100, "mono", [(0.5, 3.0, 0.18), (4.5, 3.0, 0.16)],
        rng, 8.5,
    )
    _write_wav(os.path.join(output_dir, "dialogue_mono_quiet.wav"),
               mono, 44100, 2)
    _write_srt(os.path.join(output_dir, "dialogue_mono_quiet.srt"), [
        SubtitleEntry(1, 0.20, 1.20, "Quiet opening line shifted to speech"),
        SubtitleEntry(2, 5.00, 6.00, "Second line stays aligned"),
    ])

    # 3) 5.1 48k/24-bit moderate track.
    surround = _with_silence(
        48000, "5.1", [(0.8, 3.0, 0.5), (4.8, 2.5, 0.55)],
        rng, 8.0,
    )
    _write_wav(os.path.join(output_dir, "dialogue_surround.wav"),
               surround, 48000, 3)
    _write_srt(os.path.join(output_dir, "dialogue_surround.srt"), [
        SubtitleEntry(1, 0.30, 1.30, "Surround cue shifted to first speech"),
        SubtitleEntry(2, 5.20, 6.20, "Surround cue already aligned"),
    ])

    # 4) 44.1k/24-bit stereo track with transient peaks to exercise limiter.
    peaking = _with_silence(
        44100, "stereo", [(0.3, 4.0, 0.95), (5.0, 2.5, 0.9)],
        rng, 8.0,
    )
    n_transient = int(0.02 * 44100)
    for position in (1.0, 3.0, 6.0):
        idx = int(position * 44100)
        peaking[idx:idx + n_transient] *= 1.6
    _write_wav(os.path.join(output_dir, "dialogue_peaking.wav"),
               peaking, 44100, 3)
    _write_srt(os.path.join(output_dir, "dialogue_peaking.srt"), [
        SubtitleEntry(1, 0.10, 1.10, "Transient-heavy cue shifted to speech"),
    ])

    # 5) Corrupt WAV (truncated header/data).
    # 4b) Moderate-loudness stereo 48k/16-bit track whose integrated loudness
    #     is already near -23 LUFS but whose transients sit near full scale, so
    #     the small gain toward target pushes true peak beyond -1.0 dBTP and
    #     the limiter has to engage while loudness stays on target.
    dense = _with_silence(
        48000, "stereo", [(0.3, 3.0, 0.12), (3.6, 3.4, 0.12)],
        rng, 7.2,
    )
    # Narrow full-scale transients produce high oversampled (intersample) peaks
    # without materially raising integrated loudness.
    for position in (1.0, 2.2, 4.0, 5.5):
        idx = int(position * 48000)
        dense[idx:idx + 8] = 0.999
    _write_wav(os.path.join(output_dir, "dialogue_dense_loud.wav"),
               dense, 48000, 2)
    _write_srt(os.path.join(output_dir, "dialogue_dense_loud.srt"), [
        SubtitleEntry(1, 0.10, 1.10, "Dense loud cue shifted to first speech"),
        SubtitleEntry(2, 4.20, 5.00, "Second dense cue already on speech"),
    ])

    # 5) Corrupt WAV (truncated header/data).
    with open(os.path.join(output_dir, "dialogue_corrupt.wav"), "wb") as handle:
        handle.write(b"RIFF\x10\x00\x00\x00WAVEfmt ")
    with open(os.path.join(output_dir, "dialogue_corrupt.srt"), "w",
              encoding="utf-8") as handle:
        handle.write(
            "1\n00:00:00,500 --> 00:00:01,500\nCorrupt track still has a cue\n"
        )

    # 6) Valid WAV paired with a malformed SRT.
    bad_srt_audio = _with_silence(
        48000, "mono", [(0.5, 2.0, 0.6)], rng, 4.0
    )
    _write_wav(os.path.join(output_dir, "dialogue_bad_srt.wav"),
               bad_srt_audio, 48000, 2)
    with open(os.path.join(output_dir, "dialogue_bad_srt.srt"), "w",
              encoding="utf-8") as handle:
        handle.write("1\nnot a valid timing line\nBroken subtitle block\n")

    # 7) Unsupported channel count (4 channels).
    n = 48000 * 2
    quad = np.column_stack([rng.standard_normal(n)] * 4) * 0.3
    _write_wav(os.path.join(output_dir, "dialogue_quad.wav"),
               quad, 48000, 2)
    _write_srt(os.path.join(output_dir, "dialogue_quad.srt"), [
        SubtitleEntry(1, 0.2, 1.0, "Unsupported quadraphonic layout"),
    ])

    # 8) Unsupported sample rate (32 kHz).
    n32 = 32000 * 2
    low_rate = rng.standard_normal(n32).reshape(-1, 1) * 0.3
    _write_wav(os.path.join(output_dir, "dialogue_32k.wav"),
               low_rate, 32000, 2)
    _write_srt(os.path.join(output_dir, "dialogue_32k.srt"), [
        SubtitleEntry(1, 0.2, 1.0, "Unsupported 32 kHz sample rate"),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="generate deterministic samples")
    parser.add_argument("--output", default="samples/input")
    args = parser.parse_args()
    generate(args.output)
    print(f"sample fixtures written to {args.output}")


if __name__ == "__main__":
    main()
