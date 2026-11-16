"""Deterministic local WAV fixtures; no external audio is required."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

from .analysis import round_half_up
from .wav_io import SAMPLE_RATE, samples_to_pcm16


def make_click_track(
    bpm: float,
    *,
    seconds: float = 10.0,
    sample_rate: int = SAMPLE_RATE,
    channels: int = 1,
    seed: int = 7,
    amplitude: float = 0.82,
) -> np.ndarray:
    """Create a deterministic percussive tonal track on a quarter-note grid."""

    frame_count = max(1, int(round(seconds * sample_rate)))
    output = np.zeros((frame_count, channels), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    beat_seconds = 60.0 / bpm
    beat_index = 0
    beat_frames = beat_seconds * sample_rate

    while True:
        start = round_half_up(beat_index * beat_frames)
        if start >= frame_count:
            break
        length = min(int(0.075 * sample_rate), frame_count - start)
        time = np.arange(length, dtype=np.float64) / sample_rate
        frequency = 620.0 if beat_index % 4 else 410.0
        envelope = np.exp(-time * 46.0)
        noise = rng.standard_normal(length) * 0.18
        click = envelope * (
            0.72 * np.sin(2.0 * np.pi * frequency * time)
            + 0.24 * np.sin(2.0 * np.pi * (frequency * 1.51) * time)
            + noise
        )
        for channel in range(channels):
            channel_gain = 1.0 if channels == 1 else (0.92 if channel == 0 else 0.78)
            output[start : start + length, channel] += click * channel_gain
        beat_index += 1

    peak = float(np.max(np.abs(output)))
    if peak > 0.0:
        output *= min(1.0, amplitude / peak)
    return output


def make_percussion_track(
    bpm: float,
    *,
    seconds: float = 8.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 11,
    amplitude: float = 0.78,
) -> np.ndarray:
    """Create deterministic stereo kick/snare-style percussion."""

    frame_count = int(round(seconds * sample_rate))
    output = np.zeros((frame_count, 2), dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    beat_frames = 60.0 / bpm * sample_rate
    beat_index = 0

    while True:
        start = round_half_up(beat_index * beat_frames)
        if start >= frame_count:
            break
        length = min(int(0.18 * sample_rate), frame_count - start)
        time = np.arange(length, dtype=np.float64) / sample_rate
        kick_frequency = 62.0 + 70.0 * np.exp(-time * 28.0)
        kick = np.sin(2.0 * np.pi * kick_frequency * time) * np.exp(-time * 18.0)

        hat_start = min(frame_count - 1, start + round_half_up(beat_frames * 0.5))
        hat_length = min(int(0.045 * sample_rate), frame_count - hat_start)
        hat_time = np.arange(hat_length, dtype=np.float64) / sample_rate
        hat = (
            rng.standard_normal(hat_length)
            * np.exp(-hat_time * 95.0)
            * 0.22
        )

        output[start : start + length, 0] += 0.88 * kick
        output[start : start + length, 1] += 0.88 * kick
        output[hat_start : hat_start + hat_length] += hat[:, None]
        beat_index += 1

    peak = float(np.max(np.abs(output)))
    if peak > 0.0:
        output *= min(1.0, amplitude / peak)
    return output


def make_low_confidence_noise(
    *,
    seconds: float = 6.0,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 99,
) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(seed))
    frame_count = int(round(seconds * sample_rate))
    noise = rng.standard_normal((frame_count, 1)) * 0.035
    return np.clip(noise, -0.1, 0.1)


def write_wav(
    path: str | Path,
    samples: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    sampwidth: int = 2,
) -> None:
    values = np.asarray(samples)
    if values.ndim != 2:
        raise ValueError("samples must be two-dimensional")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file_handle:
        with wave.open(file_handle, "wb") as wav_handle:
            wav_handle.setnchannels(values.shape[1])
            wav_handle.setsampwidth(sampwidth)
            wav_handle.setframerate(sample_rate)
            if sampwidth == 2:
                wav_handle.writeframes(samples_to_pcm16(values))
            else:
                quantized = np.clip(
                    np.rint(values * ((1 << (8 * sampwidth - 1)) - 1)),
                    -(1 << (8 * sampwidth - 1)),
                    (1 << (8 * sampwidth - 1)) - 1,
                ).astype(np.int8 if sampwidth == 1 else "<i4")
                wav_handle.writeframes(quantized.tobytes(order="C"))


def generate_fixture_set(directory: str | Path) -> dict[str, Path]:
    """Generate all valid and invalid fixtures used by local verification."""

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "skill": destination / "skill_120_mono.wav",
        "percussion": destination / "percussion_120_stereo.wav",
        "stereo": destination / "skill_128_stereo.wav",
        "bad_rate": destination / "bad_sample_rate_48000.wav",
        "bad_depth": destination / "bad_bit_depth_8.wav",
        "bad_header": destination / "bad_header.wav",
        "bad_truncated": destination / "bad_truncated_data.wav",
        "low_confidence": destination / "noise_low_confidence.wav",
    }

    skill = make_click_track(120.0, seconds=10.0, channels=1)
    percussion = make_percussion_track(120.0, seconds=8.0)
    stereo = make_click_track(128.0, seconds=10.0, channels=2)
    noise = make_low_confidence_noise(seconds=6.0)

    write_wav(paths["skill"], skill)
    write_wav(paths["percussion"], percussion)
    write_wav(paths["stereo"], stereo)
    write_wav(paths["bad_rate"], skill, sample_rate=48_000)
    write_wav(paths["bad_depth"], skill, sampwidth=1)
    paths["bad_header"].write_bytes(b"RIFF\x24\x00\x00\x00WAVEbroken")
    paths["bad_truncated"].write_bytes(paths["skill"].read_bytes()[:-100])
    write_wav(paths["low_confidence"], noise)
    return paths

