"""Deterministic sine/harmonic synthesis, bounded resampling, and PCM helpers."""

from __future__ import annotations

import math

import numpy as np

from .config import BakeConfig
from .input import Line, Voice

SILENCE_DBFS = -120.0


def _harmonic_wave(
    voice: Voice,
    sample_rate: int,
    length: int,
) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    t = np.arange(length, dtype=np.float64) / float(sample_rate)
    phase_modulation = np.zeros(length, dtype=np.float64)
    if voice.vibrato_hz > 0 and voice.vibrato_depth_rad > 0:
        phase_modulation = voice.vibrato_depth_rad * np.sin(
            2.0 * math.pi * voice.vibrato_hz * t
        )

    energy = sum(abs(item.gain) for item in voice.harmonics)
    wave = np.zeros(length, dtype=np.float64)
    for harmonic in voice.harmonics:
        phase = math.radians(harmonic.phase_deg)
        wave += harmonic.gain * np.sin(
            2.0
            * math.pi
            * voice.fundamental_hz
            * harmonic.multiple
            * t
            + phase
            + phase_modulation
        )
    return wave * (voice.amplitude / energy)


def synthesize_line(
    line: Line,
    voice: Voice,
    config: BakeConfig,
) -> np.ndarray:
    """Render a line with overlapping short cross-fades.

    Syllable duration annotations are onset-to-onset durations. Internal
    syllables carry a short tail that overlaps the following syllable; the
    final tail is excluded so the line length is exactly the sum of the
    annotated durations.
    """

    durations = [item.duration_samples for item in line.syllables]
    total = sum(durations)
    output = np.zeros(total, dtype=np.float64)
    crossfade = config.crossfade_samples
    attack = config.attack_samples
    release = config.release_samples
    cursor = 0
    last_index = len(durations) - 1

    for index, duration in enumerate(durations):
        is_last = index == last_index
        remaining = total - (cursor + duration)
        tail = min(crossfade, remaining) if not is_last else 0
        length = duration + tail
        wave = _harmonic_wave(voice, config.sample_rate, length)
        envelope = np.ones(length, dtype=np.float64)

        if index == 0:
            fade_in = min(attack, duration)
        else:
            fade_in = min(crossfade, duration)
        if fade_in > 0:
            envelope[:fade_in] = np.arange(fade_in, dtype=np.float64) / fade_in

        if is_last:
            fade_out = min(release, duration)
            if fade_out > 0:
                envelope[duration - fade_out : duration] = (
                    np.arange(fade_out - 1, -1, -1, dtype=np.float64) / fade_out
                )
        elif tail > 0:
            envelope[duration : duration + tail] = (
                np.arange(tail - 1, -1, -1, dtype=np.float64) / tail
            )

        output[cursor : cursor + length] += wave * envelope
        cursor += duration

    return output


def bounded_stretch(
    audio: np.ndarray,
    target_samples: int,
    min_ratio: float,
    max_ratio: float,
) -> tuple[np.ndarray | None, float, str | None]:
    """Linearly resample to the slot duration after enforcing configured bounds."""

    source_samples = int(audio.size)
    if source_samples <= 0 or target_samples <= 0:
        return None, 1.0, "EMPTY_AUDIO"
    ratio = target_samples / source_samples
    if ratio < min_ratio - 1e-12 or ratio > max_ratio + 1e-12:
        return None, ratio, "STRETCH_LIMIT_EXCEEDED"
    if source_samples == target_samples:
        return audio.copy(), 1.0, None
    if target_samples == 1:
        return np.asarray([audio[0]], dtype=np.float64), ratio, None
    source_positions = np.arange(source_samples, dtype=np.float64)
    target_positions = (
        np.arange(target_samples, dtype=np.float64)
        * ((source_samples - 1) / (target_samples - 1))
    )
    return np.interp(target_positions, source_positions, audio), ratio, None


def quantize_pcm16(audio: np.ndarray) -> np.ndarray:
    """Deterministic symmetric quantization to little-endian int16 samples."""

    clipped = np.clip(np.asarray(audio, dtype=np.float64), -1.0, 1.0) * 32767.0
    rounded = np.sign(clipped) * np.floor(np.abs(clipped) + 0.5)
    return np.asarray(rounded, dtype="<i2")


def pcm16_bytes(audio: np.ndarray) -> bytes:
    return np.asarray(audio, dtype="<i2").tobytes(order="C")


def pcm16_from_bytes(data: bytes) -> np.ndarray:
    if len(data) % 2:
        raise ValueError("PCM byte length must be divisible by two")
    return np.frombuffer(data, dtype="<i2").copy()


def dbfs(value: float) -> float:
    if value <= 0:
        return SILENCE_DBFS
    return 20.0 * math.log10(value)


def pcm_stats(audio: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    if audio.size == 0:
        return {
            "sample_peak": 0,
            "peak_dbfs": SILENCE_DBFS,
            "rms_dbfs": SILENCE_DBFS,
            "duration_samples": 0,
            "duration_seconds": 0.0,
        }
    absolute = np.abs(audio.astype(np.int64))
    peak = int(absolute.max())
    rms = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
    return {
        "sample_peak": peak,
        "peak_dbfs": round(dbfs(peak / 32768.0), 9),
        "rms_dbfs": round(dbfs(rms / 32768.0), 9),
        "duration_samples": int(audio.size),
        "duration_seconds": round(audio.size / sample_rate, 9),
    }


def place_and_mix_pcm(
    placed_pcm: list[tuple[int, np.ndarray]],
    timeline_samples: int,
) -> np.ndarray:
    mix = np.zeros(timeline_samples, dtype=np.int64)
    for start, audio in placed_pcm:
        if start < 0 or start + audio.size > timeline_samples:
            raise ValueError("entry exceeds timeline")
        mix[start : start + audio.size] += audio.astype(np.int64)
    return mix


def mix_stats(mix: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    if mix.size == 0:
        return {
            "sample_peak": 0,
            "peak_dbfs": SILENCE_DBFS,
            "rms_dbfs": SILENCE_DBFS,
            "timeline_samples": 0,
            "duration_seconds": 0.0,
            "clipped_sample_count": 0,
        }
    absolute = np.abs(mix)
    peak = int(absolute.max())
    rms = float(np.sqrt(np.mean(np.square(mix.astype(np.float64)))))
    return {
        "sample_peak": peak,
        "peak_dbfs": round(dbfs(peak / 32768.0), 9),
        "rms_dbfs": round(dbfs(rms / 32768.0), 9),
        "timeline_samples": int(mix.size),
        "duration_seconds": round(mix.size / sample_rate, 9),
        "clipped_sample_count": int(np.count_nonzero(absolute > 32767)),
    }
