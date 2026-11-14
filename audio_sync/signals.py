"""Deterministic local signal generation and PCM helpers.

No audio device or random entropy is used: every stream is a sum of sine
tones multiplied by a deterministic talkspurt envelope, so two runs with the
same parameters produce byte-identical samples.
"""

from __future__ import annotations

import numpy as np

from .constants import FRAME_SAMPLES, SAMPLE_RATE


def pcm_bytes_to_float(payload: bytes) -> np.ndarray:
    """Decode little-endian int16 PCM to float32 in the [-1, 1] range."""
    samples = np.frombuffer(payload, dtype="<i2").astype(np.float32)
    return samples / 32768.0


def float_to_pcm_bytes(frame: np.ndarray) -> bytes:
    """Clip, dither-free quantize and encode a float frame to int16 PCM."""
    clipped = np.clip(frame, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def generate_stream_signal(
    stream_id: int,
    frame_count: int,
    base_frequency: float = 220.0,
) -> np.ndarray:
    """Generate ``frame_count`` frames of deterministic mono audio.

Each stream gets a unique pair of harmonic frequencies (so mixed streams stay
distinguishable) and a deterministic amplitude envelope that emulates
talkspurts, interleaving active speech-like bursts with near-silence.
"""
    total_samples = frame_count * FRAME_SAMPLES
    time_axis = np.arange(total_samples, dtype=np.float64) / SAMPLE_RATE

    # Deterministic per-stream frequencies: musical intervals apart.
    multiplier = 1.0 + 0.25 * stream_id
    tone_a = np.sin(
        2.0 * np.pi * base_frequency * multiplier * time_axis
    )
    tone_b = 0.5 * np.sin(
        2.0 * np.pi * base_frequency * 2.0 * multiplier * time_axis
        + stream_id * 0.7
    )

    # Talkspurt envelope: slow square-ish gating smoothed by a raised cosine,
    # giving ~160 ms active bursts separated by ~120 ms gaps.
    period_frames = 14
    active_frames = 8
    frame_pos = np.arange(frame_count) % period_frames
    gate = (frame_pos < active_frames).astype(np.float64)
    # Per-stream phase shift so talkspurts do not line up across streams.
    shift = (stream_id * 5) % period_frames
    gate = np.roll(gate, shift)
    envelope = np.repeat(gate, FRAME_SAMPLES)

    # Smooth the 20 ms boundaries with a short raised-cosine ramp.
    ramp = min(120, FRAME_SAMPLES // 4)
    window = np.ones(FRAME_SAMPLES, dtype=np.float64)
    window[:ramp] = 0.5 * (1.0 - np.cos(np.pi * np.arange(ramp) / ramp))
    window[-ramp:] = window[:ramp][::-1]
    envelope *= np.tile(window, frame_count)

    signal = (tone_a + tone_b) * (0.55 * envelope + 0.02)
    return signal.astype(np.float32)


def frame_from_signal(signal: np.ndarray, frame_index: int) -> np.ndarray:
    """Slice one 20 ms frame from a full-stream float signal."""
    start = frame_index * FRAME_SAMPLES
    return signal[start:start + FRAME_SAMPLES].copy()


def rms_level(frame: np.ndarray) -> float:
    """Root-mean-square level of a float frame (0 for empty input)."""
    if frame.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
