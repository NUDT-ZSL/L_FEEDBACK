"""Audio decoding, placement, envelope application and mixing."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import ffmpeg_utils
from .errors import ProcessError
from .ffmpeg_utils import SAMPLE_RATE
from .hashing import sha256_file
from .project import Track


def total_video_samples(frame_count: int, frame_rate) -> int:
    """Number of 48 kHz samples covering the whole video (integer arithmetic)."""
    return frame_count * SAMPLE_RATE * frame_rate.denominator // frame_rate.numerator


def offset_to_samples(offset_ms: int) -> int:
    return offset_ms * SAMPLE_RATE // 1000


def decode_track(track: Track, cache_dir: Path) -> np.ndarray:
    """Decode a track to float32 stereo PCM, cached by source file content hash."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(track.path)
    cache_path = cache_dir / f"{digest}.f32le"
    if not cache_path.exists():
        pcm = ffmpeg_utils.decode_audio_pcm(track.path)
        tmp_path = cache_path.with_suffix(".f32le.tmp")
        with open(tmp_path, "wb") as handle:
            handle.write(pcm)
        os.replace(tmp_path, cache_path)
    raw = cache_path.read_bytes()
    if len(raw) % (2 * 4) != 0:
        raise ProcessError(
            f"decoded audio for track {track.index} is not stereo float32 aligned"
        )
    samples = np.frombuffer(raw, dtype="<f4")
    return samples.reshape(-1, 2).astype(np.float32, copy=True)


def apply_envelope(samples: np.ndarray, envelope: tuple[float, ...]) -> np.ndarray:
    """Linearly interpolate envelope points across the track's own duration."""
    frame_count = samples.shape[0]
    point_count = len(envelope)
    if frame_count == 0:
        return samples
    point_positions = np.linspace(0.0, frame_count - 1, point_count)
    indices = np.arange(frame_count, dtype=np.float64)
    gains = np.interp(indices, point_positions, np.asarray(envelope, dtype=np.float64))
    return samples.astype(np.float64) * gains[:, np.newaxis]


def place_track(
    track_samples: np.ndarray,
    offset_ms: int,
    total_samples: int,
) -> np.ndarray:
    """Map a track onto the global timeline, honoring negative/positive offsets."""
    shift = offset_to_samples(offset_ms)
    source_len = track_samples.shape[0]

    source_start = max(0, -shift)
    source_end = min(source_len, total_samples - shift)
    if source_end <= source_start:
        return np.zeros((total_samples, 2), dtype=np.float64)

    dest_start = max(0, shift)
    dest_end = dest_start + (source_end - source_start)
    placed = np.zeros((total_samples, 2), dtype=np.float64)
    placed[dest_start:dest_end] = track_samples[source_start:source_end]
    return placed


def mix_tracks(
    tracks: tuple[Track, ...],
    decoded: list[np.ndarray],
    total_samples: int,
) -> tuple[np.ndarray, float]:
    """Sum all tracks and apply peak-normalization protection.

    Returns the float32 stereo mix and the applied scaling factor.
    """
    mix = np.zeros((total_samples, 2), dtype=np.float64)
    for track, samples in zip(tracks, decoded):
        shaped = apply_envelope(samples, track.envelope)
        mix += place_track(shaped, track.offset_ms, total_samples)

    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1.0:
        scale = 0.99 / peak
        mix *= scale
    else:
        scale = 1.0
    return mix.astype(np.float32), scale
