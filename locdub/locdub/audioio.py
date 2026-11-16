"""Audio reading, validation and per-segment adaptation."""

from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import soundfile as sf

from . import wsola


class AudioError(Exception):
    """Recoverable audio problem (missing/non-WAV/unreadable file)."""


@dataclass
class AudioInfo:
    sample_rate: int
    channels: int
    frames: int
    subtype: str

    @property
    def duration_ms(self) -> float:
        return self.frames * 1000.0 / self.sample_rate


def ensure_wav(path: Path) -> None:
    """Raise :class:`AudioError` unless ``path`` is an existing ``.wav`` file."""
    if not path.exists():
        raise AudioError(f"dub file not found: {path}")
    if path.is_dir():
        raise AudioError(f"dub path is a directory, expected WAV: {path}")
    if path.suffix.lower() != ".wav":
        raise AudioError(f"dub file must be a .wav file, got: {path.name}")


def probe_wav(path: Path) -> AudioInfo:
    ensure_wav(path)
    try:
        info = sf.info(str(path))
    except (RuntimeError, ValueError) as exc:
        raise AudioError(f"cannot read WAV metadata for {path.name}: {exc}") from exc
    if info.samplerate <= 0 or info.channels <= 0:
        raise AudioError(f"invalid WAV header in {path.name}")
    return AudioInfo(
        sample_rate=info.samplerate,
        channels=info.channels,
        frames=info.frames,
        subtype=info.subtype,
    )


def read_wav_blocks(path: Path, block_seconds: float = 60.0) -> Iterator[np.ndarray]:
    """Yield float64 blocks without loading the whole file into memory."""
    info = probe_wav(path)
    block_frames = max(1, int(round(info.sample_rate * block_seconds)))
    try:
        with sf.SoundFile(str(path), "r") as handle:
            while True:
                block = handle.read(block_frames, dtype="float64",
                                    always_2d=True)
                if len(block) == 0:
                    break
                yield block
    except (RuntimeError, ValueError) as exc:
        raise AudioError(f"error while reading {path.name}: {exc}") from exc


def read_wav_full(path: Path) -> Tuple[np.ndarray, AudioInfo]:
    blocks = list(read_wav_blocks(path))
    if not blocks:
        info = probe_wav(path)
        return np.zeros((0, info.channels), dtype=np.float64), info
    return np.concatenate(blocks, axis=0), probe_wav(path)


@dataclass
class Adaptation:
    audio: np.ndarray                 # float64, shape (frames, channels)
    sample_rate: int
    original_duration_ms: float
    adapted_duration_ms: float
    requested_ratio: float
    applied_ratio: float
    boundary_clamped: bool


def adapt_segment(path: Path, target_duration_ms: int,
                  min_ratio: float, max_ratio: float,
                  max_block_frames: int) -> Adaptation:
    """Read one WAV, stretch it to ``target_duration_ms`` and return PCM.

    The requested ratio is ``target / original``. When it falls outside the
    segment's allowed window the ratio is clamped and the caller is informed
    via ``boundary_clamped`` so it can emit a warning.
    """
    info = probe_wav(path)
    if info.frames <= 0:
        raise AudioError(f"empty WAV file: {path.name}")

    target_frames_at_native = int(round(
        info.sample_rate * target_duration_ms / 1000.0
    ))
    if target_frames_at_native <= 0:
        raise AudioError(
            f"target duration {target_duration_ms}ms yields zero samples "
            f"at {info.sample_rate}Hz"
        )

    requested_ratio = target_frames_at_native / info.frames
    applied_ratio = min(max(requested_ratio, min_ratio), max_ratio)
    clamped = not math.isclose(applied_ratio, requested_ratio, rel_tol=1e-12)

    # A dialogue segment is processed as one continuous WSOLA stream so frame
    # alignment never restarts at an artificial block boundary. Long clips
    # are guarded by max_block_frames; the final mix-down itself is fully
    # streamed block by block in mixer.py.
    if info.frames > max_block_frames:
        raise AudioError(
            f"segment {path.name} has {info.frames} frames, exceeding the "
            f"per-task limit of {max_block_frames} frames"
        )
    source, _ = read_wav_full(path)
    target_frames = int(round(info.frames * applied_ratio))
    audio = wsola.stretch_samples(source, applied_ratio, info.sample_rate)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    # Exact-length guarantee: pad with silence or hard-trim so the adapted
    # clip occupies precisely target_frames samples (deterministic).
    if len(audio) < target_frames:
        audio = np.pad(audio, ((0, target_frames - len(audio)), (0, 0)))
    elif len(audio) > target_frames:
        audio = audio[:target_frames]

    adapted_ms = target_frames * 1000.0 / info.sample_rate
    return Adaptation(
        audio=audio,
        sample_rate=info.sample_rate,
        original_duration_ms=info.duration_ms,
        adapted_duration_ms=adapted_ms,
        requested_ratio=requested_ratio,
        applied_ratio=applied_ratio,
        boundary_clamped=clamped,
    )


def silence_like(sample_rate: int, duration_ms: int,
                 channels: int) -> np.ndarray:
    frames = int(round(sample_rate * duration_ms / 1000.0))
    return np.zeros((frames, channels), dtype=np.float64)
