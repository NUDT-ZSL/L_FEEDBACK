"""Streaming WAV sources with deterministic linear resampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .errors import (
    W_DAMAGED_SOURCE,
    W_SAMPLE_RATE_MISMATCH,
    W_SOURCE_UNREADABLE,
)


@dataclass
class SourceInfo:
    sample_rate: int
    frames: int
    channels: int


def probe_source(path: str | Path) -> tuple[SourceInfo | None, str | None]:
    """Open a source just long enough to read its metadata."""
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError, ValueError) as exc:
        return None, f"{W_SOURCE_UNREADABLE}: {exc}"
    if info.frames < 0 or info.samplerate <= 0:
        return None, f"{W_DAMAGED_SOURCE}: cannot read metadata"
    return (
        SourceInfo(info.samplerate, int(info.frames), int(info.channels)),
        None,
    )


class LinearResampler:
    """Streaming fixed-rate linear resampler.

    Source frames are read in fixed chunks and resampled output is generated
    in fixed output quanta. External callers may request any block size; they
    only slice the same deterministic output stream, which guarantees
    bit-identical samples regardless of the configured block size.
    """

    SOURCE_CHUNK = 512
    OUTPUT_QUANTUM = 512

    def __init__(
        self,
        sound: sf.SoundFile,
        channels: int,
        rate: float,
        start_frame: int,
        read_block: int,
    ) -> None:
        del read_block  # I/O quantum is intentionally fixed for determinism
        self._sound = sound
        self._channels = channels
        self._rate = float(rate)
        self._initial_position = float(max(0, start_frame))
        self._position = self._initial_position
        self._emitted = 0
        self._source = np.zeros((0, channels), dtype=np.float64)
        self._source_start = max(0, start_frame)
        self._exhausted = False
        self._ready = np.zeros((0, channels), dtype=np.float64)
        self._ready_cursor = 0

    def pull(self, frames: int) -> np.ndarray:
        parts: list[np.ndarray] = []
        remaining = int(frames)
        while remaining > 0:
            if self._ready_cursor >= self._ready.shape[0]:
                self._produce_quantum()
            if self._ready_cursor >= self._ready.shape[0]:
                break
            take = min(remaining, self._ready.shape[0] - self._ready_cursor)
            parts.append(
                self._ready[
                    self._ready_cursor : self._ready_cursor + take
                ].copy()
            )
            self._ready_cursor += take
            remaining -= take
        if not parts:
            return np.zeros((frames, self._channels), dtype=np.float64)
        output = np.concatenate(parts, axis=0)
        if output.shape[0] < frames:
            output = np.vstack(
                (
                    output,
                    np.zeros(
                        (frames - output.shape[0], self._channels),
                        dtype=np.float64,
                    ),
                )
            )
        if self._ready_cursor >= self._ready.shape[0]:
            self._ready = np.zeros((0, self._channels), dtype=np.float64)
            self._ready_cursor = 0
        return output

    def _produce_quantum(self) -> None:
        wanted = self.OUTPUT_QUANTUM
        produced_frames = 0
        chunks: list[np.ndarray] = []
        while produced_frames < wanted:
            target_position = (
                self._position + (wanted - produced_frames) * self._rate
            )
            self._fill_source(target_position + 1.0)
            local = self._position - self._source_start
            if local >= self._source.shape[0] - 1:
                break
            take = wanted - produced_frames
            indices = local + np.arange(take, dtype=np.float64) * self._rate
            valid = indices <= self._source.shape[0] - 2
            valid_count = int(np.count_nonzero(valid))
            if valid_count == 0:
                break
            idx = indices[:valid_count]
            base = np.floor(idx).astype(np.int64)
            fraction = (idx - base)[:, None]
            interpolated = (
                self._source[base] * (1.0 - fraction)
                + self._source[base + 1] * fraction
            )
            chunks.append(interpolated)
            produced_frames += valid_count
            self._emitted += valid_count
            self._position = self._initial_position + self._emitted * self._rate
            self._trim_source()
        self._ready = (
            np.concatenate(chunks, axis=0)
            if chunks
            else np.zeros((0, self._channels), dtype=np.float64)
        )
        self._ready_cursor = 0

    def _fill_source(self, target_position: float) -> None:
        last_absolute = self._source_start + self._source.shape[0] - 1
        while not self._exhausted and target_position > last_absolute:
            chunk = self._sound.read(
                self.SOURCE_CHUNK, dtype="float64", always_2d=True
            )
            if chunk.shape[0] == 0:
                self._exhausted = True
                if self._source.shape[0]:
                    self._source = np.vstack(
                        (self._source, self._source[-1:])
                    )
                break
            self._source = (
                np.vstack((self._source, chunk))
                if self._source.shape[0]
                else chunk.astype(np.float64, copy=True)
            )
            last_absolute = self._source_start + self._source.shape[0] - 1

    def _trim_source(self) -> None:
        local = self._position - self._source_start
        keep_from = max(0, int(np.floor(local)) - 1)
        if keep_from > 0:
            self._source = self._source[keep_from:]
            self._source_start += keep_from

    def close(self) -> None:
        self._sound.close()

    def working_bytes(self) -> int:
        """Current retained source/output buffer footprint in bytes."""
        return int(self._source.nbytes + self._ready.nbytes)


def open_source(
    path: str | Path, expected_sample_rate: int
) -> tuple[sf.SoundFile | None, SourceInfo | None, str | None]:
    """Open a source, returning a warning when rate mismatches or it is broken."""
    info, error = probe_source(path)
    if error is not None:
        return None, None, error
    assert info is not None
    if info.sample_rate != expected_sample_rate:
        return (
            None,
            info,
            (
                f"{W_SAMPLE_RATE_MISMATCH}: source rate {info.sample_rate} Hz "
                f"!= scene rate {expected_sample_rate} Hz"
            ),
        )
    try:
        sound = sf.SoundFile(str(path), mode="r")
    except (RuntimeError, OSError, ValueError) as exc:
        return None, info, f"{W_DAMAGED_SOURCE}: {exc}"
    return sound, info, None
