from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import List

from .errors import ConfigError


@dataclass(frozen=True)
class TimelineSegment:
    index: int
    start_sample: int
    end_sample: int

    @property
    def sample_count(self) -> int:
        return self.end_sample - self.start_sample


def plan_timeline(
    total_samples: int,
    sample_rate: int,
    segment_duration: Decimal,
) -> List[TimelineSegment]:
    if total_samples <= 0:
        raise ConfigError("reference input contains no audio samples")
    segment_samples = int(
        (segment_duration * Decimal(sample_rate)).to_integral_value(
            rounding="ROUND_FLOOR"
        )
    )
    if segment_samples <= 0:
        raise ConfigError("segment_duration produces zero samples")

    segments: List[TimelineSegment] = []
    start = 0
    index = 0
    while start < total_samples:
        end = min(start + segment_samples, total_samples)
        segments.append(
            TimelineSegment(index=index, start_sample=start, end_sample=end)
        )
        start = end
        index += 1
    return segments


def relative_shift_frames(
    variant_offset_ms: Decimal,
    variant_start_time: Decimal,
    reference_offset_ms: Decimal,
    reference_start_time: Decimal,
    sample_rate: int,
) -> int:
    seconds = (
        (variant_offset_ms - reference_offset_ms) / Decimal(1000)
        + (variant_start_time - reference_start_time)
    )
    frames = (seconds * Decimal(sample_rate)).to_integral_value(
        rounding=ROUND_HALF_UP
    )
    return int(frames)


def _write_silence(handle: object, frames: int, frame_bytes: int) -> None:
    remaining = frames * frame_bytes
    zero_chunk = b"\x00" * (1024 * 1024)
    while remaining:
        amount = min(remaining, len(zero_chunk))
        handle.write(zero_chunk[:amount])
        remaining -= amount


def _copy_frames(
    source: object,
    destination: object,
    frames: int,
    frame_bytes: int,
) -> None:
    remaining = frames * frame_bytes
    chunk_size = 1024 * 1024
    while remaining:
        chunk = source.read(min(remaining, chunk_size))
        if not chunk:
            raise ConfigError("decoded PCM ended while copying alignment samples")
        destination.write(chunk)
        remaining -= len(chunk)


def align_pcm(
    source_pcm: Path,
    destination_pcm: Path,
    target_frames: int,
    frame_bytes: int,
    shift_frames: int,
) -> None:
    source_size = source_pcm.stat().st_size
    if source_size % frame_bytes != 0:
        raise ConfigError(f"decoded PCM has a partial sample frame: {source_pcm}")
    source_frames = source_size // frame_bytes

    written = 0
    with source_pcm.open("rb") as source, destination_pcm.open("wb") as destination:
        if shift_frames < 0:
            padding = min(-shift_frames, target_frames)
            _write_silence(destination, padding, frame_bytes)
            written += padding
            source_offset = 0
        else:
            source_offset = min(shift_frames, source_frames)

        if written < target_frames and source_offset < source_frames:
            source.seek(source_offset * frame_bytes)
            copy_frames = min(
                target_frames - written, source_frames - source_offset
            )
            _copy_frames(source, destination, copy_frames, frame_bytes)
            written += copy_frames

        if written < target_frames:
            _write_silence(
                destination, target_frames - written, frame_bytes
            )

    expected_size = target_frames * frame_bytes
    actual_size = destination_pcm.stat().st_size
    if actual_size != expected_size:
        raise ConfigError(
            f"aligned PCM size mismatch: expected {expected_size}, got {actual_size}"
        )

