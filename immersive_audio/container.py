from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .audio import ObjectBusInfo, SceneAudioAnalysis
from .constants import (
    CONTAINER_MAGIC,
    CONTAINER_VERSION,
    DEFAULT_CHUNK_SAMPLES,
    MAX_OBJECTS,
    METADATA_RECORD_SIZE,
)
from .errors import (
    CONTAINER_CHECKSUM,
    CONTAINER_COMMENT,
    CONTAINER_LAYOUT,
    CONTAINER_TRUNCATED,
    CONTAINER_MAGIC as ERROR_CONTAINER_MAGIC,
    CONTAINER_VERSION as ERROR_CONTAINER_VERSION,
    PipelineError,
)
from .scene import Scene
from .timeline import block_count, iter_metadata_blocks

HEADER_STRUCT = struct.Struct("<8sHHII IHH QQQQQQQQ 32s".replace(" ", ""))
HEADER_SIZE = HEADER_STRUCT.size


@dataclass(frozen=True)
class ContainerHeader:
    magic: bytes
    version: int
    header_size: int
    flags: int
    sample_rate: int
    frame_length: int
    channel_count: int
    object_count: int
    sample_count: int
    frame_count: int
    metadata_offset: int
    metadata_size: int
    audio_offset: int
    audio_size: int
    comment_offset: int
    comment_size: int
    payload_sha256: bytes


@dataclass(frozen=True)
class ContainerWriteStats:
    metadata_seconds: float
    comment_seconds: float
    audio_seconds: float
    bytes_written: int
    sha256: str


def _pack_header(header: ContainerHeader) -> bytes:
    raw = HEADER_STRUCT.pack(
        header.magic,
        header.version,
        header.header_size,
        header.flags,
        header.sample_rate,
        header.frame_length,
        header.channel_count,
        header.object_count,
        header.sample_count,
        header.frame_count,
        header.metadata_offset,
        header.metadata_size,
        header.audio_offset,
        header.audio_size,
        header.comment_offset,
        header.comment_size,
        header.payload_sha256,
    )
    if len(raw) != HEADER_SIZE:
        raise AssertionError("container header size mismatch")
    return raw


def _read_header_bytes(raw: bytes) -> ContainerHeader:
    if len(raw) < HEADER_SIZE:
        raise PipelineError(CONTAINER_TRUNCATED, "container header is truncated")
    values = HEADER_STRUCT.unpack_from(raw)
    header = ContainerHeader(*values)
    if header.magic != CONTAINER_MAGIC:
        raise PipelineError(ERROR_CONTAINER_MAGIC, "invalid container magic")
    if header.version != CONTAINER_VERSION:
        raise PipelineError(
            ERROR_CONTAINER_VERSION,
            f"unsupported container version: {header.version}",
        )
    return header


def _validate_header(header: ContainerHeader, file_size: int) -> None:
    if header.header_size != HEADER_SIZE:
        raise PipelineError(CONTAINER_LAYOUT, "invalid header size")
    if header.flags != 0:
        raise PipelineError(CONTAINER_LAYOUT, "unsupported header flags")
    if header.sample_rate <= 0 or header.frame_length <= 0:
        raise PipelineError(CONTAINER_LAYOUT, "invalid sample rate or frame length")
    if not 1 <= header.object_count <= MAX_OBJECTS:
        raise PipelineError(CONTAINER_LAYOUT, "invalid object count")
    if header.channel_count != header.object_count:
        raise PipelineError(CONTAINER_LAYOUT, "channel count must equal object count")
    expected_frames = block_count(header.sample_count, header.frame_length)
    if header.frame_count != expected_frames:
        raise PipelineError(CONTAINER_LAYOUT, "metadata frame count is inconsistent")
    expected_metadata = (
        header.frame_count * header.channel_count * METADATA_RECORD_SIZE
    )
    if header.metadata_size != expected_metadata:
        raise PipelineError(CONTAINER_LAYOUT, "invalid metadata block size")
    expected_audio = header.sample_count * header.channel_count * 4
    if header.audio_size != expected_audio:
        raise PipelineError(CONTAINER_LAYOUT, "invalid audio block size")
    if header.metadata_offset != HEADER_SIZE:
        raise PipelineError(CONTAINER_LAYOUT, "invalid metadata offset")
    if header.comment_offset != header.metadata_offset + header.metadata_size:
        raise PipelineError(CONTAINER_LAYOUT, "invalid comment offset")
    if header.audio_offset != header.comment_offset + header.comment_size:
        raise PipelineError(CONTAINER_LAYOUT, "invalid audio offset")
    expected_file_size = header.audio_offset + header.audio_size
    if file_size < expected_file_size:
        raise PipelineError(CONTAINER_TRUNCATED, "container payload is truncated")
    if file_size != expected_file_size:
        raise PipelineError(CONTAINER_LAYOUT, "container has trailing or extra data")


def build_container_comments(
    scene: Scene, analysis: SceneAudioAnalysis
) -> dict[str, object]:
    objects: list[dict[str, object]] = []
    for obj, obj_analysis in zip(scene.objects, analysis.objects):
        clips = [
            {
                "frames": item.target_frames,
                "loops": item.clip.loops,
                "path": item.clip.display_path,
                "sha256": item.sha256,
                "start": str(item.clip.start),
            }
            for item in obj_analysis.clips
        ]
        objects.append(
            {
                "clips": clips,
                "id": obj.object_id,
                "index": obj.index,
            }
        )
    return {
        "format": CONTAINER_MAGIC.decode("ascii"),
        "objects": objects,
        "version": CONTAINER_VERSION,
    }


def encode_comments(comments: dict[str, object]) -> bytes:
    return json.dumps(
        comments,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def write_container(
    output_path: str | Path,
    scene: Scene,
    analysis: SceneAudioAnalysis,
    buses: tuple[ObjectBusInfo, ...],
    comments: dict[str, object],
    chunk_samples: int = DEFAULT_CHUNK_SAMPLES,
    progress: Callable[[], None] | None = None,
) -> ContainerWriteStats:
    import time

    if len(buses) != scene.object_count:
        raise PipelineError(CONTAINER_LAYOUT, "object bus count does not match scene")
    for obj, bus in zip(scene.objects, buses):
        if obj.index != bus.object_index or bus.total_frames != analysis.total_frames:
            raise PipelineError(CONTAINER_LAYOUT, "object bus metadata mismatch")
        if not bus.path.is_file() or bus.path.stat().st_size != analysis.total_frames * 4:
            raise PipelineError(CONTAINER_LAYOUT, "object bus file has an invalid size")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    comment_bytes = encode_comments(comments)
    frame_count = block_count(analysis.total_frames, scene.frame_length)
    metadata_size = frame_count * scene.object_count * METADATA_RECORD_SIZE
    comment_offset = HEADER_SIZE + metadata_size
    audio_offset = comment_offset + len(comment_bytes)
    audio_size = analysis.total_frames * scene.object_count * 4
    placeholder = ContainerHeader(
        magic=CONTAINER_MAGIC,
        version=CONTAINER_VERSION,
        header_size=HEADER_SIZE,
        flags=0,
        sample_rate=scene.sample_rate,
        frame_length=scene.frame_length,
        channel_count=scene.object_count,
        object_count=scene.object_count,
        sample_count=analysis.total_frames,
        frame_count=frame_count,
        metadata_offset=HEADER_SIZE,
        metadata_size=metadata_size,
        audio_offset=audio_offset,
        audio_size=audio_size,
        comment_offset=comment_offset,
        comment_size=len(comment_bytes),
        payload_sha256=b"\x00" * 32,
    )

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    metadata_seconds = 0.0
    comment_seconds = 0.0
    audio_seconds = 0.0
    bytes_written = 0
    handles: list = []
    try:
        with temporary.open("w+b") as output:
            output.write(_pack_header(placeholder))
            start = time.perf_counter()
            for _, block in iter_metadata_blocks(scene, analysis.total_frames):
                raw = np.ascontiguousarray(block, dtype="<f4").tobytes()
                output.write(raw)
                digest.update(raw)
                bytes_written += len(raw)
                if progress is not None:
                    progress()
            metadata_seconds = time.perf_counter() - start

            start = time.perf_counter()
            output.write(comment_bytes)
            digest.update(comment_bytes)
            bytes_written += len(comment_bytes)
            comment_seconds = time.perf_counter() - start

            start = time.perf_counter()
            for bus in buses:
                handles.append(bus.path.open("rb"))
            chunk_frames = max(1, chunk_samples // max(1, scene.object_count))
            for start_frame in range(0, analysis.total_frames, chunk_frames):
                count = min(chunk_frames, analysis.total_frames - start_frame)
                interleaved = np.empty(
                    (count, scene.object_count), dtype="<f4"
                )
                for channel, handle in enumerate(handles):
                    raw = handle.read(count * 4)
                    if len(raw) != count * 4:
                        raise PipelineError(
                            CONTAINER_TRUNCATED, "object bus was truncated while writing"
                        )
                    interleaved[:, channel] = np.frombuffer(raw, dtype="<f4")
                raw = interleaved.tobytes(order="C")
                output.write(raw)
                digest.update(raw)
                bytes_written += len(raw)
                if progress is not None:
                    progress()
            audio_seconds = time.perf_counter() - start
            payload_digest = digest.digest()
            final_header = ContainerHeader(
                **{
                    **placeholder.__dict__,
                    "payload_sha256": payload_digest,
                }
            )
            output.seek(0)
            output.write(_pack_header(final_header))
            output.flush()
        os.replace(temporary, target)
    except PipelineError:
        for handle in handles:
            handle.close()
        temporary.unlink(missing_ok=True)
        raise
    except OSError as exc:
        for handle in handles:
            handle.close()
        temporary.unlink(missing_ok=True)
        raise PipelineError(CONTAINER_LAYOUT, "container write failed") from exc
    finally:
        for handle in handles:
            if not handle.closed:
                handle.close()
        temporary.unlink(missing_ok=True)
    return ContainerWriteStats(
        metadata_seconds=metadata_seconds,
        comment_seconds=comment_seconds,
        audio_seconds=audio_seconds,
        bytes_written=bytes_written,
        sha256=payload_digest.hex(),
    )


class ContainerReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise PipelineError(CONTAINER_TRUNCATED, f"container does not exist: {self.path}")
        try:
            self.file_size = self.path.stat().st_size
            with self.path.open("rb") as handle:
                raw = handle.read(HEADER_SIZE)
        except OSError as exc:
            raise PipelineError(CONTAINER_TRUNCATED, "cannot read container") from exc
        self.header = _read_header_bytes(raw)
        _validate_header(self.header, self.file_size)
        self.comments = self._read_comments()

    def _read_comments(self) -> dict[str, object]:
        if self.header.comment_size > 64 * 1024 * 1024:
            raise PipelineError(CONTAINER_COMMENT, "comment block is too large")
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.header.comment_offset)
                raw = handle.read(self.header.comment_size)
            comments = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PipelineError(CONTAINER_COMMENT, "container comments are invalid") from exc
        if not isinstance(comments, dict):
            raise PipelineError(CONTAINER_COMMENT, "container comments must be an object")
        return comments

    def verify_payload(self) -> str:
        digest = hashlib.sha256()
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.header.metadata_offset)
                remaining = (
                    self.header.metadata_size
                    + self.header.comment_size
                    + self.header.audio_size
                )
                while remaining:
                    raw = handle.read(min(DEFAULT_CHUNK_SAMPLES, remaining))
                    if not raw:
                        raise PipelineError(
                            CONTAINER_TRUNCATED, "container payload is truncated"
                        )
                    digest.update(raw)
                    remaining -= len(raw)
        except OSError as exc:
            raise PipelineError(CONTAINER_TRUNCATED, "cannot read container payload") from exc
        actual = digest.digest()
        if actual != self.header.payload_sha256:
            raise PipelineError(CONTAINER_CHECKSUM, "container payload checksum mismatch")
        return actual.hex()

    def iter_metadata_blocks(self) -> Iterator[np.ndarray]:
        block_size = self.header.object_count * METADATA_RECORD_SIZE
        try:
            with self.path.open("rb") as handle:
                for block_index in range(self.header.frame_count):
                    handle.seek(
                        self.header.metadata_offset + block_index * block_size
                    )
                    raw = handle.read(block_size)
                    if len(raw) != block_size:
                        raise PipelineError(
                            CONTAINER_TRUNCATED, "metadata payload is truncated"
                        )
                    yield np.frombuffer(raw, dtype="<f4").reshape(
                        self.header.object_count, 5
                    ).copy()
        except OSError as exc:
            raise PipelineError(CONTAINER_TRUNCATED, "cannot read metadata") from exc

    def iter_audio_blocks(self) -> Iterator[np.ndarray]:
        sample = self.header.frame_length
        channels = self.header.channel_count
        try:
            with self.path.open("rb") as handle:
                for block_index in range(self.header.frame_count):
                    first_sample = block_index * sample
                    count = min(sample, self.header.sample_count - first_sample)
                    expected = count * channels * 4
                    handle.seek(
                        self.header.audio_offset + first_sample * channels * 4
                    )
                    raw = handle.read(expected)
                    if len(raw) != expected:
                        raise PipelineError(
                            CONTAINER_TRUNCATED, "audio payload is truncated"
                        )
                    yield np.frombuffer(raw, dtype="<f4").reshape(
                        count, channels
                    ).copy()
        except OSError as exc:
            raise PipelineError(CONTAINER_TRUNCATED, "cannot read audio") from exc

    def iter_blocks(self) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
        metadata_iterator = self.iter_metadata_blocks()
        audio_iterator = self.iter_audio_blocks()
        for block_index, (metadata, audio) in enumerate(
            zip(metadata_iterator, audio_iterator)
        ):
            yield block_index, metadata, audio

    def read_metadata(self) -> np.ndarray:
        blocks = list(self.iter_metadata_blocks())
        if not blocks:
            return np.empty((0, self.header.object_count, 5), dtype=np.float32)
        return np.stack(blocks)

    def read_audio(self) -> np.ndarray:
        blocks = list(self.iter_audio_blocks())
        if not blocks:
            return np.empty((0, self.header.channel_count), dtype=np.float32)
        return np.concatenate(blocks, axis=0)

    def read_all(self) -> tuple[np.ndarray, np.ndarray]:
        return self.read_metadata(), self.read_audio()
