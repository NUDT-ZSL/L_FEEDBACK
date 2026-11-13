"""RIFF/WAVE parsing and WAVE_FORMAT_EXTENSIBLE packaging.

The reader intentionally supports only integer PCM 16/24/32-bit and IEEE float
32-bit.  The writer emits a standards-compliant extensible PCM file with cue
points, cue labels, LIST/INFO metadata, and an embedded canonical playback
timeline JSON chunk.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .errors import WavError
from .resources import ResourceGovernor
from .util import canonical_json_bytes

RIFF_ID = b"RIFF"
WAVE_ID = b"WAVE"
FMT_ID = b"fmt "
DATA_ID = b"data"
CUE_ID = b"cue "
LIST_ID = b"LIST"
JSON_ID = b"JSON"

FMT_PCM = 0x0001
FMT_IEEE_FLOAT = 0x0003
FMT_EXTENSIBLE = 0xFFFE

PCM_SUBFORMAT = bytes.fromhex("0100000000001000800000aa00389b71")
FLOAT_SUBFORMAT = bytes.fromhex("0300000000001000800000aa00389b71")

SPEAKER_MASKS: dict[str, int] = {
    "FL": 0x00000001,
    "FR": 0x00000002,
    "FC": 0x00000004,
    "LFE": 0x00000008,
    "BL": 0x00000010,
    "BR": 0x00000020,
    "FLC": 0x00000040,
    "FRC": 0x00000080,
    "BC": 0x00000100,
    "SL": 0x00000200,
    "SR": 0x00000400,
    "TC": 0x00000800,
    "TFL": 0x00001000,
    "TFC": 0x00002000,
    "TFR": 0x00004000,
    "TBL": 0x00008000,
    "TBC": 0x00010000,
    "TBR": 0x00020000,
}


@dataclass(frozen=True)
class WavInfo:
    path: Path
    format_tag: int
    audio_format: str
    channels: int
    sample_rate: int
    bits_per_sample: int
    valid_bits: int
    block_align: int
    channel_mask: int
    data_offset: int
    data_size: int
    frames: int

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate

    @property
    def bytes_per_sample(self) -> int:
        return (self.bits_per_sample + 7) // 8


def _unpack_u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def _validated_chunk_bounds(
    file_size: int, offset: int, size: int, *, what: str
) -> tuple[int, int]:
    if offset < 12 or size < 0:
        raise WavError(f"invalid {what} chunk bounds")
    end = offset + size
    if end > file_size:
        raise WavError(
            f"{what} chunk is truncated: declares {size} bytes at offset {offset}, "
            f"file has {max(0, file_size - offset)}"
        )
    return offset, end


def parse_wav(path: Path | str) -> WavInfo:
    path = Path(path)
    try:
        file_size = path.stat().st_size
        with open(path, "rb") as handle:
            header = handle.read(12)
            if len(header) < 12:
                raise WavError("file is shorter than a RIFF/WAVE header")
            if header[0:4] != RIFF_ID or header[8:12] != WAVE_ID:
                raise WavError("missing RIFF/WAVE signature")
            fmt_payload: bytes | None = None
            data_offset = -1
            data_size = -1
            offset = 12
            while offset + 8 <= file_size:
                handle.seek(offset)
                chunk_header = handle.read(8)
                if len(chunk_header) != 8:
                    raise WavError(f"truncated chunk header at offset {offset}")
                chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
                payload_offset = offset + 8
                _validated_chunk_bounds(
                    file_size, payload_offset, chunk_size, what=chunk_id.decode("latin1")
                )
                if chunk_id == FMT_ID:
                    if chunk_size < 16:
                        raise WavError("fmt chunk is too short")
                    handle.seek(payload_offset)
                    fmt_payload = handle.read(chunk_size)
                elif chunk_id == DATA_ID:
                    data_offset = payload_offset
                    data_size = chunk_size
                offset = payload_offset + chunk_size + (chunk_size & 1)

            if fmt_payload is None:
                raise WavError("missing fmt chunk")
            if data_offset < 0:
                raise WavError("missing data chunk")
    except FileNotFoundError as exc:
        raise WavError(f"file not found: {path}") from exc
    except OSError as exc:
        raise WavError(f"cannot read WAV header: {exc}") from exc

    if len(fmt_payload) < 16:
        raise WavError("invalid fmt chunk")
    (
        format_tag,
        channels,
        sample_rate,
        avg_bytes_per_sec,
        block_align,
        bits_per_sample,
    ) = struct.unpack_from("<HHIIHH", fmt_payload, 0)

    valid_bits = bits_per_sample
    channel_mask = 0
    subtype = format_tag
    if format_tag == FMT_EXTENSIBLE:
        if len(fmt_payload) < 40:
            raise WavError("extensible fmt chunk is shorter than 40 bytes")
        cb_size, valid_bits, channel_mask = struct.unpack_from("<HHI", fmt_payload, 16)
        if cb_size < 22:
            raise WavError("extensible fmt chunk has invalid cbSize")
        subtype_bytes = fmt_payload[24:40]
        if subtype_bytes == PCM_SUBFORMAT:
            subtype = FMT_PCM
        elif subtype_bytes == FLOAT_SUBFORMAT:
            subtype = FMT_IEEE_FLOAT
        else:
            raise WavError("unsupported WAVE_FORMAT_EXTENSIBLE subformat")

    if channels < 1 or channels > 64:
        raise WavError(f"unsupported channel count: {channels}")
    if sample_rate < 8000 or sample_rate > 768000:
        raise WavError(f"unsupported sample rate: {sample_rate}")
    if subtype == FMT_IEEE_FLOAT:
        if bits_per_sample != 32:
            raise WavError("only 32-bit IEEE float WAV input is supported")
        audio_format = "float"
    elif subtype == FMT_PCM:
        if bits_per_sample not in (16, 24, 32):
            raise WavError(
                f"unsupported PCM bit depth {bits_per_sample}; expected 16, 24, or 32"
            )
        audio_format = "pcm"
    else:
        raise WavError(f"unsupported WAV format tag 0x{format_tag:04x}")

    container_bytes = (bits_per_sample + 7) // 8
    expected_align = channels * container_bytes
    if block_align != expected_align:
        raise WavError(
            f"invalid block alignment {block_align}; expected {expected_align} "
            f"for {channels} channel(s) at {bits_per_sample} bits"
        )
    if avg_bytes_per_sec != sample_rate * block_align:
        raise WavError(
            f"invalid average byte rate {avg_bytes_per_sec}; "
            f"expected {sample_rate * block_align}"
        )
    if valid_bits < 1 or valid_bits > bits_per_sample:
        raise WavError(f"invalid valid-bits value: {valid_bits}")
    if data_size % block_align:
        raise WavError(
            f"data chunk has {data_size % block_align} trailing byte(s); "
            "not a whole sample frame"
        )

    return WavInfo(
        path=path,
        format_tag=format_tag,
        audio_format=audio_format,
        channels=channels,
        sample_rate=sample_rate,
        bits_per_sample=bits_per_sample,
        valid_bits=valid_bits,
        block_align=block_align,
        channel_mask=channel_mask,
        data_offset=data_offset,
        data_size=data_size,
        frames=data_size // block_align,
    )


def _decode_pcm(raw: bytes, info: WavInfo, frame_count: int) -> np.ndarray:
    channels = info.channels
    bits = info.bits_per_sample
    if bits == 16:
        values = np.frombuffer(raw, dtype="<i2").reshape(frame_count, channels)
        return values.astype(np.float64) / 32768.0
    if bits == 32 and info.audio_format == "pcm":
        values = np.frombuffer(raw, dtype="<i4").reshape(frame_count, channels)
        return values.astype(np.float64) / 2147483648.0
    if bits == 32 and info.audio_format == "float":
        values = np.frombuffer(raw, dtype="<f4").reshape(frame_count, channels)
        audio = values.astype(np.float64)
        return np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    if bits == 24:
        triplets = np.frombuffer(raw, dtype=np.uint8).reshape(frame_count, channels, 3)
        values = (
            triplets[..., 0].astype(np.int32)
            | (triplets[..., 1].astype(np.int32) << 8)
            | (triplets[..., 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float64) / 8388608.0
    raise WavError(f"cannot decode {bits}-bit {info.audio_format} audio")


def read_frames(
    info: WavInfo,
    start: int,
    count: int,
    governor: ResourceGovernor | None = None,
) -> tuple[int, np.ndarray]:
    """Read a clamped frame range.

    Returns ``(actual_start, samples)`` where ``samples`` has shape
    ``(actual_count, channels)``.  Requests wholly outside the file return an
    empty array.
    """

    start = max(0, int(start))
    stop = min(info.frames, start + max(0, int(count)))
    if stop <= start:
        return start, np.empty((0, info.channels), dtype=np.float64)
    nbytes = (stop - start) * info.block_align
    governor_context = governor.memory(nbytes) if governor is not None else None
    if governor_context is not None:
        governor_context.__enter__()
    try:
        handle_context = governor.wav_handle() if governor is not None else None
        if handle_context is not None:
            handle_context.__enter__()
        try:
            with open(info.path, "rb") as handle:
                handle.seek(info.data_offset + start * info.block_align)
                raw = handle.read(nbytes)
        finally:
            if handle_context is not None:
                handle_context.__exit__(None, None, None)
    finally:
        if governor_context is not None:
            governor_context.__exit__(None, None, None)
    expected = (stop - start) * info.block_align
    if len(raw) != expected:
        raise WavError(
            f"short read at frame {start}: expected {expected} bytes, got {len(raw)}"
        )
    return start, _decode_pcm(raw, info, stop - start)


def _chunk(fourcc: bytes, payload: bytes) -> bytes:
    """Build one RIFF child chunk, including its even-byte padding."""

    if len(fourcc) != 4:
        raise ValueError("RIFF chunk id must be four bytes")
    if len(payload) > 0xFFFFFFFF:
        raise WavError("RIFF child chunk exceeds the 4 GiB format limit")
    return fourcc + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) & 1 else b"")


def _info_list(metadata: Mapping[str, Any]) -> bytes:
    entries: list[tuple[bytes, str]] = [
        (b"INAM", str(metadata.get("title", ""))),
        (b"IART", str(metadata.get("artist", ""))),
        (b"ICRD", str(metadata.get("date", ""))),
        (b"ISFT", str(metadata.get("software", "EventMix 1.0.0"))),
        (b"ICMT", str(metadata.get("comment", ""))),
    ]
    payload = bytearray(b"INFO")
    for fourcc, text in entries:
        if text:
            encoded = text.encode("utf-8") + b"\x00"
            payload.extend(_chunk(fourcc, encoded))
    return _chunk(LIST_ID, bytes(payload))


def _cue_chunks(cues: Sequence[Mapping[str, Any]]) -> tuple[bytes, bytes]:
    if len(cues) > 0xFFFF:
        raise WavError("WAV cue chunk supports at most 65535 cue points")
    cue_entries = bytearray()
    label_payload = bytearray(b"adtl")
    for cue in cues:
        cue_id = int(cue["wav_cue_id"])
        frame = int(cue["frame"])
        if cue_id < 1 or cue_id > 0xFFFFFFFF:
            raise WavError(f"invalid WAV cue id {cue_id}")
        if frame < 0 or frame > 0xFFFFFFFF:
            raise WavError(f"cue frame {frame} is outside the 32-bit RIFF range")
        cue_entries.extend(
            struct.pack("<II4sIII", cue_id, frame, DATA_ID, 0, 0, frame)
        )
        label = str(cue.get("name") or cue.get("id") or cue_id)
        label_payload.extend(_chunk(b"labl", struct.pack("<I", cue_id) + label.encode("utf-8") + b"\x00"))
    cue_chunk = _chunk(CUE_ID, struct.pack("<I", len(cues)) + bytes(cue_entries))
    label_chunk = _chunk(LIST_ID, bytes(label_payload))
    return cue_chunk, label_chunk


def _fmt_extensible(
    channels: int, sample_rate: int, bits: int, channel_mask: int
) -> bytes:
    block_align = channels * ((bits + 7) // 8)
    payload = struct.pack(
        "<HHIIHHH",
        FMT_EXTENSIBLE,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits,
        22,
    )
    payload += struct.pack("<HI", bits, channel_mask)
    payload += PCM_SUBFORMAT
    return _chunk(FMT_ID, payload)


def float_to_pcm_bytes(audio: np.ndarray, bits: int) -> tuple[bytes, int]:
    """Quantize float audio in [-1, 1] to integer PCM bytes.

    A final integer clamp is present as a last-resort safety net.  The renderer
    keeps samples inside the representable range, so its clamp count is expected
    to remain zero and is returned for reporting.
    """

    audio = np.asarray(audio, dtype=np.float64)
    if bits == 16:
        scale = 32768.0
        lo, hi = -(1 << 15), (1 << 15) - 1
    elif bits == 24:
        scale = 8388608.0
        lo, hi = -(1 << 23), (1 << 23) - 1
    elif bits == 32:
        scale = 2147483648.0
        lo, hi = -(1 << 31), (1 << 31) - 1
    else:
        raise WavError(f"unsupported output bit depth {bits}")

    rounded = np.rint(audio * scale)
    clamp_count = int(np.count_nonzero((rounded < lo) | (rounded > hi)))
    integers = np.clip(rounded, lo, hi).astype(np.int64).ravel()
    if bits == 16:
        return integers.astype("<i2").tobytes(), clamp_count
    if bits == 32:
        return integers.astype("<i4").tobytes(), clamp_count
    unsigned = integers & 0xFFFFFF
    out = np.empty((integers.size, 3), dtype=np.uint8)
    out[:, 0] = unsigned & 0xFF
    out[:, 1] = (unsigned >> 8) & 0xFF
    out[:, 2] = (unsigned >> 16) & 0xFF
    return out.tobytes(), clamp_count


class ExtensibleWavSink:
    """Streaming writer for final program audio.

    The output path should be a temporary path; callers should atomically move it
    into place after ``close`` succeeds.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        sample_rate: int,
        bits: int,
        channels: int,
        channel_mask: int,
        total_frames: int,
        cues: Sequence[Mapping[str, Any]],
        timeline_bytes: bytes,
        info_metadata: Mapping[str, Any],
    ):
        if bits not in (16, 24, 32):
            raise WavError("output bit depth must be 16, 24, or 32")
        if channels < 1 or channels > 64:
            raise WavError("output channel count must be between 1 and 64")
        if channel_mask < 0 or channel_mask > 0x3FFFF:
            raise WavError("invalid WAV channel mask")
        if total_frames < 0:
            raise WavError("negative frame count")
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.bits = bits
        self.channels = channels
        self.channel_mask = channel_mask
        self.total_frames = int(total_frames)
        self.block_align = channels * ((bits + 7) // 8)
        self.data_size = self.total_frames * self.block_align
        if self.data_size > 0xFFFFFFFF:
            raise WavError(
                "program audio exceeds the 4 GiB RIFF data limit; "
                "split the event into shorter program segments"
            )
        self._written_frames = 0
        self._clamp_count = 0
        self._handle = None

        cue_chunk, label_chunk = _cue_chunks(cues)
        metadata_chunks = (
            _fmt_extensible(channels, sample_rate, bits, channel_mask)
            + _info_list(info_metadata)
            + cue_chunk
            + label_chunk
            + _chunk(JSON_ID, timeline_bytes)
        )
        data_header = DATA_ID + struct.pack("<I", self.data_size)
        self._data_offset = 12 + len(metadata_chunks) + 8
        data_padded_size = self.data_size + (self.data_size & 1)
        riff_size = 4 + len(metadata_chunks) + 8 + data_padded_size
        if riff_size > 0xFFFFFFFF:
            raise WavError("RIFF file would exceed the 4 GiB RIFF limit")
        self._riff_size = riff_size
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "wb")
        self._handle.write(RIFF_ID + struct.pack("<I", self._riff_size) + WAVE_ID)
        self._handle.write(metadata_chunks)
        self._handle.write(data_header)

    @property
    def clamp_count(self) -> int:
        return self._clamp_count

    def write_float(self, audio: np.ndarray) -> None:
        if self._handle is None:
            raise WavError("WAV sink is closed")
        audio = np.asarray(audio, dtype=np.float64)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        if audio.shape[1] != self.channels:
            raise WavError(
                f"expected {self.channels} channels, got {audio.shape[1]}"
            )
        if self._written_frames + audio.shape[0] > self.total_frames:
            raise WavError("attempt to write more frames than declared")
        payload, clamp_count = float_to_pcm_bytes(audio, self.bits)
        self._handle.write(payload)
        self._written_frames += audio.shape[0]
        self._clamp_count += clamp_count

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            if self._written_frames != self.total_frames:
                raise WavError(
                    f"wrote {self._written_frames} frames, expected {self.total_frames}"
                )
            if self.data_size & 1:
                self._handle.write(b"\x00")
            self._handle.flush()
            self._handle.seek(4)
            self._handle.write(struct.pack("<I", self._riff_size))
            self._handle.flush()
        finally:
            self._handle.close()
            self._handle = None


def iter_chunks(
    path: Path | str, *, include_data: bool = False
) -> Iterable[tuple[bytes, bytes]]:
    path = Path(path)
    file_size = path.stat().st_size
    with open(path, "rb") as handle:
        header = handle.read(12)
        if len(header) < 12 or header[0:4] != RIFF_ID or header[8:12] != WAVE_ID:
            raise WavError("missing RIFF/WAVE signature")
        offset = 12
        while offset + 8 <= file_size:
            handle.seek(offset)
            raw = handle.read(8)
            if len(raw) != 8:
                break
            fourcc, size = struct.unpack("<4sI", raw)
            payload_offset = offset + 8
            if payload_offset + size > file_size:
                raise WavError(f"truncated {fourcc!r} chunk")
            if fourcc == DATA_ID and not include_data:
                payload = b""
            else:
                handle.seek(payload_offset)
                payload = handle.read(size)
            yield fourcc, payload
            offset = payload_offset + size + (size & 1)


def _parse_info(payload: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    if not payload.startswith(b"INFO"):
        return result
    offset = 4
    while offset + 8 <= len(payload):
        fourcc, size = struct.unpack_from("<4sI", payload, offset)
        start = offset + 8
        end = start + size
        if end > len(payload):
            break
        text = payload[start:end].rstrip(b"\x00").decode("utf-8", errors="replace")
        result[fourcc.decode("latin1")] = text
        offset = end + (size & 1)
    return result


def _parse_cues(payload: bytes) -> list[dict[str, Any]]:
    if len(payload) < 4:
        raise WavError("cue chunk is too short")
    count = _unpack_u32(payload, 0)
    if len(payload) < 4 + count * 24:
        raise WavError("cue chunk entry table is truncated")
    cues: list[dict[str, Any]] = []
    for index in range(count):
        (
            cue_id,
            position,
            data_chunk_id,
            chunk_start,
            block_start,
            sample_offset,
        ) = struct.unpack_from("<II4sIII", payload, 4 + index * 24)
        cues.append(
            {
                "wav_cue_id": cue_id,
                "position": position,
                "data_chunk_id": data_chunk_id.decode("latin1"),
                "chunk_start": chunk_start,
                "block_start": block_start,
                "sample_offset": sample_offset,
            }
        )
    return cues


def _parse_labels(payload: bytes) -> dict[int, str]:
    labels: dict[int, str] = {}
    if not payload.startswith(b"adtl"):
        return labels
    offset = 4
    while offset + 8 <= len(payload):
        fourcc, size = struct.unpack_from("<4sI", payload, offset)
        start = offset + 8
        end = start + size
        if end > len(payload):
            break
        if fourcc == b"labl" and size >= 4:
            cue_id = _unpack_u32(payload, start)
            labels[cue_id] = (
                payload[start + 4 : end].rstrip(b"\x00").decode("utf-8", errors="replace")
            )
        offset = end + (size & 1)
    return labels


def read_wav_metadata(path: Path | str) -> dict[str, Any]:
    """Parse format, cue, labels, INFO metadata, and the embedded JSON chunk."""

    info = parse_wav(path)
    result: dict[str, Any] = {
        "format": info,
        "info": {},
        "cues": [],
        "labels": {},
        "timeline_json": None,
        "timeline_bytes": None,
    }
    for fourcc, payload in iter_chunks(path):
        if fourcc == LIST_ID:
            if payload.startswith(b"INFO"):
                result["info"].update(_parse_info(payload))
            elif payload.startswith(b"adtl"):
                result["labels"].update(_parse_labels(payload))
        elif fourcc == CUE_ID:
            result["cues"] = _parse_cues(payload)
        elif fourcc == JSON_ID:
            result["timeline_bytes"] = payload
            result["timeline_json"] = json.loads(payload.decode("utf-8"))
    return result


def metadata_comment(event: Mapping[str, Any], timeline_digest: str, cue_count: int) -> str:
    return (
        f"event_id={event.get('id', '')}; timeline_sha256={timeline_digest}; "
        f"cue_count={cue_count}; schema=eventmix-playback/v1"
    )


def timeline_payload_bytes(timeline: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(timeline)

