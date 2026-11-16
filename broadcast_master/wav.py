from __future__ import annotations

import struct
from pathlib import Path

from .errors import BusinessError, ProcessingError
from .models import WavInfo


_PCM = 1
_IEEE_FLOAT = 3
_EXTENSIBLE = 0xFFFE


def _read_exact(handle, size: int, description: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ProcessingError(3002, f"truncated WAV while reading {description}")
    return data


def probe_wav(path: str | Path, *, validate_alignment: bool = False) -> WavInfo:
    wav_path = Path(path)
    try:
        file_size = wav_path.stat().st_size
        handle = wav_path.open("rb")
    except FileNotFoundError as exc:
        raise ProcessingError(3001, f"language WAV not found: {wav_path}") from exc
    except OSError as exc:
        raise ProcessingError(3001, f"cannot open language WAV: {wav_path}: {exc}") from exc

    with handle:
        riff = _read_exact(handle, 12, "RIFF header")
        if riff[0:4] != b"RIFF" or riff[8:12] != b"WAVE":
            raise ProcessingError(3002, f"not a RIFF/WAVE file: {wav_path}")

        audio_format: int | None = None
        channels: int | None = None
        sample_rate: int | None = None
        bits_per_sample: int | None = None
        block_align: int | None = None
        data_offset: int | None = None
        data_size: int | None = None

        while handle.tell() + 8 <= file_size:
            header = _read_exact(handle, 8, "chunk header")
            chunk_id = header[0:4]
            chunk_size = struct.unpack("<I", header[4:8])[0]
            payload_start = handle.tell()
            if chunk_id == b"fmt ":
                payload = _read_exact(handle, chunk_size, "fmt chunk")
                if chunk_size < 16:
                    raise ProcessingError(3002, f"invalid fmt chunk in {wav_path}")
                (
                    tag,
                    channels,
                    sample_rate,
                    _byte_rate,
                    block_align,
                    bits_per_sample,
                ) = struct.unpack("<HHIIHH", payload[:16])
                audio_format = tag
                if tag == _EXTENSIBLE:
                    if chunk_size < 40:
                        raise ProcessingError(3002, f"truncated WAVE_FORMAT_EXTENSIBLE header in {wav_path}")
                    subformat_tag = struct.unpack("<H", payload[24:26])[0]
                    audio_format = subformat_tag
            elif chunk_id == b"data":
                data_offset = payload_start
                data_size = chunk_size
                handle.seek(payload_start + chunk_size + (chunk_size & 1))
            else:
                handle.seek(payload_start + chunk_size + (chunk_size & 1))

        if (
            audio_format is None
            or channels is None
            or sample_rate is None
            or bits_per_sample is None
            or block_align is None
            or data_offset is None
            or data_size is None
        ):
            raise ProcessingError(3002, f"WAV is missing fmt or data chunk: {wav_path}")
        if audio_format not in (_PCM, _IEEE_FLOAT):
            raise ProcessingError(3002, f"unsupported WAV codec tag {audio_format}: {wav_path}")
        if channels not in (1, 2):
            raise ProcessingError(3002, f"unsupported WAV channel count {channels}: {wav_path}")
        if sample_rate <= 0 or bits_per_sample <= 0 or block_align <= 0:
            raise ProcessingError(3002, f"invalid WAV format fields: {wav_path}")

        expected_block_align = channels * ((bits_per_sample + 7) // 8)
        if block_align != expected_block_align:
            raise ProcessingError(
                3002,
                f"WAV block_align={block_align} does not match channels={channels} "
                f"and bits_per_sample={bits_per_sample}: {wav_path}",
            )
        block_aligned = data_size % expected_block_align == 0
        if validate_alignment and not block_aligned:
            raise ProcessingError(
                3003,
                f"WAV data size {data_size} is not a multiple of block align "
                f"{expected_block_align}: {wav_path}",
                {"path": str(wav_path), "data_size": data_size, "block_align": expected_block_align},
            )

        truncated = file_size < data_offset + data_size
        if validate_alignment and truncated:
            raise ProcessingError(
                3001,
                f"truncated WAV data chunk: {wav_path}",
                {
                    "path": str(wav_path),
                    "declared_data_size": data_size,
                    "available_data_size": max(0, file_size - data_offset),
                },
            )

        return WavInfo(
            path=wav_path,
            audio_format=audio_format,
            channels=channels,
            sample_rate=sample_rate,
            bits_per_sample=bits_per_sample,
            block_align=expected_block_align,
            data_offset=data_offset,
            data_size=data_size,
            data_frames=data_size // expected_block_align,
            block_aligned=block_aligned,
            file_size=file_size,
            truncated=truncated,
        )


def validate_wav_against_manifest(
    info: WavInfo, sample_rate: int, channels: int, bit_depth: int
) -> None:
    if info.sample_rate != sample_rate:
        raise BusinessError(
            1005,
            f"language WAV sample_rate={info.sample_rate} is inconsistent with manifest {sample_rate}: {info.path}",
        )
    if info.channels != channels:
        raise BusinessError(
            1005,
            f"language WAV channels={info.channels} is inconsistent with manifest {channels}: {info.path}",
        )
    if info.bits_per_sample != bit_depth:
        raise BusinessError(
            1006,
            f"language WAV bit_depth={info.bits_per_sample} is inconsistent with manifest {bit_depth}: {info.path}",
        )
