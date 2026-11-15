"""WAV header inspection and deterministic multichannel PCM writing."""

from __future__ import annotations

import os
import struct
from typing import Any, Dict, Optional, Tuple

import numpy as np
import soundfile as sf

from .errors import MasterError
from .layouts import default_roles_for_count, format_mask, mask_for_roles


WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

PCM_SUBFORMAT_GUID = bytes.fromhex("0100000000001000800000aa00389b71")
FLOAT_SUBFORMAT_GUID = bytes.fromhex("0300000000001000800000aa00389b71")


def _read_exact(handle: Any, count: int) -> bytes:
    data = handle.read(count)
    if len(data) != count:
        raise MasterError("E_AUDIO_READ", "Truncated WAV file")
    return data


def inspect_wav(path: str) -> Dict[str, Any]:
    """Read RIFF/fmt metadata, including WAVE_FORMAT_EXTENSIBLE mask fields."""

    if not os.path.isfile(path):
        raise MasterError(
            "E_INPUT_MISSING", "Audio input file does not exist", {"file": path}
        )
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            header = _read_exact(handle, 12)
            if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise MasterError(
                    "E_AUDIO_READ",
                    "Only little-endian RIFF/WAVE files are supported",
                    {"file": path},
                )
            fmt: Optional[Dict[str, Any]] = None
            data_bytes: Optional[int] = None
            while handle.tell() + 8 <= size:
                chunk_header = _read_exact(handle, 8)
                chunk_id = chunk_header[:4]
                chunk_size = struct.unpack("<I", chunk_header[4:])[0]
                chunk_start = handle.tell()
                if chunk_start + chunk_size > size:
                    raise MasterError(
                        "E_AUDIO_READ",
                        "WAV chunk extends beyond the end of the file",
                        {"file": path, "chunk": chunk_id.decode("latin-1", "replace")},
                    )
                if chunk_id == b"fmt ":
                    payload = _read_exact(handle, chunk_size)
                    fmt = _parse_fmt(payload, path)
                elif chunk_id == b"data":
                    data_bytes = chunk_size
                    handle.seek(chunk_size, os.SEEK_CUR)
                else:
                    handle.seek(chunk_size, os.SEEK_CUR)
                if chunk_size & 1:
                    handle.seek(1, os.SEEK_CUR)
            if fmt is None or data_bytes is None:
                raise MasterError(
                    "E_AUDIO_READ",
                    "WAV file is missing fmt or data chunk",
                    {"file": path},
                )
    except MasterError:
        raise
    except OSError as exc:
        raise MasterError(
            "E_AUDIO_READ", "Unable to inspect WAV file", {"file": path, "error": str(exc)}
        )

    fmt.update(
        {
            "file": path,
            "data_bytes": data_bytes,
            "data_frames": data_bytes // max(1, fmt["block_align"]),
            "file_size": size,
        }
    )
    if data_bytes % fmt["block_align"] != 0:
        raise MasterError(
            "E_AUDIO_READ",
            "WAV data chunk is not an integer number of sample frames",
            {
                "file": path,
                "data_bytes": data_bytes,
                "block_align": fmt["block_align"],
            },
        )
    return fmt


def _parse_fmt(payload: bytes, path: str) -> Dict[str, Any]:
    if len(payload) < 16:
        raise MasterError(
            "E_AUDIO_READ", "WAV fmt chunk is too short", {"file": path}
        )
    (
        format_tag,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    ) = struct.unpack("<HHIIHH", payload[:16])
    channel_mask = 0
    valid_bits = bits_per_sample
    subformat_tag: Optional[int] = None
    cb_size = 0
    if format_tag == WAVE_FORMAT_EXTENSIBLE:
        if len(payload) < 40:
            raise MasterError(
                "E_AUDIO_READ",
                "WAVE_FORMAT_EXTENSIBLE fmt chunk is too short",
                {"file": path},
            )
        cb_size, valid_bits, channel_mask = struct.unpack("<HHI", payload[16:24])
        guid = payload[24:40]
        if len(guid) != 16:
            raise MasterError(
                "E_AUDIO_READ", "Invalid WAVE_FORMAT_EXTENSIBLE subformat", {"file": path}
            )
        if guid == PCM_SUBFORMAT_GUID:
            subformat_tag = WAVE_FORMAT_PCM
        elif guid == FLOAT_SUBFORMAT_GUID:
            subformat_tag = WAVE_FORMAT_IEEE_FLOAT
        else:
            subformat_tag = struct.unpack("<I", guid[:4])[0]
    if channels < 1:
        raise MasterError(
            "E_CHANNEL_MISMATCH", "WAV declares zero channels", {"file": path}
        )
    if sample_rate < 1:
        raise MasterError(
            "E_AUDIO_READ", "WAV declares an invalid sample rate", {"file": path}
        )
    expected_block_align = channels * ((bits_per_sample + 7) // 8)
    if block_align != expected_block_align:
        raise MasterError(
            "E_AUDIO_READ",
            "WAV block alignment is inconsistent with channels and bit depth",
            {
                "file": path,
                "block_align": block_align,
                "expected": expected_block_align,
            },
        )
    effective_tag = format_tag if format_tag != WAVE_FORMAT_EXTENSIBLE else subformat_tag
    if effective_tag not in (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT):
        raise MasterError(
            "E_AUDIO_READ",
            "Unsupported WAV sample format",
            {"file": path, "format_tag": format_tag, "subformat_tag": subformat_tag},
        )
    return {
        "format_tag": format_tag,
        "effective_format_tag": effective_tag,
        "format_name": "WAVE_FORMAT_EXTENSIBLE"
        if format_tag == WAVE_FORMAT_EXTENSIBLE
        else "WAVE_FORMAT_PCM"
        if format_tag == WAVE_FORMAT_PCM
        else "WAVE_FORMAT_IEEE_FLOAT",
        "channels": channels,
        "sample_rate": sample_rate,
        "byte_rate": byte_rate,
        "block_align": block_align,
        "bits_per_sample": bits_per_sample,
        "valid_bits_per_sample": valid_bits,
        "channel_mask": channel_mask,
        "channel_mask_hex": format_mask(channel_mask),
        "cb_size": cb_size,
        "subformat_tag": subformat_tag,
    }


def resolve_source_roles(
    info: Dict[str, Any],
    configured_layout: Optional[list],
    channel_map: Optional[Any],
) -> Tuple[Any, list, str]:
    """Resolve source channel roles with explicit config taking precedence."""

    channels = int(info["channels"])
    if configured_layout is not None:
        if len(configured_layout) != channels:
            raise MasterError(
                "E_CHANNEL_MISMATCH",
                "input_layout length does not match WAV channels",
                {
                    "file": info["file"],
                    "file_channels": channels,
                    "input_layout_channels": len(configured_layout),
                },
            )
        roles = list(configured_layout)
        source = "config.input_layout"
    else:
        mask = int(info["channel_mask"])
        if info["format_tag"] == WAVE_FORMAT_EXTENSIBLE and mask:
            from .layouts import roles_from_mask

            roles = roles_from_mask(mask, channels)
            source = "wav_extensible_channel_mask"
        else:
            roles, source = default_roles_for_count(channels)

    if channel_map is not None:
        if isinstance(channel_map, list):
            if len(channel_map) != channels:
                raise MasterError(
                    "E_CHANNEL_MISMATCH",
                    "channel_map length does not match WAV channels",
                    {
                        "file": info["file"],
                        "file_channels": channels,
                        "channel_map_channels": len(channel_map),
                    },
                )
            final_roles = list(channel_map)
        else:
            final_roles = [None] * channels
            for role, index in channel_map.items():
                if index >= channels:
                    raise MasterError(
                        "E_CHANNEL_MISMATCH",
                        "channel_map source index is outside the WAV channel range",
                        {
                            "file": info["file"],
                            "role": role,
                            "source_index": index,
                            "file_channels": channels,
                        },
                    )
                final_roles[index] = role
            if any(role is None for role in final_roles):
                missing = [i for i, role in enumerate(final_roles) if role is None]
                raise MasterError(
                    "E_CHANNEL_MISMATCH",
                    "Every source channel needs a role when channel_map is an object",
                    {"file": info["file"], "unmapped_source_channels": missing},
                )
        source = "config.channel_map"
        roles = final_roles

    if len(roles) != channels:
        raise MasterError(
            "E_CHANNEL_MISMATCH",
            "Resolved channel role count does not match WAV channels",
            {"file": info["file"], "file_channels": channels, "roles": roles},
        )
    return mask_for_roles([role for role in roles if role is not None]), roles, source


def read_audio(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    info = inspect_wav(path)
    try:
        data, sample_rate = sf.read(
            path, dtype="float64", always_2d=True, frames=-1
        )
    except Exception as exc:
        raise MasterError(
            "E_AUDIO_READ",
            "Unable to decode WAV audio",
            {"file": path, "error": str(exc)},
        )
    if sample_rate != info["sample_rate"]:
        raise MasterError(
            "E_AUDIO_READ",
            "WAV header and decoder disagree on sample rate",
            {
                "file": path,
                "header_sample_rate": info["sample_rate"],
                "decoder_sample_rate": sample_rate,
            },
        )
    if data.shape[1] != info["channels"]:
        raise MasterError(
            "E_CHANNEL_MISMATCH",
            "WAV header and decoder disagree on channel count",
            {
                "file": path,
                "header_channels": info["channels"],
                "decoded_channels": int(data.shape[1]),
            },
        )
    if data.shape[0] != info["data_frames"]:
        raise MasterError(
            "E_AUDIO_READ",
            "Decoded WAV frame count does not match its data chunk",
            {
                "file": path,
                "header_frames": info["data_frames"],
                "decoded_frames": int(data.shape[0]),
            },
        )
    if not np.all(np.isfinite(data)):
        raise MasterError(
            "E_AUDIO_READ", "WAV contains non-finite samples", {"file": path}
        )
    return np.ascontiguousarray(data), info


def quantize_pcm(data: np.ndarray, bits_per_sample: int) -> np.ndarray:
    if bits_per_sample not in (16, 24):
        raise ValueError("Only 16-bit and 24-bit PCM output is supported")
    scale = float(1 << (bits_per_sample - 1))
    minimum = -(1 << (bits_per_sample - 1))
    maximum = (1 << (bits_per_sample - 1)) - 1
    integers = np.rint(np.clip(data, -1.0, 1.0 - 1.0 / scale) * scale)
    integers = np.clip(integers, minimum, maximum).astype(np.int32)
    return integers.astype(np.float64) / scale


def _pcm_bytes(data: np.ndarray, bits_per_sample: int) -> bytes:
    scale = float(1 << (bits_per_sample - 1))
    minimum = -(1 << (bits_per_sample - 1))
    maximum = (1 << (bits_per_sample - 1)) - 1
    integers = np.rint(np.clip(data, -1.0, 1.0 - 1.0 / scale) * scale)
    integers = np.clip(integers, minimum, maximum).astype("<i4")
    flat = np.ascontiguousarray(integers).reshape(-1)
    if bits_per_sample == 16:
        return flat.astype("<i2").tobytes()
    raw = flat.view(np.uint8).reshape(-1, 4)[:, :3]
    return raw.tobytes()


def write_wav(
    path: str,
    data: np.ndarray,
    sample_rate: int,
    channel_mask: int,
    bits_per_sample: int = 24,
    extensible: bool = True,
) -> None:
    """Write deterministic interleaved PCM WAV (optionally EXTENSIBLE)."""

    if data.ndim != 2:
        raise ValueError("Expected a (frames, channels) array")
    frames, channels = data.shape
    if frames < 0 or channels < 1:
        raise ValueError("Invalid WAV dimensions")
    sample_bytes = bits_per_sample // 8
    block_align = channels * sample_bytes
    byte_rate = sample_rate * block_align
    payload = _pcm_bytes(data, bits_per_sample)
    data_size = len(payload)
    pad = b"\x00" if data_size & 1 else b""

    if extensible:
        fmt_payload = struct.pack(
            "<HHIIHHHHI",
            WAVE_FORMAT_EXTENSIBLE,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            22,
            bits_per_sample,
            int(channel_mask) & 0xFFFFFFFF,
        ) + PCM_SUBFORMAT_GUID
    else:
        fmt_payload = struct.pack(
            "<HHIIHH",
            WAVE_FORMAT_PCM,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        )
    fmt_chunk = b"fmt " + struct.pack("<I", len(fmt_payload)) + fmt_payload
    data_chunk = b"data" + struct.pack("<I", data_size) + payload + pad
    body = b"WAVE" + fmt_chunk + data_chunk
    riff = b"RIFF" + struct.pack("<I", len(body)) + body
    try:
        with open(path, "wb") as handle:
            handle.write(riff)
    except OSError as exc:
        raise MasterError(
            "E_OUTPUT_WRITE", "Unable to write WAV file", {"file": path, "error": str(exc)}
        )

