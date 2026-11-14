from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np


MASTER_FILENAME = "master.wav"
LOG_FILENAME = "processing_log.json"

CHANNEL_MASKS: Mapping[str, int] = {
    "FL": 0x0001,
    "FR": 0x0002,
    "FC": 0x0004,
    "LFE": 0x0008,
    "BL": 0x0010,
    "BR": 0x0020,
    "BC": 0x0100,
    "SL": 0x0200,
    "SR": 0x0400,
}

PCM_SUBFORMAT_GUID = (
    b"\x01\x00\x00\x00"
    b"\x00\x00"
    b"\x10\x00"
    b"\x80\x00\x00\xaa\x00\x38\x9b\x71"
)


class OutputError(RuntimeError):
    pass


def _channel_mask(labels: Sequence[str]) -> int:
    mask = 0
    for label in labels:
        mask |= CHANNEL_MASKS[label.upper()]
    return mask


def _quantize(samples: np.ndarray, bit_depth: int) -> np.ndarray:
    scale = float(1 << (bit_depth - 1))
    scaled = samples * scale
    rounded = np.where(
        scaled >= 0.0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5)
    )
    rounded = np.clip(rounded, -scale, scale - 1.0)
    integer = rounded.astype(np.int64)
    if bit_depth == 8:
        return (integer + 128).astype(np.uint8)
    if bit_depth == 16:
        return integer.astype("<i2")
    if bit_depth == 32:
        return integer.astype("<i4")

    # PCM_24 has no native NumPy dtype. Store the low three bytes of each
    # little-endian signed 32-bit word.
    as_int32 = integer.astype("<i4")
    byte_view = as_int32.view(np.uint8).reshape(-1, 4)
    return byte_view[:, :3].copy()


def _wavex_header(
    channels: int,
    sample_rate: int,
    bit_depth: int,
    data_size: int,
    labels: Sequence[str],
) -> bytes:
    block_align = channels * bit_depth // 8
    byte_rate = sample_rate * block_align
    mask = _channel_mask(labels)
    fmt_payload = (
        struct.pack(
            "<HHIIHHHHI",
            0xFFFE,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bit_depth,
            22,
            bit_depth,
            mask,
        )
        + PCM_SUBFORMAT_GUID
    )
    padded_data_size = data_size + (data_size & 1)
    riff_size = 4 + 8 + len(fmt_payload) + 8 + padded_data_size
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt_payload))
        + fmt_payload
        + b"data"
        + struct.pack("<I", data_size)
    )


def write_master_wav(
    path: Path,
    samples: np.ndarray,
    sample_rate: int,
    bit_depth: int,
    channel_labels: Sequence[str],
) -> None:
    if samples.ndim != 2 or samples.shape[1] != len(channel_labels):
        raise OutputError("master shape does not match channel layout")
    if samples.shape[0] != 0 and not np.all(np.isfinite(samples)):
        raise OutputError("master contains NaN or infinite samples")

    frame_count, channels = samples.shape
    data_size = frame_count * channels * bit_depth // 8
    try:
        with path.open("wb") as handle:
            handle.write(
                _wavex_header(
                    channels,
                    sample_rate,
                    bit_depth,
                    data_size,
                    channel_labels,
                )
            )
            handle.write(_quantize(samples, bit_depth).tobytes(order="C"))
            if data_size & 1:
                handle.write(b"\x00")
    except (OSError, MemoryError, ValueError) as exc:
        raise OutputError("cannot write WAV master: {}".format(exc))


def write_json_log(path: Path, log: Dict[str, object]) -> None:
    try:
        payload = json.dumps(
            log,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise OutputError("cannot serialize processing log: {}".format(exc))
    try:
        path.write_text(payload + "\n", encoding="utf-8")
    except OSError as exc:
        raise OutputError("cannot write processing log: {}".format(exc))


def publish_outputs(
    output_dir: Path,
    master: np.ndarray,
    log: Dict[str, object],
    sample_rate: int,
    bit_depth: int,
    channel_labels: Sequence[str],
    write_master: bool,
) -> None:
    """Publish master and log atomically from the caller's perspective."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OutputError("cannot create output directory: {}".format(exc))
    if not output_dir.is_dir():
        raise OutputError("output path is not a directory")

    master_path = output_dir / MASTER_FILENAME
    log_path = output_dir / LOG_FILENAME
    master_temp = output_dir / ".master.wav.tmp"
    log_temp = output_dir / ".processing_log.json.tmp"
    temporary_paths: List[Path] = [master_temp, log_temp]

    try:
        write_json_log(log_temp, log)
        if write_master:
            write_master_wav(
                master_temp,
                master,
                sample_rate,
                bit_depth,
                channel_labels,
            )

        if write_master:
            os.replace(str(master_temp), str(master_path))
        elif master_path.exists():
            master_path.unlink()
        os.replace(str(log_temp), str(log_path))
    except OSError as exc:
        raise OutputError("cannot publish outputs: {}".format(exc))
    finally:
        for temporary_path in temporary_paths:
            try:
                if temporary_path.exists():
                    temporary_path.unlink()
            except OSError:
                pass
