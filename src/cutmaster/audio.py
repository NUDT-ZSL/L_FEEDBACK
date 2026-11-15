"""Audio I/O, stereo downmixing and deterministic sample-rate conversion."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf

from .errors import err
from .timing import resampled_frame_count

DEFAULT_BLOCK_FRAMES = 65_536
RESAMPLE_CHUNK_FRAMES = 16_384
RESAMPLE_TAPS = 64
RESAMPLE_KAISER_BETA = 10.0
RAW_DTYPE = np.dtype("<f4")


def sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inspect_source(path: Path, declared_rate: int, declared_channels: int) -> sf._SoundFileInfo:
    if not path.is_file():
        raise err(
            "E_MISSING_TRACK",
            "audio track does not exist",
            exit_code=3,
            path=str(path),
        )
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError, ValueError) as exc:
        raise err(
            "E_UNREADABLE_TRACK",
            "audio track cannot be decoded",
            exit_code=3,
            path=str(path),
            reason=str(exc),
        ) from exc
    if info.samplerate != declared_rate:
        raise err(
            "E_METADATA_MISMATCH",
            "declared sample rate does not match the audio file",
            exit_code=3,
            path=str(path),
            declared=declared_rate,
            actual=info.samplerate,
        )
    if info.channels != declared_channels:
        raise err(
            "E_METADATA_MISMATCH",
            "declared channel count does not match the audio file",
            exit_code=3,
            path=str(path),
            declared=declared_channels,
            actual=info.channels,
        )
    if info.frames <= 0:
        raise err(
            "E_EMPTY_TRACK",
            "audio track contains no frames",
            exit_code=3,
            path=str(path),
        )
    if info.channels < 1:
        raise err(
            "E_BAD_CHANNELS",
            "audio track has no channels",
            exit_code=3,
            path=str(path),
        )
    return info


def downmix_to_stereo(data: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    channels = data.shape[1]
    if channels == 1:
        return np.repeat(data.astype(np.float32, copy=False), 2, axis=1)
    if channels == 2:
        return data.astype(np.float32, copy=False)

    left = data[:, 0].astype(np.float32, copy=True)
    right = data[:, 1].astype(np.float32, copy=True)
    if channels == 3:
        left += data[:, 2] * np.float32(0.70710678)
        right += data[:, 2] * np.float32(0.70710678)
    elif channels == 4:
        left += data[:, 2] * np.float32(0.70710678)
        right += data[:, 3] * np.float32(0.70710678)
    elif channels in (6, 8):
        left += data[:, 2] * np.float32(0.70710678)
        right += data[:, 2] * np.float32(0.70710678)
        left += data[:, 3] * np.float32(0.5)
        right += data[:, 3] * np.float32(0.5)
        left += data[:, 4] * np.float32(0.70710678)
        right += data[:, 5] * np.float32(0.70710678)
        if channels == 8:
            left += data[:, 6] * np.float32(0.70710678)
            right += data[:, 7] * np.float32(0.70710678)
    else:
        # Deterministic fallback: keep L/R and pan remaining channels to both
        # sides with a 1/sqrt(N) conservative summing law.
        remaining = data[:, 2:]
        if remaining.shape[1]:
            gain = np.float32(1.0 / math.sqrt(float(remaining.shape[1])))
            mixed = remaining.sum(axis=1, dtype=np.float32) * gain
            left += mixed
            right += mixed
    return np.column_stack((left, right)).astype(np.float32, copy=False)


def read_stereo_float32(path: Path) -> np.ndarray:
    try:
        data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    except (RuntimeError, OSError, ValueError) as exc:
        raise err(
            "E_UNREADABLE_TRACK",
            "audio track cannot be decoded",
            exit_code=3,
            path=str(path),
            reason=str(exc),
        ) from exc
    return downmix_to_stereo(data)


def _resample_phase_table(
    source_rate: int, target_rate: int
) -> tuple[int, int, np.ndarray | None]:
    gcd = math.gcd(source_rate, target_rate)
    step = source_rate // gcd
    period = target_rate // gcd
    if period > 16_384:
        return step, period, None

    offsets = np.arange(-(RESAMPLE_TAPS // 2) + 1, RESAMPLE_TAPS // 2 + 1, dtype=np.int64)
    window = np.kaiser(RESAMPLE_TAPS, RESAMPLE_KAISER_BETA).astype(np.float64)
    cutoff = min(1.0, target_rate / source_rate)
    fractions = np.arange(period, dtype=np.float64) / float(period)
    distances = fractions[:, None] - offsets[None, :]
    weights = (
        cutoff
        * np.sinc(cutoff * distances)
        * window[None, :]
    )
    sums = weights.sum(axis=1, keepdims=True)
    weights /= sums
    return step, period, weights.astype(np.float64)


def resample_stereo(
    data: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if source_rate == target_rate:
        return data.astype(np.float32, copy=False)
    source_frames = int(data.shape[0])
    target_frames = resampled_frame_count(source_frames, source_rate, target_rate)
    if target_frames <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    step, period, phase_table = _resample_phase_table(source_rate, target_rate)
    half = RESAMPLE_TAPS // 2
    padded = np.pad(
        data.astype(np.float32, copy=False),
        ((half, RESAMPLE_TAPS + 1), (0, 0)),
        mode="edge",
    )
    output = np.empty((target_frames, 2), dtype=np.float32)

    for start in range(0, target_frames, RESAMPLE_CHUNK_FRAMES):
        stop = min(start + RESAMPLE_CHUNK_FRAMES, target_frames)
        positions = np.arange(start, stop, dtype=np.int64) * step
        base = positions // period
        remainder = positions % period
        offsets = np.arange(
            -(RESAMPLE_TAPS // 2) + 1, RESAMPLE_TAPS // 2 + 1, dtype=np.int64
        )
        indices = base[:, None] + offsets[None, :] + half
        values = padded[indices]

        if phase_table is not None:
            weights = phase_table[remainder]
        else:
            fractions = remainder.astype(np.float64) / float(period)
            distances = fractions[:, None] - offsets[None, :]
            window = np.kaiser(RESAMPLE_TAPS, RESAMPLE_KAISER_BETA).astype(np.float64)
            cutoff = min(1.0, target_rate / source_rate)
            weights = cutoff * np.sinc(cutoff * distances) * window[None, :]
            weights /= weights.sum(axis=1, keepdims=True)
        output[start:stop] = np.einsum(
            "bt,btc->bc", weights, values, optimize=True
        ).astype(np.float32)
    return output


def prepare_stereo_track(
    path: Path, source_rate: int, target_rate: int
) -> np.ndarray:
    data = read_stereo_float32(path)
    return resample_stereo(data, source_rate, target_rate)


def iter_raw_blocks(
    path: Path,
    frames: int,
    *,
    block_frames: int = DEFAULT_BLOCK_FRAMES,
) -> Iterator[tuple[int, np.ndarray]]:
    with path.open("rb") as handle:
        frame = 0
        while frame < frames:
            count = min(block_frames, frames - frame)
            raw = handle.read(count * 2 * RAW_DTYPE.itemsize)
            expected = count * 2 * RAW_DTYPE.itemsize
            if len(raw) != expected:
                raise err(
                    "E_STATE_CORRUPT",
                    "cached track ended unexpectedly",
                    exit_code=7,
                    path=str(path),
                    expected_bytes=expected,
                    actual_bytes=len(raw),
                )
            block = np.frombuffer(raw, dtype=RAW_DTYPE).reshape(count, 2).copy()
            yield frame, block
            frame += count


def write_raw_track(path: Path, data: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(data, dtype=RAW_DTYPE)
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for start in range(0, contiguous.shape[0], DEFAULT_BLOCK_FRAMES):
            block = contiguous[start : start + DEFAULT_BLOCK_FRAMES]
            raw = block.tobytes(order="C")
            digest.update(raw)
            handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return digest.hexdigest()


def write_float_wav(path: Path, data: np.ndarray, sample_rate: int) -> None:
    with sf.SoundFile(
        str(path),
        mode="w",
        samplerate=sample_rate,
        channels=2,
        format="WAV",
        subtype="FLOAT",
    ) as handle:
        for start in range(0, data.shape[0], DEFAULT_BLOCK_FRAMES):
            handle.write(data[start : start + DEFAULT_BLOCK_FRAMES])


def copy_float_wav_scaled(
    source: Path,
    destination: Path,
    sample_rate: int,
    gain: float,
) -> tuple[float, float]:
    peak = 0.0
    energy = 0.0
    frames = 0
    with sf.SoundFile(str(source), mode="r") as reader:
        with sf.SoundFile(
            str(destination),
            mode="w",
            samplerate=sample_rate,
            channels=2,
            format="WAV",
            subtype="FLOAT",
        ) as writer:
            while True:
                block = reader.read(DEFAULT_BLOCK_FRAMES, dtype="float32", always_2d=True)
                if block.shape[0] == 0:
                    break
                block *= np.float32(gain)
                peak = max(peak, float(np.max(np.abs(block))))
                energy += float(np.sum(block.astype(np.float64) ** 2))
                frames += int(block.shape[0])
                writer.write(block)
    return peak, energy


def write_master_wav(
    source_float_wav: Path,
    temporary_master: Path,
    sample_rate: int,
    subtype: str,
) -> dict[str, object]:
    peak = 0.0
    energy = 0.0
    frames = 0
    digest = hashlib.sha256()
    with sf.SoundFile(str(source_float_wav), mode="r") as reader:
        with sf.SoundFile(
            str(temporary_master),
            mode="w",
            samplerate=sample_rate,
            channels=2,
            format="WAV",
            subtype=subtype,
        ) as writer:
            while True:
                block = reader.read(DEFAULT_BLOCK_FRAMES, dtype="float32", always_2d=True)
                if block.shape[0] == 0:
                    break
                peak = max(peak, float(np.max(np.abs(block))))
                energy += float(np.sum(block.astype(np.float64) ** 2))
                frames += int(block.shape[0])
                writer.write(block)
            writer.flush()
    with temporary_master.open("rb") as handle:
        while True:
            chunk = handle.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return {
        "frames": frames,
        "sample_peak_dbfs": linear_to_db(peak),
        "rms_dbfs": linear_to_db(math.sqrt(energy / max(1, frames * 2))),
        "sha256": digest.hexdigest(),
        "bytes": temporary_master.stat().st_size,
    }


def linear_to_db(value: float) -> float:
    if value <= 0.0:
        return -math.inf
    return 20.0 * math.log10(value)


def db_to_linear(value: float) -> float:
    return 10.0 ** (value / 20.0)
