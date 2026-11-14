"""Small deterministic, streaming-safe helpers."""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path

import numpy as np


UINT64_MASK = (1 << 64) - 1


def round_half_up(value: float) -> int:
    """Round a non-negative value with the conventional half-up rule."""

    if value < 0:
        return -int(math.floor(-value + 0.5))
    return int(math.floor(value + 0.5))


def finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _stream_salt(name: str) -> np.uint64:
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return np.uint64(int.from_bytes(digest, "little", signed=False))


def _mix64(values: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer on a uint64 NumPy array (wrapping arithmetic)."""

    z = values + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def deterministic_noise(
    seed: int,
    stream: str,
    start_sample: int,
    count: int,
    channels: int = 2,
) -> np.ndarray:
    """Return white noise addressed by global sample index.

    The value for a given (seed, stream, sample index, channel) tuple does not
    depend on block size, call order or process state.
    """

    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return np.empty((0, channels), dtype=np.float64)

    indices = np.arange(start_sample, start_sample + count, dtype=np.uint64)
    stream_key = np.uint64(seed & UINT64_MASK) ^ _stream_salt(stream)
    output = np.empty((count, channels), dtype=np.float64)
    scale = 2.0 / 9007199254740992.0

    with np.errstate(over="ignore", invalid="ignore"):
        for channel in range(channels):
            mixed = _mix64(
                indices
                ^ stream_key
                ^ np.uint64(
                    ((channel + 1) * 0xD1B54A32D192ED03) & UINT64_MASK
                )
            )
            # Top 53 bits give an exactly representable double in [-1, 1).
            output[:, channel] = (
                (mixed >> np.uint64(11)).astype(np.float64) * scale - 1.0
            )
    return output


def make_temp_file(target: Path, kind: str) -> Path:
    """Create a closed, uniquely named temp file next to ``target``."""

    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.{kind}.",
        suffix=".tmp",
        dir=str(parent),
    )
    os.close(fd)
    return Path(raw_path)


def remove_quietly(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def soft_limit(samples: np.ndarray, threshold: float) -> np.ndarray:
    """C1-continuous soft limiter with an asymptote at +/-1."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("soft limiter threshold must be in (0, 1)")
    magnitude = np.abs(samples)
    over = magnitude > threshold
    if not np.any(over):
        return samples.copy()
    span = 1.0 - threshold
    limited = threshold + span * np.tanh((magnitude - threshold) / span)
    return np.where(over, np.copysign(limited, samples), samples)


def pcm16_bytes(samples: np.ndarray) -> bytes:
    """Convert float stereo samples in [-1, 1] to little-endian PCM16."""

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.rint(clipped * 32767.0).astype("<i2", copy=False)
    return np.ascontiguousarray(pcm).tobytes()


def linear_frame_interp(
    values: np.ndarray,
    sample_start: int,
    count: int,
    samples_per_frame: float,
) -> np.ndarray:
    """Linearly resample per-frame values at global audio sample coordinates.

    Frame 0 is anchored to sample 0. Values beyond the final frame are held.
    This implementation uses only local indices, so block boundaries do not
    alter the interpolation formula.
    """

    if count == 0:
        return np.empty(0, dtype=np.float64)
    frame_count = len(values)
    if frame_count == 0:
        raise ValueError("cannot interpolate an empty frame array")
    positions = (sample_start + np.arange(count, dtype=np.float64)) / samples_per_frame
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, frame_count - 1)
    lower = np.clip(lower, 0, frame_count - 1)
    fraction = positions - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction

