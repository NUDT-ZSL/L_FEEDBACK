from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .constants import MIN_DISTANCE_METERS, RING_CHANNELS
from .container import ContainerReader
from .errors import RENDER_INVALID, RENDER_IO, PipelineError


@dataclass(frozen=True)
class RenderStats:
    samples: int
    channels: int
    checksum_seconds: float
    render_seconds: float
    write_seconds: float
    total_seconds: float


def vbap_gains(
    azimuth_deg: float,
    elevation_deg: float = 0.0,
    channels: int = RING_CHANNELS,
) -> np.ndarray:
    if channels < 2:
        raise PipelineError(RENDER_INVALID, "VBAP requires at least two channels")
    if not math.isfinite(azimuth_deg) or not math.isfinite(elevation_deg):
        raise PipelineError(RENDER_INVALID, "object coordinates must be finite")
    spacing = 360.0 / channels
    azimuth = float(azimuth_deg) % 360.0
    sector = int(math.floor(azimuth / spacing))
    left_angle = sector * spacing
    right_angle = left_angle + spacing
    denominator = math.sin(math.radians(right_angle - left_angle))
    left = math.sin(math.radians(right_angle - azimuth)) / denominator
    right = math.sin(math.radians(azimuth - left_angle)) / denominator
    gains = np.zeros(channels, dtype=np.float64)
    gains[sector % channels] = max(0.0, left)
    gains[(sector + 1) % channels] = max(0.0, right)
    norm = math.sqrt(float(np.dot(gains, gains)))
    if norm > 0:
        gains /= norm
    elevation = min(90.0, max(-90.0, float(elevation_deg)))
    if abs(elevation) >= 90.0:
        gains.fill(0.0)
        return gains
    elevation_scale = max(0.0, math.cos(math.radians(elevation)))
    gains *= elevation_scale
    return gains


def distance_attenuation(
    distance: float, minimum_distance: float = MIN_DISTANCE_METERS
) -> float:
    if not math.isfinite(distance):
        raise PipelineError(RENDER_INVALID, "object distance must be finite")
    if not math.isfinite(minimum_distance) or minimum_distance <= 0:
        raise PipelineError(RENDER_INVALID, "minimum distance must be positive")
    effective = max(abs(float(distance)), float(minimum_distance))
    return 1.0 / (effective * effective)


def spatial_gain_matrix(
    metadata: np.ndarray,
    channels: int = RING_CHANNELS,
    minimum_distance: float = MIN_DISTANCE_METERS,
) -> np.ndarray:
    values = np.asarray(metadata, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 5:
        raise PipelineError(RENDER_INVALID, "metadata must have five fields per object")
    matrix = np.empty((values.shape[0], channels), dtype=np.float64)
    for index, (azimuth, elevation, distance, gain, _size) in enumerate(values):
        if not np.all(np.isfinite(values[index])):
            raise PipelineError(RENDER_INVALID, "metadata must be finite")
        pan = vbap_gains(float(azimuth), float(elevation), channels)
        matrix[index] = pan * (distance_attenuation(float(distance), minimum_distance) * float(gain))
    return matrix


def render_block(
    object_audio: np.ndarray,
    metadata: np.ndarray,
    minimum_distance: float = MIN_DISTANCE_METERS,
) -> np.ndarray:
    audio = np.asarray(object_audio, dtype=np.float64)
    values = np.asarray(metadata, dtype=np.float64)
    if audio.ndim != 2:
        raise PipelineError(RENDER_INVALID, "object audio must be a 2D array")
    if values.ndim != 2 or values.shape[1] < 5:
        raise PipelineError(RENDER_INVALID, "metadata must have five fields per object")
    if audio.shape[1] != values.shape[0]:
        raise PipelineError(RENDER_INVALID, "audio channels and metadata objects differ")
    matrix = spatial_gain_matrix(values, RING_CHANNELS, minimum_distance)
    with np.errstate(over="ignore", invalid="ignore"):
        rendered = audio @ matrix
    if not np.all(np.isfinite(rendered)):
        raise PipelineError(RENDER_INVALID, "rendered audio contains non-finite samples")
    return rendered.astype(np.float32, copy=False)


def render_container(
    reader: ContainerReader,
    output_path: str | Path,
    verify_checksum: bool = True,
    minimum_distance: float = MIN_DISTANCE_METERS,
    progress: Callable[[], None] | None = None,
) -> RenderStats:
    if not math.isfinite(minimum_distance) or minimum_distance <= 0:
        raise PipelineError(RENDER_INVALID, "minimum distance must be positive")
    started = time.perf_counter()
    checksum_seconds = 0.0
    if verify_checksum:
        checksum_start = time.perf_counter()
        reader.verify_payload()
        checksum_seconds = time.perf_counter() - checksum_start

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    render_seconds = 0.0
    write_seconds = 0.0
    try:
        with sf.SoundFile(
            os.fspath(temporary),
            mode="w",
            samplerate=reader.header.sample_rate,
            channels=RING_CHANNELS,
            format="WAV",
            subtype="FLOAT",
        ) as output:
            for _block_index, metadata, object_audio in reader.iter_blocks():
                render_start = time.perf_counter()
                rendered = render_block(object_audio, metadata, minimum_distance)
                if rendered.shape[0] != object_audio.shape[0]:
                    raise PipelineError(RENDER_INVALID, "rendered block length mismatch")
                render_seconds += time.perf_counter() - render_start
                write_start = time.perf_counter()
                output.write(rendered)
                write_seconds += time.perf_counter() - write_start
                if progress is not None:
                    progress()
        os.replace(temporary, target)
    except PipelineError:
        temporary.unlink(missing_ok=True)
        raise
    except (OSError, RuntimeError) as exc:
        temporary.unlink(missing_ok=True)
        raise PipelineError(RENDER_IO, "cannot write rendered WAV") from exc
    finally:
        temporary.unlink(missing_ok=True)
    total_seconds = time.perf_counter() - started
    return RenderStats(
        samples=reader.header.sample_count,
        channels=RING_CHANNELS,
        checksum_seconds=checksum_seconds,
        render_seconds=render_seconds,
        write_seconds=write_seconds,
        total_seconds=total_seconds,
    )
