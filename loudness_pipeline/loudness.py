"""ITU-R BS.1770-4 gated integrated loudness measurement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .channels import channel_weights
from .kweighting import filter_kweighting


ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0
LOUDNESS_OFFSET = -0.691
DEFAULT_WINDOW_SECONDS = 0.4
DEFAULT_HOP_SECONDS = 0.1


def _loudness_from_energy(energy: float) -> Optional[float]:
    if not math.isfinite(energy) or energy <= 0.0:
        return None
    return LOUDNESS_OFFSET + 10.0 * math.log10(energy)


@dataclass(frozen=True)
class LoudnessBlock:
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    loudness_lufs: Optional[float]
    passes_absolute_gate: bool
    passes_relative_gate: bool
    is_partial: bool


@dataclass(frozen=True)
class LoudnessMeasurement:
    integrated_lufs: float
    blocks: Tuple[LoudnessBlock, ...]
    absolute_gate_lufs: float = ABSOLUTE_GATE_LUFS
    relative_gate_lufs: Optional[float] = None
    gated_block_count: int = 0


def _weighted_frame_power(
    weighted_audio: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(channel_weights(weighted_audio.shape[1]), dtype=np.float64)
    if weights.shape[0] != weighted_audio.shape[1]:
        raise ValueError("channel weight count does not match audio channel count")
    power = np.sum(weighted_audio * weighted_audio * weights[None, :], axis=1)
    cumulative = np.empty(power.shape[0] + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(power, dtype=np.float64, out=cumulative[1:])
    return cumulative, weights


def _block_energies(
    cumulative_power: np.ndarray,
    frame_count: int,
    window_frames: int,
    hop_frames: int,
    allow_partial: bool,
) -> List[Tuple[int, int, bool]]:
    if frame_count <= 0:
        return []
    full_blocks: List[Tuple[int, int, bool]] = []
    last_full_start = frame_count - window_frames
    if last_full_start >= 0:
        for start in range(0, last_full_start + 1, hop_frames):
            full_blocks.append((start, start + window_frames, False))
    elif allow_partial:
        full_blocks.append((0, frame_count, True))
    return full_blocks


def measure_loudness(
    samples: np.ndarray,
    sample_rate: int,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
    allow_partial: bool = True,
) -> LoudnessMeasurement:
    """Measure gated integrated loudness and the underlying 400 ms blocks.

    A 400 ms block is used with a 100 ms hop (75 percent overlap), matching the
    gating block definition in BS.1770. Absolute gating at -70 LUFS is applied
    first; the relative gate is the absolute-gated integrated loudness minus
    10 LU. A recording with no energy above the absolute gate raises E_SILENT
    at the pipeline layer.
    """

    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim != 2:
        raise ValueError("samples must be a two-dimensional array")
    if not (0.0 < window_seconds <= 10.0):
        raise ValueError("window_seconds must be in (0, 10]")
    if not (0.0 < hop_seconds <= window_seconds):
        raise ValueError("hop_seconds must be in (0, window_seconds]")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    window_frames = max(1, int(round(window_seconds * sample_rate)))
    hop_frames = max(1, int(round(hop_seconds * sample_rate)))
    frame_count = audio.shape[0]
    if frame_count == 0:
        return LoudnessMeasurement(
            integrated_lufs=float("-inf"),
            blocks=(),
            gated_block_count=0,
        )

    weighted_audio = filter_kweighting(audio, sample_rate)
    cumulative_power, _ = _weighted_frame_power(weighted_audio)
    block_specs = _block_energies(
        cumulative_power,
        frame_count,
        window_frames,
        hop_frames,
        allow_partial,
    )

    raw_energies: List[float] = []
    raw_loudness: List[Optional[float]] = []
    for start, end, _ in block_specs:
        length = end - start
        energy = float((cumulative_power[end] - cumulative_power[start]) / length)
        raw_energies.append(energy)
        raw_loudness.append(_loudness_from_energy(energy))

    absolute_indices = [
        index
        for index, value in enumerate(raw_loudness)
        if value is not None and value >= ABSOLUTE_GATE_LUFS
    ]
    if not absolute_indices:
        blocks = tuple(
            LoudnessBlock(
                start_seconds=start / float(sample_rate),
                end_seconds=end / float(sample_rate),
                duration_seconds=(end - start) / float(sample_rate),
                loudness_lufs=raw_loudness[index],
                passes_absolute_gate=False,
                passes_relative_gate=False,
                is_partial=partial,
            )
            for index, (start, end, partial) in enumerate(block_specs)
        )
        return LoudnessMeasurement(
            integrated_lufs=float("-inf"),
            blocks=blocks,
            relative_gate_lufs=None,
            gated_block_count=0,
        )

    absolute_mean_energy = float(np.mean([raw_energies[i] for i in absolute_indices]))
    absolute_loudness = _loudness_from_energy(absolute_mean_energy)
    assert absolute_loudness is not None
    relative_gate_lufs = absolute_loudness + RELATIVE_GATE_LU

    relative_indices = [
        index
        for index in absolute_indices
        if raw_loudness[index] is not None
        and raw_loudness[index] >= relative_gate_lufs
    ]
    relative_mean_energy = float(np.mean([raw_energies[i] for i in relative_indices]))
    integrated = _loudness_from_energy(relative_mean_energy)
    assert integrated is not None

    blocks = tuple(
        LoudnessBlock(
            start_seconds=start / float(sample_rate),
            end_seconds=end / float(sample_rate),
            duration_seconds=(end - start) / float(sample_rate),
            loudness_lufs=raw_loudness[index],
            passes_absolute_gate=index in absolute_indices,
            passes_relative_gate=index in relative_indices,
            is_partial=partial,
        )
        for index, (start, end, partial) in enumerate(block_specs)
    )
    return LoudnessMeasurement(
        integrated_lufs=integrated,
        blocks=blocks,
        relative_gate_lufs=relative_gate_lufs,
        gated_block_count=len(relative_indices),
    )


def sample_peak(samples: np.ndarray) -> float:
    """Return the maximum absolute sample value (0 dBFS equals 1.0)."""

    audio = np.asarray(samples, dtype=np.float64)
    if audio.size == 0:
        return 0.0
    return float(np.max(np.abs(audio)))


def peak_dbfs(samples: np.ndarray) -> float:
    peak = sample_peak(samples)
    if peak <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(peak)

