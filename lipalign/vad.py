"""Energy and zero-crossing based sentence splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import LipAlignError
from .timeline import Sentence


@dataclass(frozen=True)
class SplitSettings:
    silence_rms: float = 0.006
    silence_zcr: float = 0.080
    frame_ms: float = 20.0
    hop_ms: float = 5.0
    min_silence_ms: float = 80.0
    min_active_ms: float = 60.0
    boundary_guard_ms: float = 2.0

    def validate(self) -> None:
        if not 0.0 < self.silence_rms < 1.0:
            raise LipAlignError(
                "E_PARAMETER", "silence_rms must be in (0, 1)", exit_code=2
            )
        if not 0.0 <= self.silence_zcr <= 1.0:
            raise LipAlignError(
                "E_PARAMETER", "silence_zcr must be in [0, 1]", exit_code=2
            )
        if self.frame_ms <= 0 or self.hop_ms <= 0 or self.hop_ms > self.frame_ms:
            raise LipAlignError(
                "E_PARAMETER",
                "frame_ms and hop_ms must be positive and hop_ms <= frame_ms",
                exit_code=2,
            )
        if self.min_silence_ms < 0 or self.min_active_ms <= 0:
            raise LipAlignError(
                "E_PARAMETER",
                "min_silence_ms must be >= 0 and min_active_ms must be positive",
                exit_code=2,
            )
        if self.boundary_guard_ms < 0:
            raise LipAlignError(
                "E_PARAMETER", "boundary_guard_ms must be >= 0", exit_code=2
            )

    def as_dict(self) -> Dict[str, float]:
        return {
            "silence_rms": self.silence_rms,
            "silence_zcr": self.silence_zcr,
            "frame_ms": self.frame_ms,
            "hop_ms": self.hop_ms,
            "min_silence_ms": self.min_silence_ms,
            "min_active_ms": self.min_active_ms,
            "boundary_guard_ms": self.boundary_guard_ms,
        }


@dataclass
class SplitResult:
    runs: List[Tuple[int, int]]
    segments: List[Optional[Tuple[int, int]]]
    active_run_count: int
    silence_run_count: int


def _frame_rms_and_zcr(
    samples: np.ndarray, frame_length: int, hop_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    if len(samples) < frame_length:
        padded = np.zeros(frame_length, dtype=np.float64)
        padded[: len(samples)] = samples
    else:
        padded = samples
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_length)[
        ::hop_length
    ]
    if frames.shape[0] == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    signs = np.signbit(frames)
    crossing = np.count_nonzero(signs[:, 1:] != signs[:, :-1], axis=1).astype(
        np.float64
    )
    zcr = crossing / float(max(1, frame_length - 1))
    return rms, zcr


def _refine_boundary(
    samples: np.ndarray, rough_start: int, rough_end: int, threshold: float
) -> Tuple[int, int]:
    start = max(0, int(rough_start))
    end = min(len(samples), int(rough_end))
    if end <= start:
        return start, end
    magnitude = np.abs(samples[start:end])
    active = np.flatnonzero(magnitude > threshold)
    if active.size == 0:
        return start, end
    return start + int(active[0]), start + int(active[-1]) + 1


def detect_active_runs(
    samples: np.ndarray, sample_rate: int, settings: SplitSettings
) -> Dict[str, object]:
    """Detect non-silent runs using both RMS energy and zero-crossing rate."""

    settings.validate()
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    if len(x) == 0:
        return {
            "runs": [],
            "active_run_count": 0,
            "silence_run_count": 0,
            "frame_count": 0,
        }
    frame_length = max(2, int(round(settings.frame_ms * sample_rate / 1000.0)))
    hop_length = max(1, int(round(settings.hop_ms * sample_rate / 1000.0)))
    rms, zcr = _frame_rms_and_zcr(x, frame_length, hop_length)
    if rms.size == 0:
        return {
            "runs": [],
            "active_run_count": 0,
            "silence_run_count": 0,
            "frame_count": 0,
        }
    is_silence = (rms <= settings.silence_rms) & (
        zcr <= settings.silence_zcr
    )
    is_active = ~is_silence

    runs: List[Tuple[int, int]] = []
    index = 0
    while index < len(is_active):
        if not is_active[index]:
            index += 1
            continue
        run_start = index
        while index + 1 < len(is_active) and is_active[index + 1]:
            index += 1
        run_end = index
        samples_start = run_start * hop_length
        samples_end = min(len(x), run_end * hop_length + frame_length)
        runs.append((samples_start, samples_end))
        index += 1

    min_silence = int(round(settings.min_silence_ms * sample_rate / 1000.0))
    merged: List[Tuple[int, int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= min_silence:
            merged[-1] = (merged[-1][0], max(merged[-1][1], run[1]))
        else:
            merged.append(run)

    guard = int(round(settings.boundary_guard_ms * sample_rate / 1000.0))
    refined: List[Tuple[int, int]] = []
    min_active = int(round(settings.min_active_ms * sample_rate / 1000.0))
    for rough_start, rough_end in merged:
        start, end = _refine_boundary(
            x,
            max(0, rough_start - guard),
            min(len(x), rough_end + guard),
            max(settings.silence_rms * 0.45, 1.0 / 32768.0),
        )
        start = max(0, start - guard)
        end = min(len(x), end + guard)
        if end - start >= min_active:
            refined.append((start, end))

    silence_run_count = 0
    cursor = 0
    for start, end in refined:
        if start > cursor:
            silence_run_count += 1
        cursor = max(cursor, end)
    if cursor < len(x):
        silence_run_count += 1
    return {
        "runs": refined,
        "active_run_count": len(refined),
        "silence_run_count": silence_run_count,
        "frame_count": int(rms.size),
    }


def assign_runs_to_sentences(
    runs: Sequence[Tuple[int, int]], sentences: Sequence[Sentence]
) -> List[Optional[Tuple[int, int]]]:
    """Assign detected runs to nearest sentence target by temporal midpoint.

    If multiple runs belong to one sentence (for example a very short closure),
    they are merged.  A sentence with no run is reported as a failed sentence
    while other sentences continue.
    """

    if not sentences:
        return []
    if len(runs) == len(sentences):
        return [tuple(run) for run in runs]

    target_centers = np.array(
        [(sentence.start_sample + sentence.end_sample) / 2.0 for sentence in sentences],
        dtype=np.float64,
    )
    grouped: List[List[Tuple[int, int]]] = [[] for _ in sentences]
    for start, end in runs:
        center = (start + end) / 2.0
        index = int(np.argmin(np.abs(target_centers - center)))
        grouped[index].append((int(start), int(end)))

    result: List[Optional[Tuple[int, int]]] = []
    for group in grouped:
        if not group:
            result.append(None)
        else:
            result.append((min(item[0] for item in group), max(item[1] for item in group)))
    return result


def split_language(
    samples: np.ndarray,
    sample_rate: int,
    sentences: Sequence[Sentence],
    settings: SplitSettings,
) -> SplitResult:
    detected = detect_active_runs(samples, sample_rate, settings)
    runs = list(detected["runs"])
    assignments = assign_runs_to_sentences(runs, sentences)
    return SplitResult(
        runs=runs,
        segments=assignments,
        active_run_count=int(detected["active_run_count"]),
        silence_run_count=int(detected["silence_run_count"]),
    )

