"""End-to-end deterministic loudness normalization pipeline."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .audio import (
    same_file,
    read_wav,
    write_text_atomic,
    write_wav_atomic,
)
from .errors import PipelineError
from .limiter import LimiterResult, limit_peaks
from .loudness import LoudnessMeasurement, measure_loudness, sample_peak
from .report import build_report, serialize_report


MIN_TARGET_LUFS = -35.0
MAX_TARGET_LUFS = -5.0
DEFAULT_TARGET_LUFS = -23.0
DEFAULT_SEGMENT_SECONDS = 3.0
OUTPUT_CEILING = 1.0


@dataclass
class ProcessResult:
    input_path: Path
    output_path: Path
    report_path: Path
    input_loudness_lufs: float
    output_loudness_lufs: float
    requested_gain_db: float
    applied_base_gain_db: float
    max_limiting_db: float
    limiter_applied: bool
    elapsed_seconds: float
    report: Dict[str, object]


def _validate_options(
    target_lufs: float,
    segment_seconds: float,
) -> None:
    if not math.isfinite(target_lufs) or not (
        MIN_TARGET_LUFS <= target_lufs <= MAX_TARGET_LUFS
    ):
        raise PipelineError(
            "E_USAGE",
            "target loudness must be between {0:.0f} and {1:.0f} LUFS".format(
                MIN_TARGET_LUFS, MAX_TARGET_LUFS
            ),
        )
    if not math.isfinite(segment_seconds) or segment_seconds <= 0.0:
        raise PipelineError(
            "E_USAGE", "segment duration must be a positive number of seconds"
        )


def _require_measurable(
    measurement: LoudnessMeasurement, stage: str
) -> float:
    loudness = float(measurement.integrated_lufs)
    if not math.isfinite(loudness):
        raise PipelineError(
            "E_SILENT",
            "{0} audio has no energy above the -70 LUFS absolute gate".format(
                stage
            ),
        )
    return loudness


def _limit_to_target(
    audio: np.ndarray,
    sample_rate: int,
    target_lufs: float,
    initial_gain_db: float,
) -> Tuple[np.ndarray, LoudnessMeasurement, float, LimiterResult, int]:
    """Limit peaks and iteratively raise pre-limiter gain toward the target.

    A limiter lowers integrated loudness when it acts on dense peaks. Feeding
    a slightly higher base gain and limiting again restores the target while
    retaining the ceiling. The secant update is deterministic and bounded.
    """

    working_gain_db = initial_gain_db
    previous_gain_db: Optional[float] = None
    previous_loudness: Optional[float] = None
    best_audio: Optional[np.ndarray] = None
    best_measurement: Optional[LoudnessMeasurement] = None
    best_limit: Optional[LimiterResult] = None
    best_gain_db = working_gain_db
    best_error = float("inf")
    iterations = 0

    for iteration in range(16):
        iterations = iteration + 1
        pre_limited = audio * (10.0 ** (working_gain_db / 20.0))
        limited = limit_peaks(
            pre_limited,
            sample_rate=sample_rate,
            ceiling=OUTPUT_CEILING,
        )
        measurement = measure_loudness(limited.samples, sample_rate)
        loudness = float(measurement.integrated_lufs)
        error = target_lufs - loudness
        if abs(error) < abs(best_error):
            best_audio = limited.samples
            best_measurement = measurement
            best_limit = limited
            best_gain_db = working_gain_db
            best_error = abs(error)
        if abs(error) <= 0.02:
            best_audio = limited.samples
            best_measurement = measurement
            best_limit = limited
            best_gain_db = working_gain_db
            best_error = abs(error)
            break

        if (
            previous_gain_db is not None
            and previous_loudness is not None
            and abs(working_gain_db - previous_gain_db) > 1.0e-9
        ):
            slope = (loudness - previous_loudness) / (
                working_gain_db - previous_gain_db
            )
            if slope > 0.05:
                adjustment = error / slope
            else:
                adjustment = error * 2.0
        else:
            adjustment = error * 2.0
        adjustment = max(-6.0, min(6.0, adjustment))
        if abs(adjustment) < 1.0e-6:
            adjustment = math.copysign(1.0e-6, error if error != 0.0 else 1.0)
        previous_gain_db = working_gain_db
        previous_loudness = loudness
        working_gain_db += adjustment

    assert best_audio is not None
    assert best_measurement is not None
    assert best_limit is not None
    return (
        np.clip(best_audio, -OUTPUT_CEILING, OUTPUT_CEILING),
        best_measurement,
        best_gain_db,
        best_limit,
        iterations,
    )


def process_file(
    input_path: Path,
    output_path: Path,
    target_lufs: float = DEFAULT_TARGET_LUFS,
    limiter_enabled: bool = True,
    report_path: Optional[Path] = None,
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
) -> ProcessResult:
    """Normalize one WAV file and write a new WAV plus deterministic JSON."""

    started = time.perf_counter()
    source = Path(input_path)
    destination = Path(output_path)
    report_destination = (
        Path(report_path)
        if report_path is not None
        else Path(str(destination) + ".json")
    )

    _validate_options(target_lufs, segment_seconds)
    if same_file(source, destination):
        raise PipelineError(
            "E_USAGE",
            "output path must differ from the input path; the source is never modified",
        )
    if same_file(source, report_destination):
        raise PipelineError(
            "E_USAGE", "report path must differ from the input path"
        )
    if same_file(destination, report_destination):
        raise PipelineError(
            "E_USAGE", "report path must differ from the output WAV path"
        )

    audio_file = read_wav(source)
    before_audio = audio_file.samples
    before_measurement = measure_loudness(
        before_audio, audio_file.sample_rate
    )
    input_loudness = _require_measurable(before_measurement, "input")

    requested_gain_db = target_lufs - input_loudness
    requested_linear_gain = 10.0 ** (requested_gain_db / 20.0)
    desired_audio = before_audio * requested_linear_gain
    desired_peak = sample_peak(desired_audio)

    limiter_applied = False
    max_limiting_db = 0.0
    limited_sample_count = 0
    maximum_overshoot = 0.0
    applied_base_gain_db = requested_gain_db
    limiter_iterations = 0
    after_measurement: Optional[LoudnessMeasurement] = None

    if desired_peak <= OUTPUT_CEILING:
        after_audio = desired_audio
    elif limiter_enabled:
        (
            after_audio,
            after_measurement,
            applied_base_gain_db,
            limited,
            limiter_iterations,
        ) = _limit_to_target(
            before_audio,
            sample_rate=audio_file.sample_rate,
            target_lufs=target_lufs,
            initial_gain_db=requested_gain_db,
        )
        limiter_applied = True
        max_limiting_db = limited.max_reduction_db
        limited_sample_count = limited.limited_sample_count
        maximum_overshoot = limited.maximum_overshoot
    else:
        safe_linear_gain = requested_linear_gain * (
            OUTPUT_CEILING / desired_peak
        )
        after_audio = before_audio * safe_linear_gain
        applied_base_gain_db = 20.0 * math.log10(safe_linear_gain)

    after_audio = np.clip(after_audio, -OUTPUT_CEILING, OUTPUT_CEILING)
    if after_measurement is None:
        after_measurement = measure_loudness(
            after_audio, audio_file.sample_rate
        )
    output_loudness = _require_measurable(after_measurement, "processed")

    report = build_report(
        input_path=source,
        output_path=destination,
        sample_rate=audio_file.sample_rate,
        format_name=audio_file.format,
        subtype=audio_file.subtype,
        channel_count=audio_file.channels,
        target_lufs=target_lufs,
        segment_seconds=segment_seconds,
        before_audio=before_audio,
        after_audio=after_audio,
        before_measurement=before_measurement,
        after_measurement=after_measurement,
        requested_gain_db=requested_gain_db,
        applied_base_gain_db=applied_base_gain_db,
        limiter_enabled=limiter_enabled,
        limiter_applied=limiter_applied,
        max_limiting_db=max_limiting_db,
        limited_sample_count=limited_sample_count,
        maximum_overshoot=maximum_overshoot,
        limiter_iterations=limiter_iterations,
    )

    write_wav_atomic(
        destination=destination,
        samples=after_audio,
        sample_rate=audio_file.sample_rate,
        subtype=audio_file.subtype,
        format_name=audio_file.format,
    )
    write_text_atomic(report_destination, serialize_report(report))
    elapsed = time.perf_counter() - started

    return ProcessResult(
        input_path=source,
        output_path=destination,
        report_path=report_destination,
        input_loudness_lufs=input_loudness,
        output_loudness_lufs=output_loudness,
        requested_gain_db=requested_gain_db,
        applied_base_gain_db=applied_base_gain_db,
        max_limiting_db=max_limiting_db,
        limiter_applied=limiter_applied,
        elapsed_seconds=elapsed,
        report=report,
    )

