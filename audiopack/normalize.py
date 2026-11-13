"""Equal-power loudness normalization over fixed 100ms RMS windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config

EPS = 1.0e-12


@dataclass
class LoudnessInterval:
    start_frame: int
    end_frame: int
    start_ms: float
    end_ms: float
    measured_db: float

    def to_dict(self) -> dict:
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_ms": round(self.start_ms, 3),
            "end_ms": round(self.end_ms, 3),
            "measured_db": round(self.measured_db, 3),
        }


@dataclass
class NormalizationResult:
    samples: np.ndarray
    measured_loudness_db: float
    target_db: float
    requested_gain_db: float
    applied_gain_db: float
    gain_clamped: bool
    non_silent_windows: int
    warnings: list[LoudnessInterval]

    def to_dict(self) -> dict:
        return {
            "measured_loudness_db": round(self.measured_loudness_db, 3),
            "target_loudness_db": round(self.target_db, 3),
            "requested_gain_db": round(self.requested_gain_db, 3),
            "applied_gain_db": round(self.applied_gain_db, 3),
            "gain_clamped": self.gain_clamped,
            "non_silent_windows": self.non_silent_windows,
            "out_of_target_intervals": [item.to_dict() for item in self.warnings],
        }


def _window_rms_db(samples: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-window RMS in dB and the start frame of each full window."""
    full_windows = samples.shape[0] // window
    if full_windows == 0:
        trimmed = samples
        rms = np.array(
            [np.sqrt(np.mean(np.square(trimmed))) + EPS], dtype=np.float64
        )
        starts = np.array([0], dtype=np.int64)
        return 20.0 * np.log10(rms), starts

    usable = full_windows * window
    reshaped = samples[:usable].reshape(full_windows, window)
    rms = np.sqrt(np.mean(np.square(reshaped.astype(np.float64)), axis=1)) + EPS
    starts = np.arange(full_windows, dtype=np.int64) * window
    return 20.0 * np.log10(rms), starts


def normalize_track(samples: np.ndarray, config: Config) -> NormalizationResult:
    """Align whole-track loudness to target with a clamped linear gain."""
    window = config.rms_window_frames
    rms_db, starts = _window_rms_db(samples, window)

    silence_floor = config.silence_floor_db
    voiced_mask = rms_db > silence_floor
    voiced_db = rms_db[voiced_mask]

    if voiced_db.size == 0:
        # Fully silent track: nothing meaningful to normalize; keep as-is and
        # report a neutral result so downstream hashing stays well defined.
        return NormalizationResult(
            samples=samples.copy(),
            measured_loudness_db=float("-inf"),
            target_db=config.target_loudness_db,
            requested_gain_db=0.0,
            applied_gain_db=0.0,
            gain_clamped=False,
            non_silent_windows=0,
            warnings=[],
        )

    # Equal-power measure: average the window energies, then convert to dB.
    measured_db = 10.0 * np.log10(np.mean(np.power(10.0, voiced_db / 10.0)))
    requested_gain = config.target_loudness_db - measured_db
    applied_gain = float(
        np.clip(requested_gain, config.gain_min_db, config.gain_max_db)
    )
    clamped = not np.isclose(applied_gain, requested_gain)

    gain_linear = np.power(10.0, applied_gain / 20.0)
    normalized = (samples.astype(np.float32) * np.float32(gain_linear)).astype(
        np.float32
    )

    post_db, post_starts = _window_rms_db(normalized, window)
    warnings = _collect_intervals(
        post_db, post_starts, window, config
    )

    return NormalizationResult(
        samples=normalized,
        measured_loudness_db=float(measured_db),
        target_db=config.target_loudness_db,
        requested_gain_db=float(requested_gain),
        applied_gain_db=applied_gain,
        gain_clamped=clamped,
        non_silent_windows=int(voiced_db.size),
        warnings=warnings,
    )


def _collect_intervals(
    rms_db: np.ndarray, starts: np.ndarray, window: int, config: Config
) -> list[LoudnessInterval]:
    """Merge adjacent out-of-target non-silent windows into intervals."""
    low = config.target_loudness_db - config.loudness_tolerance_db
    high = config.target_loudness_db + config.loudness_tolerance_db
    bad = (rms_db > config.silence_floor_db) & ((rms_db < low) | (rms_db > high))

    intervals: list[LoudnessInterval] = []
    index = 0
    count = rms_db.shape[0]
    while index < count:
        if not bad[index]:
            index += 1
            continue
        begin = index
        while index + 1 < count and bad[index + 1]:
            index += 1
        end = index
        start_frame = int(starts[begin])
        end_frame = int(starts[end]) + window
        # Representative level is the mean energy across the merged windows.
        segment_db = 10.0 * np.log10(
            np.mean(np.power(10.0, rms_db[begin : end + 1] / 10.0))
        )
        intervals.append(
            LoudnessInterval(
                start_frame=start_frame,
                end_frame=end_frame,
                start_ms=start_frame / config.ms_frame,
                end_ms=end_frame / config.ms_frame,
                measured_db=float(segment_db),
            )
        )
        index += 1
    return intervals
