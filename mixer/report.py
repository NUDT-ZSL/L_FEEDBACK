"""Construction of deterministic timeline alignment reports."""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np

from .config import ProjectConfig, ShotSpec
from .errors import MixerError

SCHEMA_VERSION = 1
SILENCE_DBFS = -120.0


def dbfs(peak: float) -> float:
    if not math.isfinite(peak) or peak <= 1e-12:
        return SILENCE_DBFS
    return 20.0 * math.log10(peak)


def peak_of(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.max(np.abs(samples)))


def shot_entry(shot: ShotSpec, raw_peak: float | None = None) -> dict:
    return {
        "shot_id": shot.shot_id,
        "start_sample": int(shot.start_sample),
        "end_sample": int(shot.end_sample),
        "duration_samples": int(shot.duration_samples),
        "peak_dbfs": dbfs(raw_peak) if raw_peak is not None else None,
        "raw_peak": raw_peak,
    }


def build_partial_report(
    config: ProjectConfig,
    shots: Iterable[ShotSpec],
) -> dict:
    unique: dict[str, ShotSpec] = {}
    for shot in shots:
        unique[shot.shot_id] = shot
    ordered = sorted(unique.values(), key=lambda shot: (shot.start_sample, shot.shot_id))
    total_samples = max((shot.end_sample for shot in ordered), default=0)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "target_sample_rate": int(config.target_sample_rate),
        "total_samples": int(total_samples),
        "shots": [shot_entry(shot) for shot in ordered],
        "bus_weights": {
            key: float(config.bus_weights[key]) for key in ("dialogue", "ambience", "music")
        },
        "bus_peak_dbfs": None,
        "master_peak_dbfs": None,
        "limiter_active": False,
        "limiter_gain": 1.0,
    }


def build_error_report(error: MixerError) -> dict:
    report = error.context.get("partial_report")
    if isinstance(report, dict):
        result = dict(report)
    else:
        result = {
            "schema_version": SCHEMA_VERSION,
            "target_sample_rate": error.context.get("target_sample_rate"),
            "total_samples": error.context.get("total_samples"),
            "shots": [],
            "bus_weights": None,
            "bus_peak_dbfs": None,
            "master_peak_dbfs": None,
            "limiter_active": False,
            "limiter_gain": 1.0,
        }
    result["status"] = "error"
    result["error_code"] = error.error_code
    result["error_message"] = error.message
    return result


def build_success_report(
    config: ProjectConfig,
    shots: list[ShotSpec],
    *,
    buses: Mapping[str, np.ndarray],
    master: np.ndarray,
    shot_peaks: Mapping[str, float],
    total_samples: int,
    limiter_active: bool,
    limiter_gain: float,
) -> dict:
    entries = []
    for shot in sorted(shots, key=lambda item: (item.start_sample, item.shot_id)):
        entries.append(shot_entry(shot, float(shot_peaks.get(shot.shot_id, 0.0))))

    bus_peaks = {
        key: dbfs(peak_of(buses[key])) for key in ("ambience", "dialogue", "music")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "ok",
        "target_sample_rate": int(config.target_sample_rate),
        "total_samples": int(total_samples),
        "shots": entries,
        "bus_weights": {
            key: float(config.bus_weights[key]) for key in ("dialogue", "ambience", "music")
        },
        "bus_peak_dbfs": bus_peaks,
        "master_peak_dbfs": dbfs(peak_of(master)),
        "limiter_active": bool(limiter_active),
        "limiter_gain": float(limiter_gain),
    }
