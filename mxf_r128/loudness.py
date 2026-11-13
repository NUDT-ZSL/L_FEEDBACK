"""EBU R128 measurement and deterministic gain calculation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .constants import (
    LUFS_TOLERANCE,
    TARGET_LUFS,
    TRUE_PEAK_LIMIT_DBTP,
)
from .process import run_command


class LoudnessMeasurementError(Exception):
    pass


class LoudnessMeasurementTimeout(LoudnessMeasurementError):
    pass


@dataclass(frozen=True)
class LoudnessMeasurement:
    integrated_lufs: float | None
    lra: float | None
    true_peak_dbtp: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "lufs": self.integrated_lufs,
            "lra": self.lra,
            "true_peak_dbtp": self.true_peak_dbtp,
        }


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _extract_measurement(stdout: str, stderr: str) -> LoudnessMeasurement:
    text = f"{stdout}\n{stderr}"
    candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
    payload: dict[str, object] | None = None
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "input_i" in parsed:
            payload = parsed
            break
    if payload is None:
        raise LoudnessMeasurementError("loudness summary unavailable")
    return LoudnessMeasurement(
        integrated_lufs=_finite_float(payload.get("input_i")),
        lra=_finite_float(payload.get("input_lra")),
        true_peak_dbtp=_finite_float(payload.get("input_tp")),
    )


def measure_loudness(
    ffmpeg: str,
    path: Path,
    *,
    timeout: float,
) -> LoudnessMeasurement:
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "info",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-af",
        f"loudnorm=I={TARGET_LUFS}:TP={TRUE_PEAK_LIMIT_DBTP}:"
        "LRA=11:print_format=json",
        "-f",
        "null",
        "-",
    ]
    result = run_command(command, timeout=timeout)
    if result.timed_out:
        raise LoudnessMeasurementTimeout("loudness measurement timed out")
    if result.returncode != 0:
        raise LoudnessMeasurementError("loudness measurement failed")
    return _extract_measurement(result.stdout, result.stderr)


def normalization_gain_db(measurement: LoudnessMeasurement) -> float | None:
    if measurement.integrated_lufs is None or measurement.true_peak_dbtp is None:
        return None
    gain = TARGET_LUFS - measurement.integrated_lufs
    peak_limited_gain = TRUE_PEAK_LIMIT_DBTP - measurement.true_peak_dbtp
    return min(gain, peak_limited_gain)


def loudness_passes(measurement: LoudnessMeasurement) -> bool:
    if measurement.integrated_lufs is None or measurement.true_peak_dbtp is None:
        return False
    integrated_ok = (
        abs(measurement.integrated_lufs - TARGET_LUFS) <= LUFS_TOLERANCE
    )
    peak_ok = measurement.true_peak_dbtp <= TRUE_PEAK_LIMIT_DBTP
    return integrated_ok and peak_ok
