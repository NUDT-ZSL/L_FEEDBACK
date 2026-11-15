"""EBU R128 loudness and true-peak measurement through FFmpeg."""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .errors import err

_INTEGRATED_RE = re.compile(r"\bI:\s*(-?(?:\d+(?:\.\d+)?|inf))\s+LUFS", re.IGNORECASE)
_PEAK_RE = re.compile(r"\bPeak:\s*(-?(?:\d+(?:\.\d+)?|inf))\s+dBFS", re.IGNORECASE)


@dataclass(frozen=True)
class LoudnessMeasurement:
    integrated_lufs: float | None
    true_peak_dbtp: float | None
    ffmpeg_version: str | None

    @property
    def is_silent(self) -> bool:
        return self.integrated_lufs is None


def find_ffmpeg() -> str:
    configured = os.environ.get("FFMPEG_BINARY")
    candidate = configured or "ffmpeg"
    found = shutil.which(candidate)
    if found is None:
        raise err(
            "E_FFMPEG_MISSING",
            "FFmpeg executable was not found on PATH",
            exit_code=3,
            configured=candidate,
        )
    return found


def _parse_measurement(value: str, *, unit: str) -> float | None:
    lowered = value.lower()
    if "inf" in lowered:
        return None
    result = float(value)
    if unit == "lufs" and result <= -69.99:
        return None
    return result


def measure_loudness(path: Path, *, timeout: float = 600.0) -> LoudnessMeasurement:
    ffmpeg = find_ffmpeg()
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-i",
        str(path),
        "-filter:a",
        "ebur128=peak=true",
        "-f",
        "null",
        "-",
    ]
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise err(
            "E_LOUDNESS_MEASURE",
            "FFmpeg loudness measurement failed to run",
            exit_code=6,
            reason=str(exc),
        ) from exc
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.splitlines()[-12:])
        raise err(
            "E_LOUDNESS_MEASURE",
            "FFmpeg could not measure the audio file",
            exit_code=6,
            returncode=completed.returncode,
            stderr=tail,
        )

    stderr = completed.stderr
    if "Summary:" not in stderr:
        raise err(
            "E_LOUDNESS_MEASURE",
            "FFmpeg ebur128 output did not contain a summary",
            exit_code=6,
        )
    summary = stderr.rsplit("Summary:", 1)[1]
    integrated_matches = _INTEGRATED_RE.findall(summary)
    peak_matches = _PEAK_RE.findall(summary)
    if not integrated_matches or not peak_matches:
        raise err(
            "E_LOUDNESS_MEASURE",
            "FFmpeg ebur128 summary is missing integrated loudness or true peak",
            exit_code=6,
        )

    version_match = re.search(r"ffmpeg version\s+(\S+)", stderr, re.IGNORECASE)
    integrated = _parse_measurement(integrated_matches[-1], unit="lufs")
    peak = _parse_measurement(peak_matches[-1], unit="dbtp")
    return LoudnessMeasurement(
        integrated_lufs=integrated,
        true_peak_dbtp=peak,
        ffmpeg_version=version_match.group(1) if version_match else None,
    )


def target_gain_db(measurement: LoudnessMeasurement, target_lufs: float) -> float:
    if measurement.integrated_lufs is None:
        raise err(
            "E_SILENT_MIX",
            "mix has no measurable integrated loudness",
            exit_code=6,
        )
    gain = target_lufs - measurement.integrated_lufs
    if not math.isfinite(gain):
        raise err(
            "E_LOUDNESS_MEASURE",
            "computed normalization gain is not finite",
            exit_code=6,
        )
    return gain


def apply_true_peak_limiter(
    source: Path,
    destination: Path,
    ceiling_dbtp: float,
    *,
    safety_margin_db: float = 1.0,
    timeout: float = 600.0,
) -> float:
    """Apply FFmpeg's look-ahead limiter and return the linear limit threshold."""
    ffmpeg = find_ffmpeg()
    limit_dbtp = ceiling_dbtp - safety_margin_db
    limit_linear = 10.0 ** (limit_dbtp / 20.0)
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-af",
        (
            f"alimiter=limit={limit_linear:.12f}:level=false:"
            "attack=5:release=50:latency=true"
        ),
        "-c:a",
        "pcm_f32le",
        str(destination),
    ]
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise err(
            "E_LIMITER",
            "FFmpeg limiter failed to run",
            exit_code=6,
            reason=str(exc),
        ) from exc
    if completed.returncode != 0 or not destination.is_file():
        raise err(
            "E_LIMITER",
            "FFmpeg could not apply true-peak limiting",
            exit_code=6,
            returncode=completed.returncode,
            stderr="\n".join(completed.stderr.splitlines()[-12:]),
        )
    return limit_linear
