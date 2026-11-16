"""ITU-R BS.1770 loudness analysis with FFmpeg EBU R128."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .errors import ProcessingError
from .mixing import seconds_to_samples
from .scene import DialogueIntervalSpec


_SILENCE_FLOOR_LUFS = -120.7
_FRAME_PATTERN = re.compile(
    r"\[Parsed_ebur128[^\]]*\]\s+t:\s*([0-9]+(?:\.[0-9]+)?)"
    r".*?\bS:\s*(-?(?:[0-9]+(?:\.[0-9]+)?|inf))",
    re.IGNORECASE,
)
_SUMMARY_I_PATTERN = re.compile(
    r"Integrated loudness:\s*\n\s*I:\s*(-?(?:[0-9]+(?:\.[0-9]+)?|inf))\s*LUFS",
    re.IGNORECASE,
)
_SUMMARY_PEAK_PATTERN = re.compile(
    r"True peak:\s*\n\s*Peak:\s*(-?(?:[0-9]+(?:\.[0-9]+)?|inf))\s*dBFS",
    re.IGNORECASE,
)


def _parse_number(text: str) -> float:
    lowered = text.strip().lower()
    if lowered in ("inf", "+inf", "infinity", "+infinity"):
        return math.inf
    if lowered in ("-inf", "-infinity"):
        return -math.inf
    return float(text)


def _safe_loudness(value: float) -> float:
    if not math.isfinite(value) or value <= _SILENCE_FLOOR_LUFS:
        return _SILENCE_FLOOR_LUFS
    return value


def _run_ebur128(path: Path) -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise ProcessingError(
            "ffmpeg was not found on PATH; install FFmpeg and make it available"
        )
    command = [
        executable,
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "verbose",
        "-i",
        str(path),
        "-filter_complex",
        "ebur128=peak=true:metadata=1",
        "-f",
        "null",
        "-",
    ]
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            **kwargs,
        )
    except OSError as exc:
        raise ProcessingError(f"unable to execute ffmpeg: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        tail = "\n".join(detail[-12:]) if detail else "no diagnostic output"
        raise ProcessingError(
            f"ffmpeg loudness analysis failed with exit code {completed.returncode}:\n{tail}"
        )
    return completed.stderr


def _parse_ffmpeg_output(stderr: str) -> Tuple[float, float, List[Tuple[float, float]]]:
    integrated_match = _SUMMARY_I_PATTERN.search(stderr)
    peak_match = _SUMMARY_PEAK_PATTERN.search(stderr)
    if integrated_match is None or peak_match is None:
        raise ProcessingError(
            "could not parse FFmpeg ebur128 summary "
            "(integrated loudness/true peak)"
        )
    integrated = _safe_loudness(_parse_number(integrated_match.group(1)))
    true_peak = _safe_loudness(_parse_number(peak_match.group(1)))
    frames: List[Tuple[float, float]] = []
    for match in _FRAME_PATTERN.finditer(stderr):
        frame_time = float(match.group(1))
        short_term = _parse_number(match.group(2))
        frames.append((frame_time, short_term))
    return integrated, true_peak, frames


def _k_weighting_coefficients(sample_rate: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return BS.1770 K-weighting shelf and high-pass IIR coefficients."""

    # Stage 1: high-frequency shelving filter.
    shelf_f0 = 1681.974450955533
    shelf_gain_db = 3.999843853973347
    shelf_q = 0.7071752369554196
    k = math.tan(math.pi * shelf_f0 / sample_rate)
    vh = 10.0 ** (shelf_gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / shelf_q + k * k
    shelf_b = np.array(
        [
            (vh + vb * k / shelf_q + k * k) / a0,
            2.0 * (k * k - vh) / a0,
            (vh - vb * k / shelf_q + k * k) / a0,
        ],
        dtype=np.float64,
    )
    shelf_a = np.array(
        [
            1.0,
            2.0 * (k * k - 1.0) / a0,
            (1.0 - k / shelf_q + k * k) / a0,
        ],
        dtype=np.float64,
    )

    # Stage 2: RLB high-pass filter.
    highpass_f0 = 38.13547087602444
    highpass_q = 0.5003270373238773
    k = math.tan(math.pi * highpass_f0 / sample_rate)
    a0 = 1.0 + k / highpass_q + k * k
    highpass_b = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    highpass_a = np.array(
        [
            1.0,
            2.0 * (k * k - 1.0) / a0,
            (1.0 - k / highpass_q + k * k) / a0,
        ],
        dtype=np.float64,
    )
    return shelf_b, shelf_a, highpass_b, highpass_a


def _biquad(audio: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    output = np.empty_like(audio)
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    a1, a2 = float(a[1]), float(a[2])
    for channel in range(audio.shape[1]):
        source = audio[:, channel]
        target = output[:, channel]
        x1 = x2 = y1 = y2 = 0.0
        for index, sample in enumerate(source):
            value = b0 * sample + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            target[index] = value
            x2, x1 = x1, sample
            y2, y1 = y1, value
    return output


def _k_weight(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    shelf_b, shelf_a, highpass_b, highpass_a = _k_weighting_coefficients(sample_rate)
    filtered = _biquad(audio, shelf_b, shelf_a)
    return _biquad(filtered, highpass_b, highpass_a)


def _energy_to_lufs(energy_per_frame: float) -> float:
    if energy_per_frame <= 0.0:
        return _SILENCE_FLOOR_LUFS
    return _safe_loudness(-0.691 + 10.0 * math.log10(energy_per_frame))


def _short_term_fallback(
    filtered: np.ndarray,
    sample_rate: int,
    interval: DialogueIntervalSpec,
) -> Tuple[float, Optional[float]]:
    """BS.1770 3 s short-term fallback for scenes shorter than one EBU frame."""

    energy = np.sum(filtered * filtered, axis=1)
    start = min(seconds_to_samples(interval.start, sample_rate), energy.size)
    end = min(seconds_to_samples(interval.end, sample_rate), energy.size)
    if end <= start:
        return _SILENCE_FLOOR_LUFS, None
    window_samples = max(1, seconds_to_samples(3.0, sample_rate))
    hop_samples = max(1, seconds_to_samples(0.1, sample_rate))
    length = end - start

    if length < window_samples:
        window_energy = float(np.sum(energy[start:end])) / window_samples
        return _energy_to_lufs(window_energy), interval.start

    cumulative = np.concatenate(([0.0], np.cumsum(energy[start:end])))
    best_energy = -1.0
    best_time = interval.start
    for offset in range(0, length - window_samples + 1, hop_samples):
        window_energy = float(
            cumulative[offset + window_samples] - cumulative[offset]
        ) / window_samples
        if window_energy > best_energy:
            best_energy = window_energy
            best_time = (start + offset + window_samples) / sample_rate
    return _energy_to_lufs(best_energy), best_time


def analyze_loudness(
    output_path: Path,
    audio: np.ndarray,
    sample_rate: int,
    dialogues: Sequence[DialogueIntervalSpec],
) -> Dict[str, object]:
    """Analyze integrated/true-peak loudness and per-dialogue short-term maxima."""

    stderr = _run_ebur128(output_path)
    integrated, true_peak, frames = _parse_ffmpeg_output(stderr)
    dialogue_reports: List[Dict[str, object]] = []
    filtered_audio: Optional[np.ndarray] = None
    for index, interval in enumerate(dialogues):
        candidates = [
            (frame_time, short_term)
            for frame_time, short_term in frames
            if frame_time + 1e-9 >= interval.start
            and frame_time - 1e-9 <= interval.end
            and frame_time >= 2.9
        ]
        if candidates:
            best_time, best_value = max(candidates, key=lambda item: item[1])
            short_term = _safe_loudness(best_value)
            method = "ffmpeg_ebur128_3s_within_interval"
        else:
            if filtered_audio is None:
                filtered_audio = _k_weight(audio, sample_rate)
            short_term, best_time = _short_term_fallback(
                filtered_audio, sample_rate, interval
            )
            method = "numpy_bs1770_3s_fallback"
        dialogue_reports.append(
            {
                "index": index,
                "start_seconds": interval.start,
                "end_seconds": interval.end,
                "start_sample": seconds_to_samples(interval.start, sample_rate),
                "end_sample": seconds_to_samples(interval.end, sample_rate),
                "duck_depth_db": interval.duck_depth_db,
                "duck_gain": interval.duck_gain,
                "max_short_term_lufs": short_term,
                "max_short_term_time_seconds": best_time,
                "short_term_window_seconds": 3.0,
                "short_term_hop_seconds": 0.1,
                "method": method,
            }
        )

    return {
        "standard": "ITU-R BS.1770-4 / EBU R128",
        "sample_rate_hz": sample_rate,
        "channels": 2,
        "duration_seconds": audio.shape[0] / sample_rate,
        "total_samples": int(audio.shape[0]),
        "integrated_lufs": integrated,
        "true_peak_dbtp": true_peak,
        "integrated_method": "ffmpeg ebur128 (K-weighted, gated)",
        "true_peak_method": "ffmpeg ebur128 true peak (4x oversampled)",
        "silence_floor_lufs": _SILENCE_FLOOR_LUFS,
        "dialogue_intervals": dialogue_reports,
    }


def write_json(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError) as exc:
        raise ProcessingError(f"cannot write JSON report {path}: {exc}") from exc
