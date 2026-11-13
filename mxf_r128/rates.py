"""Frame-rate parsing, ffprobe inference, and deterministic fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping

from .constants import (
    FPS_25,
    FPS_2997,
    W_FPS_DEFAULT,
    W_FPS_INFERRED,
)


@dataclass(frozen=True)
class FrameRateResolution:
    value: Fraction
    text: str
    warning: str | None


def parse_rate(value: Any) -> Fraction | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"0/0", "N/A", "nan"}:
        return None
    try:
        if "/" in text:
            numerator_text, denominator_text = text.split("/", 1)
            numerator = int(numerator_text)
            denominator = int(denominator_text)
        else:
            numerator = int(text)
            denominator = 1
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if numerator <= 0 or denominator <= 0:
        return None
    return Fraction(numerator, denominator)


def format_rate(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def snap_supported(value: Fraction | None) -> Fraction | None:
    if value is None:
        return None
    candidates = (Fraction(25, 1), Fraction(30000, 1001))
    for candidate in candidates:
        tolerance = max(0.001, float(candidate) * 0.0001)
        if abs(float(value) - float(candidate)) <= tolerance:
            return candidate
    return None


def _infer_from_frame_count(
    stream: Mapping[str, Any], fallback_duration: Any
) -> Fraction | None:
    try:
        frame_count = int(str(stream.get("nb_frames", "")).strip())
    except (TypeError, ValueError):
        return None
    if frame_count <= 0:
        return None

    try:
        duration = float(stream.get("duration") or fallback_duration or 0)
    except (TypeError, ValueError):
        return None
    if duration <= 0:
        return None
    return snap_supported(Fraction(frame_count, 1) / Fraction(str(duration)))


def _infer_from_time_base(stream: Mapping[str, Any]) -> Fraction | None:
    time_base = parse_rate(stream.get("time_base"))
    if time_base is None or time_base <= 0:
        return None
    # A video stream time base of 1/25 or 1001/30000 is a useful last resort.
    candidate = Fraction(1, 1) / time_base
    return snap_supported(candidate)


def resolve_frame_rate(
    stream: Mapping[str, Any], fallback_duration: Any = None
) -> FrameRateResolution:
    average = snap_supported(parse_rate(stream.get("avg_frame_rate")))
    if average is not None:
        return FrameRateResolution(average, format_rate(average), None)

    nominal = snap_supported(parse_rate(stream.get("r_frame_rate")))
    if nominal is not None:
        return FrameRateResolution(nominal, format_rate(nominal), W_FPS_INFERRED)

    inferred = _infer_from_frame_count(stream, fallback_duration)
    if inferred is None:
        inferred = _infer_from_time_base(stream)
    if inferred is not None:
        return FrameRateResolution(inferred, format_rate(inferred), W_FPS_INFERRED)

    fallback = Fraction(25, 1)
    return FrameRateResolution(fallback, FPS_25, W_FPS_DEFAULT)


def is_fraction_rate(value: str | None, expected: str) -> bool:
    parsed = snap_supported(parse_rate(value))
    wanted = parse_rate(expected)
    return parsed is not None and wanted is not None and parsed == wanted


SUPPORTED_TEXT = (FPS_25, FPS_2997)
