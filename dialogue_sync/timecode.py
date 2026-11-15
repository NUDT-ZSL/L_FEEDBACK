"""Frame-rate independent HH:MM:SS.mmm utilities."""

from __future__ import annotations

import re

from .errors import TimecodeError

_TIMECODE_RE = re.compile(
    r"^(?P<hours>\d{2,}):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)\."
    r"(?P<milliseconds>\d{3})$"
)
_SRT_TIME_RE = re.compile(
    r"^(?P<hours>\d{2,}):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)"
    r"[,\.](?P<milliseconds>\d{3})$"
)


def parse_timecode(value: object, *, field: str = "timecode") -> int:
    """Parse a strict HH:MM:SS.mmm value into integer milliseconds."""

    if not isinstance(value, str):
        raise TimecodeError(f"{field} must be an HH:MM:SS.mmm string")
    match = _TIMECODE_RE.fullmatch(value.strip())
    if not match:
        raise TimecodeError(
            f"{field} must use HH:MM:SS.mmm and have exactly three fractional digits: "
            f"{value!r}"
        )
    return (
        int(match.group("hours")) * 3_600_000
        + int(match.group("minutes")) * 60_000
        + int(match.group("seconds")) * 1_000
        + int(match.group("milliseconds"))
    )


def format_timecode(milliseconds: int) -> str:
    """Format non-negative integer milliseconds as HH:MM:SS.mmm."""

    if milliseconds < 0:
        raise ValueError("negative timecode cannot be formatted")
    hours, remainder = divmod(int(milliseconds), 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def parse_srt_timecode(value: str, *, field: str = "SRT timecode") -> int:
    """Parse standard SRT timing (comma or tolerant dot separator)."""

    match = _SRT_TIME_RE.fullmatch(value.strip())
    if not match:
        raise TimecodeError(f"invalid {field}: {value!r}")
    return (
        int(match.group("hours")) * 3_600_000
        + int(match.group("minutes")) * 60_000
        + int(match.group("seconds")) * 1_000
        + int(match.group("milliseconds"))
    )


def format_srt_timecode(milliseconds: int) -> str:
    """Format an SRT timestamp with its conventional comma separator."""

    return format_timecode(milliseconds).replace(".", ",")


def ms_to_samples(milliseconds: int, sample_rate: int) -> int:
    """Round a millisecond position to the nearest output sample."""

    if milliseconds < 0:
        raise ValueError("negative sample position")
    return int(round(milliseconds * sample_rate / 1_000.0))


def samples_to_ms(samples: int, sample_rate: int) -> float:
    return samples * 1_000.0 / sample_rate

