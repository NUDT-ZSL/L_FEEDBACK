"""Exact millisecond-to-sample conversions."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


def as_decimal(value: Decimal | int | float | str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def ms_to_frames(ms: Decimal | int | float | str, sample_rate: int) -> int:
    """Round a millisecond position to the nearest sample (ties upward)."""
    return int(
        (as_decimal(ms) * Decimal(sample_rate) / Decimal(1000)).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )


def frames_to_ms(frames: int, sample_rate: int) -> float:
    return frames * 1000.0 / sample_rate


def resampled_frame_count(source_frames: int, source_rate: int, target_rate: int) -> int:
    """Return the nearest whole target-rate frame count for a source duration."""
    if source_rate == target_rate:
        return source_frames
    return int(
        (
            Decimal(source_frames) * Decimal(target_rate) / Decimal(source_rate)
        ).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    )
