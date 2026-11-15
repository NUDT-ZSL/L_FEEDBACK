from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .errors import BakerError


def parse_rate(value: Any, field: str) -> Fraction:
    if isinstance(value, bool):
        raise BakerError(
            "E_INVALID_FPS",
            "Frame rate must be a positive number or rational string.",
            {"field": field, "value": value},
        )
    try:
        if isinstance(value, int):
            rate = Fraction(value, 1)
        elif isinstance(value, float):
            rate = Fraction(str(value))
        elif isinstance(value, str):
            if "/" in value:
                numerator, denominator = value.split("/", 1)
                rate = Fraction(int(numerator), int(denominator))
            else:
                rate = Fraction(value)
        else:
            raise ValueError("unsupported type")
    except (ValueError, ZeroDivisionError) as exc:
        raise BakerError(
            "E_INVALID_FPS",
            "Frame rate must be a positive number or rational string.",
            {"field": field, "value": value},
        ) from exc
    if rate <= 0:
        raise BakerError(
            "E_INVALID_FPS",
            "Frame rate must be positive.",
            {"field": field, "value": value},
        )
    return rate


def fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True)
class Timebase:
    fps: Fraction
    sample_rate: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise BakerError(
                "E_INVALID_SAMPLE_RATE",
                "Sample rate must be positive.",
                {"sample_rate": self.sample_rate},
            )
        if Fraction(self.sample_rate, 1) / self.fps < 1:
            raise BakerError(
                "E_FPS_TOO_HIGH",
                "Frame rate must not exceed one frame per audio sample.",
                {
                    "fps": fraction_text(self.fps),
                    "sample_rate": self.sample_rate,
                },
            )

    @property
    def fps_text(self) -> str:
        return fraction_text(self.fps)

    @property
    def fps_decimal(self) -> float:
        return float(self.fps)

    @property
    def samples_per_frame(self) -> Fraction:
        return Fraction(self.sample_rate, 1) / self.fps

    def frame_to_sample(self, frame: int) -> int:
        """Map a frame boundary to the first integer sample at or after it."""
        numerator = frame * self.sample_rate * self.fps.denominator
        denominator = self.fps.numerator
        return (numerator + denominator - 1) // denominator

    def sample_to_frame(self, sample: int) -> int:
        """Map a sample boundary back to a frame boundary using floor."""
        numerator = sample * self.fps.numerator
        denominator = self.sample_rate * self.fps.denominator
        return numerator // denominator

    def total_samples(self, duration_frames: int) -> int:
        return self.frame_to_sample(duration_frames)

    def frame_sample_span(self, frame: int) -> tuple[int, int]:
        return self.frame_to_sample(frame), self.frame_to_sample(frame + 1)

    def validate_roundtrip(self, duration_frames: int) -> bool:
        return all(
            self.sample_to_frame(self.frame_to_sample(frame)) == frame
            for frame in range(duration_frames + 1)
        )

    def as_report(self, duration_frames: int) -> dict[str, Any]:
        return {
            "frame_to_sample": "ceiling_integer_sample",
            "sample_to_frame": "floor",
            "fps": self.fps_text,
            "fps_decimal": self.fps_decimal,
            "sample_rate": self.sample_rate,
            "samples_per_frame_fraction": fraction_text(self.samples_per_frame),
            "duration_frames": duration_frames,
            "duration_samples": self.total_samples(duration_frames),
            "boundary_roundtrip_ok": self.validate_roundtrip(duration_frames),
        }

