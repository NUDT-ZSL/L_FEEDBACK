from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .config import ExportConfig
from .events import EventSpec, rate_fraction
from .fingerprint import sha256_json


@dataclass(frozen=True)
class Segment:
    kind: str
    source_start: int
    source_end: int
    output_start: int
    output_end: int
    rate: Fraction
    event_index: int | None = None

    @property
    def source_frames(self) -> int:
        return self.source_end - self.source_start

    @property
    def output_frames(self) -> int:
        return self.output_end - self.output_start

    @property
    def effective_rate(self) -> Fraction:
        if self.output_frames == 0:
            return Fraction(0, 1)
        return Fraction(self.source_frames, self.output_frames)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "event_index": self.event_index,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "output_start": self.output_start,
            "output_end": self.output_end,
            "nominal_rate": _fraction_text(self.rate),
            "effective_rate": _fraction_text(self.effective_rate),
        }


@dataclass(frozen=True)
class Timeline:
    sample_rate: int
    source_frames: int
    output_frames: int
    events: tuple[EventSpec, ...]
    segments: tuple[Segment, ...]
    fingerprint: str

    def to_dict(self) -> dict:
        event_segments = {
            segment.event_index: segment
            for segment in self.segments
            if segment.event_index is not None
        }
        timeline_events = []
        for event in self.events:
            item = event.to_dict()
            segment = event_segments.get(event.index)
            if segment is not None:
                item["output_start_frame"] = segment.output_start
                item["output_end_frame"] = segment.output_end
                item["output_duration_ms"] = round(
                    segment.output_frames * 1000.0 / self.sample_rate, 9
                )
            timeline_events.append(item)
        return {
            "sample_rate": self.sample_rate,
            "source_frames": self.source_frames,
            "source_duration_ms": round(
                self.source_frames * 1000.0 / self.sample_rate, 9
            ),
            "output_frames": self.output_frames,
            "output_duration_ms": round(
                self.output_frames * 1000.0 / self.sample_rate, 9
            ),
            "events": timeline_events,
            "segments": [segment.to_dict() for segment in self.segments],
            "fingerprint": self.fingerprint,
        }

    def output_anchor_for_source_frame(self, source_frame: int) -> int | None:
        for segment in self.segments:
            if segment.source_start <= source_frame <= segment.source_end:
                if source_frame == segment.source_end:
                    return segment.output_end
                return segment.output_start
        return None


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _round_half_up(value: Fraction) -> int:
    return (2 * value.numerator + value.denominator) // (2 * value.denominator)


def build_timeline(
    source_frames: int,
    events: list[EventSpec],
    config: ExportConfig,
) -> Timeline:
    segments: list[Segment] = []
    source_cursor = 0
    output_cursor = 0
    for event in events:
        if source_cursor < event.start_frame:
            length = event.start_frame - source_cursor
            segments.append(
                Segment(
                    kind="normal",
                    source_start=source_cursor,
                    source_end=event.start_frame,
                    output_start=output_cursor,
                    output_end=output_cursor + length,
                    rate=Fraction(1, 1),
                )
            )
            output_cursor += length

        rate = rate_fraction(event.rate)
        output_length = max(
            1,
            _round_half_up(Fraction(event.end_frame - event.start_frame, 1) / rate),
        )
        segments.append(
            Segment(
                kind="event",
                source_start=event.start_frame,
                source_end=event.end_frame,
                output_start=output_cursor,
                output_end=output_cursor + output_length,
                rate=rate,
                event_index=event.index,
            )
        )
        output_cursor += output_length
        source_cursor = event.end_frame

    if source_cursor < source_frames:
        length = source_frames - source_cursor
        segments.append(
            Segment(
                kind="normal",
                source_start=source_cursor,
                source_end=source_frames,
                output_start=output_cursor,
                output_end=output_cursor + length,
                rate=Fraction(1, 1),
            )
        )
        output_cursor += length

    normalized_events = [event.to_dict() for event in events]
    normalized_segments = [segment.to_dict() for segment in segments]
    fingerprint_payload = {
        "sample_rate": config.sample_rate,
        "source_frames": source_frames,
        "output_frames": output_cursor,
        "events": normalized_events,
        "segments": normalized_segments,
    }
    return Timeline(
        sample_rate=config.sample_rate,
        source_frames=source_frames,
        output_frames=output_cursor,
        events=tuple(events),
        segments=tuple(segments),
        fingerprint=sha256_json(fingerprint_payload),
    )
