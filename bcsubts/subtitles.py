from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import BusinessError


_TC_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*$"
)


@dataclass
class Cue:
    index: int
    start_ms: float
    end_ms: float
    text: str
    source_index: int

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class Correction:
    cue_index: int
    action: str
    reason: str
    old_start_ms: float | None = None
    old_end_ms: float | None = None
    new_start_ms: float | None = None
    new_end_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "cue_index": self.cue_index,
            "action": self.action,
            "reason": self.reason,
            "old_start_ms": self.old_start_ms,
            "old_end_ms": self.old_end_ms,
            "new_start_ms": self.new_start_ms,
            "new_end_ms": self.new_end_ms,
        }


@dataclass
class TimelineMetrics:
    cue_count: int
    negative_duration_count: int
    overlap_count: int
    gap_count: int
    overlaps: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cue_count": self.cue_count,
            "negative_duration_count": self.negative_duration_count,
            "overlap_count": self.overlap_count,
            "gap_over_5s_count": self.gap_count,
            "overlaps": self.overlaps,
            "gaps_over_5s": self.gaps,
        }


def _timecode_to_ms(hours: str, minutes: str, seconds: str, millis: str) -> float:
    h, m, s, ms = int(hours), int(minutes), int(seconds), int(millis)
    if m >= 60 or s >= 60:
        raise ValueError("minutes and seconds must be below 60")
    return ((h * 60 + m) * 60 + s) * 1000.0 + ms


def parse_timecode(value: str) -> float:
    text = value.strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})", text)
    if not match:
        raise ValueError(f"invalid timecode: {value!r}")
    try:
        return _timecode_to_ms(*match.groups())
    except ValueError as exc:
        raise ValueError(f"invalid timecode: {value!r}") from exc


def parse_srt_text(data: str, source: str = "<srt>") -> list[Cue]:
    text = data.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n\s*\n", text.strip()) if text.strip() else []
    cues: list[Cue] = []
    for block_no, block in enumerate(blocks):
        lines = [line for line in block.splitlines()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            raise BusinessError(
                "E_TIMECODE",
                "subtitle_parse",
                "SRT block has no timing line",
                {"source": source, "block": block_no + 1},
            )
        match = _TC_RE.match(lines[timing_index])
        if not match:
            raise BusinessError(
                "E_TIMECODE",
                "subtitle_parse",
                "Malformed SRT timecode",
                {
                    "source": source,
                    "block": block_no + 1,
                    "line": lines[timing_index],
                },
            )
        try:
            start = _timecode_to_ms(*match.groups()[:4])
            end = _timecode_to_ms(*match.groups()[4:])
        except ValueError as exc:
            raise BusinessError(
                "E_TIMECODE",
                "subtitle_parse",
                str(exc),
                {"source": source, "block": block_no + 1},
            ) from exc
        cue_text = "\n".join(lines[timing_index + 1 :]).strip()
        cues.append(Cue(len(cues), start, end, cue_text, len(cues)))
    if not cues:
        raise BusinessError("E_FORMAT", "subtitle_parse", "Subtitle file contains no cues")
    return cues


def parse_csv_text(data: str, source: str = "<csv>") -> list[Cue]:
    text = data.lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise BusinessError("E_FORMAT", "subtitle_parse", "CSV subtitle file has no header")
    field_map = {name.strip().lower(): name for name in reader.fieldnames if name}
    missing = [name for name in ("start", "end", "text") if name not in field_map]
    if missing:
        raise BusinessError(
            "E_FORMAT",
            "subtitle_parse",
            "CSV requires start, end and text columns",
            {"missing": missing, "header": reader.fieldnames},
        )
    cues: list[Cue] = []
    for row_no, row in enumerate(reader, start=2):
        try:
            start = parse_timecode(row.get(field_map["start"], ""))
            end = parse_timecode(row.get(field_map["end"], ""))
        except ValueError as exc:
            raise BusinessError(
                "E_TIMECODE",
                "subtitle_parse",
                str(exc),
                {"source": source, "row": row_no},
            ) from exc
        cue_text = (row.get(field_map["text"]) or "").strip()
        cues.append(Cue(len(cues), start, end, cue_text, len(cues)))
    if not cues:
        raise BusinessError("E_FORMAT", "subtitle_parse", "Subtitle CSV contains no cues")
    return cues


def parse_subtitles(path: str | Path) -> tuple[list[Cue], str]:
    path = Path(path)
    if not path.is_file():
        raise BusinessError("E_INPUT", "subtitle_parse", "Subtitle file not found", {"path": str(path)})
    try:
        data = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BusinessError(
            "E_FORMAT",
            "subtitle_parse",
            "Subtitle file must be UTF-8",
            {"path": str(path)},
        ) from exc
    suffix = path.suffix.lower()
    if suffix == ".srt":
        return parse_srt_text(data, str(path)), "srt"
    if suffix == ".csv":
        return parse_csv_text(data, str(path)), "csv"
    raise BusinessError(
        "E_FORMAT",
        "subtitle_parse",
        "Subtitle format must be .srt or .csv",
        {"path": str(path)},
    )


def format_srt_time(ms: float) -> str:
    total = max(0, int(round(ms)))
    hours, rem = divmod(total, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def write_srt(path: str | Path, cues: Iterable[Cue]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    ordered = sorted(cues, key=lambda c: (c.start_ms, c.end_ms, c.index))
    for number, cue in enumerate(ordered, start=1):
        lines.append(str(number))
        lines.append(f"{format_srt_time(cue.start_ms)} --> {format_srt_time(cue.end_ms)}")
        lines.append(cue.text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_srt_text(cues: Iterable[Cue]) -> str:
    lines: list[str] = []
    for number, cue in enumerate(sorted(cues, key=lambda c: (c.start_ms, c.end_ms, c.index)), 1):
        lines.append(str(number))
        lines.append(f"{format_srt_time(cue.start_ms)} --> {format_srt_time(cue.end_ms)}")
        lines.append(cue.text)
        lines.append("")
    return "\n".join(lines)


def timeline_metrics(cues: Iterable[Cue], gap_threshold_ms: float = 5000.0) -> TimelineMetrics:
    ordered = sorted(cues, key=lambda c: (c.start_ms, c.end_ms, c.index))
    negatives = sum(1 for cue in ordered if cue.end_ms < cue.start_ms)
    overlaps: list[dict[str, Any]] = []
    for prev, cur in zip(ordered, ordered[1:]):
        if cur.start_ms < prev.end_ms:
            overlaps.append(
                {
                    "first_index": prev.index,
                    "second_index": cur.index,
                    "overlap_ms": round(prev.end_ms - cur.start_ms, 3),
                }
            )
    gaps: list[dict[str, Any]] = []
    valid = [cue for cue in ordered if cue.end_ms >= cue.start_ms]
    for prev, cur in zip(valid, valid[1:]):
        gap = cur.start_ms - prev.end_ms
        if gap > gap_threshold_ms:
            gaps.append(
                {
                    "after_index": prev.index,
                    "before_index": cur.index,
                    "gap_ms": round(gap, 3),
                }
            )
    return TimelineMetrics(
        cue_count=len(ordered),
        negative_duration_count=negatives,
        overlap_count=len(overlaps),
        gap_count=len(gaps),
        overlaps=overlaps,
        gaps=gaps,
    )


def correct_timeline(
    cues: Iterable[Cue],
    policy: str = "truncate",
) -> tuple[list[Cue], list[Correction]]:
    if policy not in {"truncate", "skip"}:
        raise ValueError("overlap policy must be truncate or skip")
    corrections: list[Correction] = []
    result: list[Cue] = []

    for cue in sorted(cues, key=lambda c: (c.start_ms, c.end_ms, c.index)):
        current = Cue(cue.index, cue.start_ms, cue.end_ms, cue.text, cue.source_index)
        if current.end_ms < current.start_ms:
            corrections.append(
                Correction(
                    current.index,
                    "skip" if policy == "skip" else "truncate",
                    "negative_duration",
                    current.start_ms,
                    current.end_ms,
                    None if policy == "skip" else current.start_ms,
                    None if policy == "skip" else current.start_ms,
                )
            )
            if policy == "skip":
                continue
            current.end_ms = current.start_ms

        if current.end_ms <= current.start_ms and current.text:
            corrections.append(
                Correction(
                    current.index,
                    "skip",
                    "zero_duration_after_correction",
                    current.start_ms,
                    current.end_ms,
                )
            )
            continue

        if result and current.start_ms < result[-1].end_ms:
            previous = result[-1]
            corrections.append(
                Correction(
                    current.index,
                    "skip" if policy == "skip" else "truncate",
                    "overlap",
                    current.start_ms,
                    current.end_ms,
                    None if policy == "skip" else current.start_ms,
                    None if policy == "skip" else min(current.end_ms, previous.end_ms),
                    )
            )
            if policy == "skip":
                continue
            corrections.append(
                Correction(
                    previous.index,
                    "truncate",
                    "overlap_clip_previous",
                    previous.start_ms,
                    previous.end_ms,
                    previous.start_ms,
                    current.start_ms,
                )
            )
            previous.end_ms = current.start_ms
            if previous.end_ms <= previous.start_ms:
                corrections.append(
                    Correction(
                        previous.index,
                        "skip",
                        "zero_duration_after_overlap_clip",
                        previous.start_ms,
                        previous.end_ms,
                    )
                )
                result.pop()
                if result and current.start_ms < result[-1].end_ms:
                    previous = result[-1]
                    corrections.append(
                        Correction(
                            previous.index,
                            "truncate",
                            "overlap_clip_previous",
                            previous.start_ms,
                            previous.end_ms,
                            previous.start_ms,
                            current.start_ms,
                        )
                    )
                    previous.end_ms = current.start_ms
        result.append(current)

    result = [cue for cue in result if cue.end_ms > cue.start_ms]
    result.sort(key=lambda c: (c.start_ms, c.end_ms, c.index))
    return result, corrections


def apply_global_offset(
    cues: Iterable[Cue],
    offset_ms: float,
    duration_ms: float,
) -> tuple[list[Cue], list[Correction]]:
    corrections: list[Correction] = []
    result: list[Cue] = []
    for cue in cues:
        start = cue.start_ms + offset_ms
        end = cue.end_ms + offset_ms
        orig_start, orig_end = start, end
        if start < 0:
            start = 0.0
            end = max(0.0, end)
        if end > duration_ms:
            overflow = end - duration_ms
            end = duration_ms
            start = max(0.0, start - overflow)
        if start < 0 or end > duration_ms:
            corrections.append(
                Correction(
                    cue.index,
                    "skip",
                    "outside_audio_after_offset",
                    orig_start,
                    orig_end,
                )
            )
            continue
        if start != orig_start or end != orig_end:
            corrections.append(
                Correction(
                    cue.index,
                    "truncate",
                    "clamped_to_audio_bounds",
                    orig_start,
                    orig_end,
                    start,
                    end,
                )
            )
        result.append(Cue(cue.index, start, end, cue.text, cue.source_index))
    return result, corrections
