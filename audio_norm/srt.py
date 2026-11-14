"""SRT subtitle parsing, formatting and silence-based timeline correction."""
from __future__ import annotations

import re
from dataclasses import dataclass


class SrtFormatError(Exception):
    """Raised when an SRT file cannot be parsed."""


@dataclass
class SubtitleEntry:
    index: int
    start: float  # seconds
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


_TIMESTAMP = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*"
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def parse_timestamp(value: str) -> float:
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*", value)
    if not match:
        raise SrtFormatError(f"invalid timestamp: {value!r}")
    hours, minutes, seconds, millis = (int(g) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    total_millis = int(round(seconds * 1000.0))
    hours, rem = divmod(total_millis, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_srt(content: str) -> list[SubtitleEntry]:
    content = content.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not content:
        raise SrtFormatError("empty SRT file")
    blocks = re.split(r"\n{2,}", content)
    entries = []
    for block in blocks:
        lines = [line for line in block.split("\n") if line.strip() != ""]
        if not lines:
            continue
        if not lines[0].strip().isdigit():
            raise SrtFormatError(f"missing numeric index in block: {block!r}")
        index = int(lines[0].strip())
        if len(lines) < 2:
            raise SrtFormatError(f"missing timing line for index {index}")
        timing = _TIMESTAMP.search(lines[1])
        if not timing:
            raise SrtFormatError(f"invalid timing line for index {index}")
        groups = timing.groups()
        start = (int(groups[0]) * 3600 + int(groups[1]) * 60
                 + int(groups[2]) + int(groups[3]) / 1000.0)
        end = (int(groups[4]) * 3600 + int(groups[5]) * 60
               + int(groups[6]) + int(groups[7]) / 1000.0)
        if end <= start:
            raise SrtFormatError(f"non-positive duration for index {index}")
        text = "\n".join(lines[2:]).strip()
        if not text:
            raise SrtFormatError(f"empty subtitle text for index {index}")
        entries.append(SubtitleEntry(index, start, end, text))
    if not entries:
        raise SrtFormatError("no subtitle entries found")
    return entries


def read_srt(path: str) -> list[SubtitleEntry]:
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            return parse_srt(handle.read())
    except (OSError, UnicodeDecodeError) as exc:
        raise SrtFormatError(f"cannot read SRT {path}: {exc}") from exc


def format_srt(entries: list[SubtitleEntry]) -> str:
    blocks = []
    for position, entry in enumerate(entries, start=1):
        blocks.append(
            f"{position}\n"
            f"{format_timestamp(entry.start)} --> {format_timestamp(entry.end)}\n"
            f"{entry.text}"
        )
    return "\n\n".join(blocks) + "\n"


def write_srt(path: str, entries: list[SubtitleEntry]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(format_srt(entries))


def adjust_timeline(entries: list[SubtitleEntry], silence_segments):
    """Shift entries whose start lies inside a silent region.

    When an entry starts within a silence segment, move its start to the
    segment end (first speech frame) while preserving its duration.

    Returns a list of adjustment dicts for the processing report.
    """
    adjustments = []
    for entry in entries:
        for seg_start, seg_end in silence_segments:
            if seg_start <= entry.start < seg_end:
                original_start = entry.start
                original_end = entry.end
                duration = entry.duration
                entry.start = seg_end
                entry.end = seg_end + duration
                adjustments.append({
                    "index": entry.index,
                    "original_start": round(original_start, 3),
                    "original_end": round(original_end, 3),
                    "new_start": round(entry.start, 3),
                    "new_end": round(entry.end, 3),
                    "silence_start": round(seg_start, 3),
                    "silence_end": round(seg_end, 3),
                })
                break
    return adjustments
