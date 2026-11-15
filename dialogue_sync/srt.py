"""Small, strict SRT parser/writer that preserves cue order and text."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import SrtError, TimecodeError
from .timecode import format_srt_timecode, parse_srt_timecode

_TIMING_RE = re.compile(
    r"^(?P<start>\d{2,}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*"
    r"(?P<end>\d{2,}:\d{2}:\d{2}[,\.]\d{3})(?P<settings>.*)$"
)


@dataclass
class SubtitleCue:
    index: int
    start_ms: int
    end_ms: int
    text: str
    settings: str = ""
    original_start_ms: int = field(init=False)
    original_end_ms: int = field(init=False)
    updated_by: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.original_start_ms = self.start_ms
        self.original_end_ms = self.end_ms


@dataclass
class SubtitleDocument:
    cues: list[SubtitleCue]

    def by_index(self) -> dict[int, SubtitleCue]:
        return {cue.index: cue for cue in self.cues}

    def render(self) -> str:
        blocks: list[str] = []
        for cue in self.cues:
            blocks.append(
                f"{cue.index}\n"
                f"{format_srt_timecode(cue.start_ms)} --> "
                f"{format_srt_timecode(cue.end_ms)}{cue.settings}\n"
                f"{cue.text}"
            )
        return "\n\n".join(blocks) + "\n"


def parse_srt_text(text: str, *, source: str = "<memory>") -> SubtitleDocument:
    if text.startswith("\ufeff"):
        text = text[1:]
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n[ \t]*\n", normalized.strip("\n"))
    if blocks == [""]:
        raise SrtError(f"{source}: subtitle file is empty")

    cues: list[SubtitleCue] = []
    seen_indices: set[int] = set()
    for block_number, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        if len(lines) < 2:
            raise SrtError(f"{source}: malformed cue block {block_number}")
        index_text = lines[0].strip()
        if not index_text.isdigit():
            raise SrtError(f"{source}: cue block {block_number} has no numeric index")
        index = int(index_text)
        if index in seen_indices:
            raise SrtError(f"{source}: duplicate subtitle index {index}")
        seen_indices.add(index)

        timing = _TIMING_RE.fullmatch(lines[1].strip())
        if not timing:
            raise SrtError(f"{source}: cue {index} has invalid timing line")
        try:
            start_ms = parse_srt_timecode(timing.group("start"), field=f"cue {index} start")
            end_ms = parse_srt_timecode(timing.group("end"), field=f"cue {index} end")
        except TimecodeError as exc:
            raise SrtError(str(exc)) from exc
        if end_ms <= start_ms:
            raise SrtError(f"{source}: cue {index} has zero or negative duration")

        cues.append(
            SubtitleCue(
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                text="\n".join(lines[2:]),
                settings=timing.group("settings"),
            )
        )
    if not cues:
        raise SrtError(f"{source}: no subtitle cues found")
    return SubtitleDocument(cues=cues)


def read_srt(path: Path) -> tuple[SubtitleDocument, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SrtError(f"cannot read subtitle file {path}: {exc}") from exc
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SrtError(f"{path}: subtitle file is not valid UTF-8") from exc
    return parse_srt_text(text, source=str(path)), raw


def find_overlaps(cues: list[SubtitleCue]) -> list[dict[str, object]]:
    """Return every pair of output cues whose timestamp intervals overlap."""

    overlaps: list[dict[str, object]] = []
    ordered = sorted(cues, key=lambda cue: (cue.start_ms, cue.end_ms, cue.index))
    active: list[SubtitleCue] = []
    for cue in ordered:
        active = [item for item in active if item.end_ms > cue.start_ms]
        for other in active:
            if other.index == cue.index:
                continue
            overlaps.append(
                {
                    "code": "W_SUBTITLE_OVERLAP",
                    "first_index": other.index,
                    "second_index": cue.index,
                    "overlap_ms": min(other.end_ms, cue.end_ms) - cue.start_ms,
                    "message": (
                        f"mapped subtitles {other.index} and {cue.index} overlap by "
                        f"{min(other.end_ms, cue.end_ms) - cue.start_ms} ms"
                    ),
                }
            )
        active.append(cue)
    return overlaps

