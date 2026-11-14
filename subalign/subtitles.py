"""Subtitle loading and saving for SRT and ASS/SSA files."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import pysubs2

from .encoding import decode_bytes, detect_encoding

FORMAT_BY_EXTENSION = {
    ".srt": "srt",
    ".ass": "ass",
    ".ssa": "ssa",
}


class SubtitleError(Exception):
    """Structured error raised when a subtitle file cannot be processed."""

    def __init__(self, path: str, error_type: str, message: str) -> None:
        super().__init__(message)
        self.path = path
        self.error_type = error_type
        self.message = message
        self.warnings: list[dict] = []

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass
class SubtitleEvent:
    """A single cue with the styling metadata needed for lossless round trips."""

    index: int
    start: float
    end: float
    text: str
    style: str = "Default"
    name: str = ""
    effect: str = ""
    layer: int = 0
    marginl: int = 0
    marginr: int = 0
    marginv: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SubtitleFile:
    path: str
    fmt: str
    encoding: str
    has_bom: bool
    events: list[SubtitleEvent]
    raw: pysubs2.SSAFile = field(repr=False)

    @property
    def basename(self) -> str:
        return os.path.basename(self.path)


def format_for_path(path: str) -> str:
    extension = os.path.splitext(path)[1].lower()
    try:
        return FORMAT_BY_EXTENSION[extension]
    except KeyError as exc:
        raise SubtitleError(
            path,
            "UnsupportedFormat",
            f"unsupported subtitle extension {extension!r}; expected .srt, .ass or .ssa",
        ) from exc


def load_subtitle(path: str) -> SubtitleFile:
    """Read and parse a subtitle file, raising :class:`SubtitleError` on failure."""

    if not os.path.exists(path):
        raise SubtitleError(path, "FileNotFound", "file does not exist")
    if os.path.isdir(path):
        raise SubtitleError(path, "NotAFile", "path points to a directory")
    fmt = format_for_path(path)
    try:
        with open(path, "rb") as handle:
            raw_bytes = handle.read()
    except OSError as exc:
        raise SubtitleError(path, "IOError", str(exc)) from exc

    detection = detect_encoding(raw_bytes)
    try:
        text, _ = decode_bytes(raw_bytes)
    except UnicodeDecodeError as exc:
        raise SubtitleError(
            path,
            "EncodingError",
            f"failed to decode file as {detection.encoding}: {exc}",
        ) from exc

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ssa = pysubs2.SSAFile.from_string(text, format_=fmt)
    except Exception as exc:  # pysubs2 raises a variety of parser exceptions
        raise SubtitleError(
            path,
            "ParseError",
            f"{type(exc).__name__}: {exc}",
        ) from exc

    events: list[SubtitleEvent] = []
    for index, event in enumerate(ssa.events):
        events.append(
            SubtitleEvent(
                index=index,
                start=float(event.start),
                end=float(event.end),
                text=event.text,
                style=event.style,
                name=event.name,
                effect=event.effect,
                layer=event.layer,
                marginl=event.marginl,
                marginr=event.marginr,
                marginv=event.marginv,
            )
        )

    return SubtitleFile(
        path=path,
        fmt=fmt,
        encoding=detection.encoding,
        has_bom=detection.has_bom,
        events=events,
        raw=ssa,
    )


def build_ssa(subtitle: SubtitleFile, aligned_events: list[SubtitleEvent]) -> pysubs2.SSAFile:
    """Clone the parsed SSA file and replace event timing, preserving styles."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssa = pysubs2.SSAFile.from_string(
            subtitle.raw.to_string(subtitle.fmt), format_=subtitle.fmt
        )
    for existing, aligned in zip(ssa.events, aligned_events):
        existing.start = int(round(aligned.start))
        existing.end = int(round(aligned.end))
        existing.style = aligned.style
        existing.name = aligned.name
        existing.effect = aligned.effect
        existing.layer = aligned.layer
        existing.marginl = aligned.marginl
        existing.marginr = aligned.marginr
        existing.marginv = aligned.marginv
    return ssa


def output_path_for(input_path: str, output_dir: str | None, suffix: str = "_aligned") -> str:
    directory = output_dir or os.path.dirname(os.path.abspath(input_path))
    stem, extension = os.path.splitext(os.path.basename(input_path))
    return os.path.join(directory, f"{stem}{suffix}{extension}")


def save_subtitle(
    subtitle: SubtitleFile,
    aligned_events: list[SubtitleEvent],
    output_path: str,
) -> None:
    """Write the aligned subtitle in the same format and UTF-8 encoding."""

    if output_dir := os.path.dirname(output_path):
        os.makedirs(output_dir, exist_ok=True)
    ssa = build_ssa(subtitle, aligned_events)
    content = ssa.to_string(subtitle.fmt)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
