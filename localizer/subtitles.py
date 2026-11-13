from __future__ import annotations

import html

from localizer.manifest import Clip


def format_vtt_timestamp(milliseconds: int) -> str:
    if milliseconds < 0:
        raise ValueError("VTT timestamps cannot be negative")
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def escape_webvtt_text(text: str) -> str:
    """Escape cue payload markup and make multi-line cue text safe."""

    normalized = (
        text.replace("\x00", "\ufffd")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
    )
    if normalized.startswith("\ufeff"):
        normalized = normalized[1:]
    normalized = html.escape(normalized, quote=False)
    lines = normalized.split("\n")
    return "\n".join("\u00a0" if not line.strip() else line for line in lines)


def build_webvtt(clips: tuple[Clip, ...]) -> str:
    cues: list[str] = []
    for index, clip in enumerate(clips, start=1):
        payload = escape_webvtt_text(clip.text)
        if not payload:
            payload = "\u00a0"
        cues.append(
            f"cue-{index:04d}\n"
            f"{format_vtt_timestamp(clip.start_ms)} --> "
            f"{format_vtt_timestamp(clip.end_ms)}\n"
            f"{payload}"
        )
    return "WEBVTT\n\n" + "\n\n".join(cues) + "\n"


def build_webvtt_bytes(clips: tuple[Clip, ...]) -> bytes:
    return build_webvtt(clips).encode("utf-8")
