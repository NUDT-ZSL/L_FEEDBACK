"""ffprobe based metadata extraction."""

from __future__ import annotations

import json
import subprocess

FFPROBE = "ffprobe"


class ProbeError(Exception):
    def __init__(self, message: str, stderr: str = ""):
        super().__init__(message)
        self.stderr = stderr


def probe(path: str, timeout: float = 30.0) -> dict:
    """Return structured metadata for one media file.

    Raises ProbeError (with the raw stderr excerpt) when ffprobe cannot
    identify the file or exits non-zero.
    """
    cmd = [
        FFPROBE, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ProbeError("ffprobe timeout", "ffprobe timeout") from exc
    if proc.returncode != 0:
        raise ProbeError("ffprobe failed", proc.stderr.strip())
    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ProbeError("ffprobe returned invalid JSON", proc.stderr.strip()) from exc

    streams = [s for s in info.get("streams", []) if s.get("codec_type") == "audio"]
    if not streams:
        raise ProbeError("no audio stream", proc.stderr.strip())
    stream = streams[0]
    fmt = info.get("format", {})

    def _f(value, default=None):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    bit_depth = stream.get("bits_per_raw_sample") or stream.get("bits_per_sample") or 0
    try:
        bit_depth = int(bit_depth)
    except (TypeError, ValueError):
        bit_depth = 0

    duration = _f(stream.get("duration"), _f(fmt.get("duration")))
    start_time = _f(stream.get("start_time"), _f(fmt.get("start_time"), 0.0))

    return {
        "sample_rate": int(stream.get("sample_rate") or 0),
        "bit_depth": bit_depth,
        "codec": stream.get("codec_name", ""),
        "channels": int(stream.get("channels") or 0),
        "channel_layout": stream.get("channel_layout") or f"{stream.get('channels', 0)}ch",
        "duration_s": round(duration, 6) if duration is not None else None,
        "start_time_ms": round((start_time or 0.0) * 1000.0, 3),
        "format_name": fmt.get("format_name", ""),
    }
