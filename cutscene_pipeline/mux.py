"""FFmpeg-based MP4 muxing: generated color video + normalized audio + chapters."""

from __future__ import annotations

import subprocess
from pathlib import Path


class MuxError(Exception):
    pass


def _write_ffmetadata(path: Path, chapters: list, total_duration: float) -> None:
    lines = [";FFMETADATA1"]
    for i, ch in enumerate(chapters):
        start_ms = int(round(ch["start"] * 1000.0))
        if i + 1 < len(chapters):
            end_ms = int(round(chapters[i + 1]["start"] * 1000.0))
        else:
            end_ms = int(round(total_duration * 1000.0))
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        title = ch["title"].replace("\n", " ").replace("=", "-").replace(";", ",")
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={start_ms}",
                  f"END={end_ms}", f"title={title}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mux_mp4(audio_wav: Path, out_mp4: Path, video_spec: dict, duration: float,
            chapters: list, work_dir: Path) -> dict:
    """Generate a solid-color video track, mux with audio and chapter marks."""
    meta = work_dir / "chapters.ffmeta"
    _write_ffmetadata(meta, chapters, duration)
    width = int(video_spec["width"])
    height = int(video_spec["height"])
    fps = video_spec["fps"]
    color = str(video_spec.get("color", "0x000000"))
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i",
        f"color=c={color}:s={width}x{height}:r={fps}:d={duration:.6f}",
        "-i", str(audio_wav),
        "-i", str(meta),
        "-map", "0:v", "-map", "1:a",
        "-map_metadata", "2", "-map_chapters", "2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-t", f"{duration:.6f}",
        str(out_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MuxError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()}")
    return {"output": str(out_mp4), "chapters_written": len(chapters),
            "video": {"width": width, "height": height, "fps": fps, "color": color},
            "duration_s": round(duration, 6)}
