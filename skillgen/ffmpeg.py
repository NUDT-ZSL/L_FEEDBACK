from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EncodeResult:
    returncode: int
    stderr_summary: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def _summary(stderr: bytes, temp_dir: Path, limit: int = 1200) -> str:
    text = stderr.decode("utf-8", errors="replace").strip()
    if not text:
        return ""
    text = text.replace(str(temp_dir), "<temp>")
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[:limit] + "...<truncated>"
    return text


def encode_pcm16_wav(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    sample_rate: int,
    temp_dir: Path,
    timeout_seconds: float = 180.0,
) -> EncodeResult:
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map_metadata",
        "-1",
        "-fflags",
        "+bitexact",
        "-flags:a",
        "+bitexact",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        str(output_path),
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr if isinstance(exc.stderr, bytes) else b""
        summary = _summary(stderr, temp_dir)
        if not summary:
            summary = f"ffmpeg timed out after {timeout_seconds:.0f}s"
        return EncodeResult(124, summary)
    except OSError as exc:
        return EncodeResult(127, f"failed to start ffmpeg: {exc}")
    summary = _summary(completed.stderr, temp_dir)
    return EncodeResult(completed.returncode, summary)
