from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .errors import BusinessError, ProcessingError
from .models import Manifest, MediaInfo, VideoInfo
from .wav import probe_wav, validate_wav_against_manifest


def ensure_ffmpeg_tools() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        raise ProcessingError(
            3000,
            "FFmpeg tools are not available on PATH: " + ", ".join(missing),
        )


def probe_video(path: Path, reference_stream_index: int | None) -> VideoInfo:
    if not path.is_file():
        raise ProcessingError(3010, f"video container not found: {path}")
    try:
        import ffmpeg

        payload: dict[str, Any] = ffmpeg.probe(
            str(path), cmd=shutil.which("ffprobe") or "ffprobe"
        )
    except Exception as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", "replace")
        raise ProcessingError(
            3010,
            f"cannot probe video container: {path}: {str(detail or exc).strip()[:1000]}",
        ) from exc

    audio_streams: list[dict[str, Any]] = []
    for stream in payload.get("streams", []):
        if stream.get("codec_type") != "audio":
            continue
        audio_streams.append(
            {
                "index": int(stream["index"]),
                "codec_name": stream.get("codec_name"),
                "sample_rate": _optional_int(stream.get("sample_rate")),
                "channels": _optional_int(stream.get("channels")),
                "duration_seconds": _optional_float(stream.get("duration")),
            }
        )

    selected: int | None
    if reference_stream_index is not None:
        valid_indices = {stream["index"] for stream in audio_streams}
        if reference_stream_index not in valid_indices:
            raise BusinessError(
                1008,
                f"video.reference_stream_index={reference_stream_index} does not identify an audio stream",
            )
        selected = reference_stream_index
    else:
        selected = audio_streams[0]["index"] if audio_streams else None

    duration = _optional_float(payload.get("format", {}).get("duration"))
    if duration is None and audio_streams:
        duration = audio_streams[0].get("duration_seconds")
    return VideoInfo(
        path=path,
        duration_seconds=duration,
        audio_streams=audio_streams,
        selected_reference_index=selected,
    )


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe_all_media(manifest: Manifest) -> MediaInfo:
    wavs = {}
    for language in manifest.languages:
        info = probe_wav(language.path, validate_alignment=False)
        validate_wav_against_manifest(
            info,
            sample_rate=manifest.audio.sample_rate,
            channels=manifest.audio.channels,
            bit_depth=manifest.audio.bit_depth,
        )
        wavs[language.language_id] = info
    video = probe_video(manifest.video_path, manifest.reference_stream_index)
    return MediaInfo(wavs=wavs, video=video)

