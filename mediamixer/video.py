"""Video decoding, frame timestamping and frame-marker overlay rendering."""

from __future__ import annotations

import zlib

import numpy as np

from .errors import ProcessError
from .ffmpeg_utils import decode_video_rgb, frame_timestamp_us
from .project import Marker

RECT_WIDTH = 40
RECT_HEIGHT = 24


def marker_color(marker: Marker) -> tuple[int, int, int]:
    """Deterministic RGB color derived from a checksum of the marker fields."""
    checksum = zlib.crc32(f"{marker.frame}:{marker.drift_ms}".encode("utf-8"))
    return (
        (checksum >> 16) & 0xFF,
        (checksum >> 8) & 0xFF,
        checksum & 0xFF,
    )


def marker_rect(marker: Marker, width: int, height: int) -> tuple[int, int, int, int]:
    """Return clamped ``(x0, y0, x1, y1)``; 2 px shift per millisecond drift."""
    center_x = width // 2 + marker.drift_ms * 2
    x0 = max(0, min(center_x - RECT_WIDTH // 2, width - RECT_WIDTH))
    y0 = max(0, (height - RECT_HEIGHT) // 2)
    return x0, y0, x0 + RECT_WIDTH, y0 + RECT_HEIGHT


def render_markers(frame: np.ndarray, markers: tuple[Marker, ...]) -> None:
    height, width, _ = frame.shape
    for marker in markers:
        x0, y0, x1, y1 = marker_rect(marker, width, height)
        frame[y0:y1, x0:x1, :] = marker_color(marker)


def process_video(
    video_path,
    width: int,
    height: int,
    frame_rate,
    frame_count: int,
    markers: tuple[Marker, ...],
    encoder,
) -> list[int]:
    """Decode RGB frames, stamp timestamps, draw markers and pipe to the encoder.

    Returns the per-frame display timestamps in microseconds.
    """
    frame_size = width * height * 3
    process = decode_video_rgb(video_path)
    stdout = process.stdout
    timestamps: list[int] = []
    marker_by_frame: dict[int, tuple[Marker, ...]] = {}
    for marker in markers:
        if 0 <= marker.frame < frame_count:
            marker_by_frame.setdefault(marker.frame, ())
            marker_by_frame[marker.frame] = marker_by_frame[marker.frame] + (marker,)

    try:
        leftover = bytearray()
        frame_index = 0
        while True:
            chunk = stdout.read(frame_size * 8)
            if not chunk:
                break
            leftover.extend(chunk)
            while len(leftover) >= frame_size:
                if frame_index >= frame_count:
                    raise ProcessError(
                        "decoded more frames than ffprobe reported; data mismatch"
                    )
                raw = bytes(leftover[:frame_size])
                del leftover[:frame_size]
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                    height, width, 3
                ).copy()
                if frame_index in marker_by_frame:
                    render_markers(frame, marker_by_frame[frame_index])
                timestamps.append(frame_timestamp_us(frame_index, frame_rate))
                encoder.stdin.write(frame.tobytes())
                frame_index += 1

        if leftover:
            raise ProcessError(
                f"trailing {len(leftover)} bytes do not form a complete RGB frame"
            )
        if frame_index != frame_count:
            raise ProcessError(
                f"decoded {frame_index} frames but expected {frame_count}"
            )
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        stderr_tail = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
        if return_code != 0:
            process.kill()
            raise ProcessError(
                "ffmpeg video decoder failed: "
                + stderr_tail.decode("utf-8", "replace").strip()[-500:]
            )
    return timestamps
