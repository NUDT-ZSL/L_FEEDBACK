"""Deterministic memory accounting for the large render buffers."""

from __future__ import annotations

import math

# float32 = 4 bytes per sample per channel.
FLOAT32_BYTES = 4
DEFAULT_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024


class MemoryGuard:
    """Tracks the high-water mark of allocated audio sample storage.

    Only the large, length-dependent buffers are counted.  The limit is
    checked before each allocation so that an oversized timeline fails with
    ``E_MEMORY_LIMIT`` instead of triggering an OS-level ``MemoryError``.
    """

    def __init__(self, limit_bytes: int = DEFAULT_MEMORY_LIMIT_BYTES) -> None:
        self.limit_bytes = int(limit_bytes)
        self.peak_bytes = 0

    def frame_cost(self, frames: int, channels: int = 2) -> int:
        return int(frames) * channels * FLOAT32_BYTES

    def check(self, extra_bytes: int) -> None:
        projected = self.peak_bytes + int(extra_bytes)
        self.peak_bytes = max(self.peak_bytes, projected)
        if projected > self.limit_bytes:
            from .errors import E_MEMORY_LIMIT, PipelineError

            raise PipelineError(
                E_MEMORY_LIMIT,
                {
                    "required_bytes": int(projected),
                    "limit_bytes": int(self.limit_bytes),
                },
            )

    def check_frames(self, frames: int, channels: int = 2) -> None:
        self.check(self.frame_cost(frames, channels))


def human_bytes(num_bytes: int) -> str:
    """Human-readable byte count for diagnostic messages."""
    units = ("B", "KiB", "MiB", "GiB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{math.floor(num_bytes)} B"
