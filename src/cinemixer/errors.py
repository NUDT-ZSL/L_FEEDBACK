"""Business error codes used across the renderer.

Every business failure is represented by a stable string error code and a
non-zero integer exit status so callers can branch deterministically.
"""

from __future__ import annotations


class CineMixerError(Exception):
    """Exception carrying a structured business error code."""

    def __init__(self, code: str, message: str = "", exit_code: int = 1):
        super().__init__(message or code)
        self.code = code
        self.message = message or code
        self.exit_code = exit_code

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.code}: {self.message}"


# Exit code table (error code -> process exit status).
EXIT_CODES: dict[str, int] = {
    "E_PROJECT_SCHEMA": 2,
    "E_PROJECT_NOT_FOUND": 3,
    "E_TRACK_LOAD_FAIL": 4,
    "E_ALL_TRACKS_FAILED": 5,
    "E_STATE_CORRUPT": 6,
    "E_FFMPEG_MISSING": 7,
    "E_OUTPUT_FAIL": 8,
}

# E_TRACK_LOAD_FAIL is per-track and does not by itself terminate the process;
# it only terminates when every track failed (E_ALL_TRACKS_FAILED).
NON_FATAL_CODES = frozenset({"E_TRACK_LOAD_FAIL"})
