"""Business error types used by the command-line tool and reports."""

from __future__ import annotations

from typing import Any, Dict, Optional


ERROR_EXIT_CODES: Dict[str, int] = {
    "E_CONFIG_MISSING": 2,
    "E_CONFIG_INVALID_JSON": 2,
    "E_CONFIG_SCHEMA": 2,
    "E_OUTPUT_DIR": 3,
    "E_INPUT_MISSING": 4,
    "E_AUDIO_READ": 4,
    "E_CHANNEL_MISMATCH": 4,
    "E_NO_DIALOGUE": 5,
    "E_FFMPEG_NOT_FOUND": 6,
    "E_FFMPEG_FAILED": 6,
    "E_FFPROBE_FAILED": 6,
    "E_DURATION_MISMATCH": 6,
    "E_OUTPUT_WRITE": 7,
    "E_INTERNAL": 1,
}


class MasterError(Exception):
    """An expected, reportable processing failure."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.exit_code = ERROR_EXIT_CODES.get(code, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


def internal_error(exc: BaseException) -> MasterError:
    return MasterError(
        "E_INTERNAL",
        "Unexpected internal failure",
        {"exception_type": type(exc).__name__, "exception": str(exc)},
    )

