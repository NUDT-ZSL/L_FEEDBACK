from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EXIT_CODES = {
    "E_CLI_USAGE": 2,
    "E_CONFIG": 2,
    "E_TIMELINE_READ": 2,
    "E_TIMELINE_JSON": 2,
    "E_TIMELINE_SCHEMA": 2,
    "E_TIMELINE_RANGE": 2,
    "E_SHOT_INVALID": 2,
    "E_EVENT_INVALID": 2,
    "E_SUBTITLE_INVALID": 2,
    "E_DIALOGUE_INVALID": 2,
    "E_DIALOGUE_UNFIT": 4,
    "E_AUDIO_READ": 3,
    "E_AUDIO_FORMAT": 3,
    "E_FFMPEG_NOT_FOUND": 3,
    "E_STRETCH_FAILED": 3,
    "E_DUCK_RAMP_UNSAFE": 3,
    "E_SAMPLE_GENERATION": 5,
    "E_OUTPUT_WRITE": 5,
    "E_STATE_WRITE": 5,
    "E_INTERRUPTED": 130,
    "E_INTERNAL": 70,
}


@dataclass
class BusinessError(Exception):
    code: str
    message: str
    entry_id: str | None = None
    path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    exit_code: int | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)

    @property
    def status(self) -> int:
        if self.exit_code is not None:
            return self.exit_code
        return EXIT_CODES.get(self.code, 1)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "entry_id": self.entry_id,
        }
        if self.path is not None:
            result["path"] = self.path
        if self.details:
            result["details"] = self.details
        return result

    @classmethod
    def from_other(
        cls,
        code: str,
        message: str,
        other: "BusinessError",
    ) -> "BusinessError":
        details = dict(other.details)
        return cls(
            code=code,
            message=message,
            entry_id=other.entry_id,
            path=other.path,
            details=details,
            exit_code=other.exit_code,
        )


def error_payload(command: str, exc: BusinessError) -> dict[str, Any]:
    return {
        "ok": False,
        "command": command,
        "error": exc.as_dict(),
    }

