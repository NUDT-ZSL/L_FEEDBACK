"""Business errors used by both the CLI and the processing API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class ErrorCode:
    UNSUPPORTED_FORMAT = "E_UNSUPPORTED_FORMAT"
    INVALID_EVENT = "E_INVALID_EVENT"
    FILE_NOT_FOUND = "E_FILE_NOT_FOUND"
    OUTPUT_DENIED = "E_OUTPUT_DENIED"

    ALL = {
        UNSUPPORTED_FORMAT,
        INVALID_EVENT,
        FILE_NOT_FOUND,
        OUTPUT_DENIED,
    }


@dataclass
class BusinessError(Exception):
    """An expected user-facing failure with a stable business error code."""

    code: str
    message: str
    event_index: Optional[int] = None
    details: Optional[dict] = None

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def with_event_index(self, event_index: int) -> "BusinessError":
        if self.event_index is None:
            self.event_index = event_index
        return self

    def as_dict(self) -> dict:
        payload = {
            "code": self.code,
            "message": self.message,
            "event_index": self.event_index,
        }
        if self.details:
            payload["details"] = self.details
        return payload

    def __str__(self) -> str:
        location = (
            f" (event {self.event_index})"
            if self.event_index is not None
            else ""
        )
        return f"{self.code}{location}: {self.message}"
