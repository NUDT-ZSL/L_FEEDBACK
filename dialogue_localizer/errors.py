"""Business errors and their stable JSON representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EXIT_INTERNAL = 1
EXIT_MANIFEST = 2
EXIT_AUDIO = 3
EXIT_TIMELINE = 4
EXIT_OUTPUT = 5


@dataclass
class BusinessError(Exception):
    code: str
    message: str
    exit_code: int
    file: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "error",
            "code": self.code,
            "exit_code": self.exit_code,
            "message": self.message,
            "file": self.file,
        }
        if self.details:
            payload["details"] = self.details
        return payload


def warning(
    code: str,
    message: str,
    *,
    segment_id: str | None = None,
    **details: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "warning",
        "code": code,
        "message": message,
    }
    if segment_id is not None:
        payload["segment_id"] = segment_id
    if details:
        payload["details"] = details
    return payload

