"""Structured error types exposed by the command-line interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MasteringError(Exception):
    code: str
    message: str
    exit_code: int = 2
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


def err(
    code: str,
    message: str,
    *,
    exit_code: int = 2,
    **details: Any,
) -> MasteringError:
    return MasteringError(code=code, message=message, exit_code=exit_code, details=details)
