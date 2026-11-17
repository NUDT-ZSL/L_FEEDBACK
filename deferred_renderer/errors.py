from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RenderError(Exception):
    """A stable, machine-readable renderer failure."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    exit_code: int = 2

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }

