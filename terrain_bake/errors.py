"""Business errors used by both the CLI and library API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_INVALID_ARGUMENT = "E_INVALID_ARGUMENT"
E_INVALID_HEIGHTFIELD = "E_INVALID_HEIGHTFIELD"
E_WEIGHT_MISMATCH = "E_WEIGHT_MISMATCH"
E_GEOMETRY_INVALID = "E_GEOMETRY_INVALID"
E_RENDER_FAILED = "E_RENDER_FAILED"
E_IO_ERROR = "E_IO_ERROR"
E_INTERNAL_ERROR = "E_INTERNAL_ERROR"

ERROR_EXIT_CODES = {
    E_INVALID_ARGUMENT: 2,
    E_INVALID_HEIGHTFIELD: 3,
    E_WEIGHT_MISMATCH: 4,
    E_GEOMETRY_INVALID: 5,
    E_RENDER_FAILED: 6,
    E_IO_ERROR: 7,
    E_INTERNAL_ERROR: 8,
}


@dataclass
class BakeError(Exception):
    """A predictable pipeline failure with a stable business error code."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }

