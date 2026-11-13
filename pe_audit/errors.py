"""Error types and stable process exit codes used by the command line tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EXIT_OK = 0
EXIT_IO = 2
EXIT_MANIFEST = 3
EXIT_BOUNDS = 4
EXIT_RULE = 5


@dataclass
class AuditError(Exception):
    """A machine-readable, user-facing input failure."""

    code: str
    message: str
    exit_code: int
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            error["details"] = self.details
        return {"schema_version": 1, "error": error}


def bounds_error(message: str, **details: Any) -> AuditError:
    return AuditError("BOUNDS_OVERFLOW", message, EXIT_BOUNDS, dict(details))


def manifest_error(message: str, **details: Any) -> AuditError:
    return AuditError("MANIFEST_INVALID", message, EXIT_MANIFEST, dict(details))


def rule_error(message: str, **details: Any) -> AuditError:
    return AuditError("RULE_INVALID", message, EXIT_RULE, dict(details))


def binary_error(message: str, **details: Any) -> AuditError:
    return AuditError("BINARY_PARSE_ERROR", message, EXIT_IO, dict(details))
