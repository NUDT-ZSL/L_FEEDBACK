"""Stable audit error codes and exit-code contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EXIT_AUDIT_CLEAN = 0
EXIT_AUDIT_COMPLETE_WITH_ERRORS = 1
EXIT_TOOL_FAILURE = 2
EXIT_STRICT_FAILURE = 3
EXIT_GUARD_FAILURE = 4

EXIT_CODES = {
    "audit_clean": EXIT_AUDIT_CLEAN,
    "guard_pass": EXIT_AUDIT_CLEAN,
    "audit_complete_with_errors": EXIT_AUDIT_COMPLETE_WITH_ERRORS,
    "tool_failure": EXIT_TOOL_FAILURE,
    "strict_failure": EXIT_STRICT_FAILURE,
    "guard_failure": EXIT_GUARD_FAILURE,
}

AUDIT_ERROR_CODES = {
    "E_INCLUDE_CYCLE",
    "E_MACRO_CONFLICT",
    "E_SNAPSHOT_CORRUPT",
    "E_TIMEOUT",
    "E_PARSE_LIMIT",
    "E_BASELINE_MISMATCH",
    "E_GUARD_FAILED",
    "E_ARGS",
    "E_IO",
    "E_CONFIG",
    "E_WORKER",
    "E_BASELINE_NOT_FOUND",
}


@dataclass(frozen=True)
class AuditError:
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message or self.code,
            "context": dict(self.context),
        }


class WorkbenchError(Exception):
    """An error that is safe to expose through the stable JSON contract."""

    def __init__(
        self,
        code: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error = AuditError(code=code, message=message or code, context=context or {})

    @property
    def code(self) -> str:
        return self.error.code

    @property
    def message(self) -> str:
        return self.error.message

    @property
    def context(self) -> dict[str, Any]:
        return self.error.context

    def to_dict(self) -> dict[str, Any]:
        return self.error.to_dict()


def error_dict(code: str, message: str, **context: Any) -> dict[str, Any]:
    return AuditError(code=code, message=message, context=context).to_dict()

