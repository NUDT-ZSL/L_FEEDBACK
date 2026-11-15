"""Business error types and exit-code mapping."""

from __future__ import annotations

from typing import Any


EXIT_MATERIAL = 2
EXIT_GL = 3
EXIT_STATE = 4
EXIT_DIFF = 5
EXIT_TEXTURE = 6
EXIT_PERF = 7


class ValidationError(Exception):
    """An expected validation failure with a stable business code."""

    def __init__(
        self,
        code: str,
        exit_code: int,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.exit_code = exit_code
        self.message = message
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "exit_code": self.exit_code,
            "message": self.message,
            "details": self.details,
        }


def material_error(message: str, **details: Any) -> ValidationError:
    return ValidationError("E_MAT_001", EXIT_MATERIAL, message, details)


def gl_compile_error(message: str, **details: Any) -> ValidationError:
    return ValidationError("E_GL_001", EXIT_GL, message, details)


def gl_link_error(message: str, **details: Any) -> ValidationError:
    return ValidationError("E_GL_002", EXIT_GL, message, details)


def gl_context_error(message: str, **details: Any) -> ValidationError:
    return ValidationError("E_GL_003", EXIT_GL, message, details)


def state_error(message: str, **details: Any) -> ValidationError:
    return ValidationError("E_STATE_001", EXIT_STATE, message, details)


def diff_error(message: str, **details: Any) -> ValidationError:
    return ValidationError("E_DIFF_001", EXIT_DIFF, message, details)


def texture_error(message: str, **details: Any) -> ValidationError:
    return ValidationError("E_TEX_001", EXIT_TEXTURE, message, details)


def perf_error(message: str, **details: Any) -> ValidationError:
    return ValidationError("E_PERF_001", EXIT_PERF, message, details)

