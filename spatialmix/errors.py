"""Business error types and stable error codes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ErrorDefinition:
    code: str
    exit_code: int
    error_type: str


E_INPUT_FORMAT = ErrorDefinition("E_INPUT_FORMAT", 10, "input_format")
E_INPUT_READ = ErrorDefinition("E_INPUT_READ", 11, "input_read")
E_CONFIG_READ = ErrorDefinition("E_CONFIG_READ", 20, "config_read")
E_CONFIG_JSON = ErrorDefinition("E_CONFIG_JSON", 21, "config_json")
E_CONFIG_MISSING_FIELD = ErrorDefinition(
    "E_CONFIG_MISSING_FIELD", 22, "config_missing_field"
)
E_CONFIG_INVALID = ErrorDefinition("E_CONFIG_INVALID", 23, "config_invalid")
E_HRTF_VERSION = ErrorDefinition("E_HRTF_VERSION", 30, "hrtf_version")
E_OUTPUT_FORMAT = ErrorDefinition("E_OUTPUT_FORMAT", 40, "output_format")
E_BIT_DEPTH = ErrorDefinition("E_BIT_DEPTH", 41, "bit_depth")
E_MEMORY_LIMIT = ErrorDefinition("E_MEMORY_LIMIT", 50, "memory_limit")
E_STATE_INVALID = ErrorDefinition("E_STATE_INVALID", 60, "state_invalid")
E_STATE_MISMATCH = ErrorDefinition("E_STATE_MISMATCH", 61, "state_mismatch")
E_IO = ErrorDefinition("E_IO", 70, "io")
E_INTERRUPTED = ErrorDefinition("E_INTERRUPTED", 130, "interrupted")
E_INTERNAL = ErrorDefinition("E_INTERNAL", 99, "internal")


class BusinessError(Exception):
    """An expected failure that maps to a stable business error code."""

    def __init__(
        self,
        definition: ErrorDefinition,
        message: str,
        details: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.definition = definition
        self.message = message
        self.details = details or {}

    @property
    def code(self) -> str:
        return self.definition.code

    @property
    def exit_code(self) -> int:
        return self.definition.exit_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": False,
            "code": self.definition.code,
            "error_code": self.definition.code,
            "exit_code": self.definition.exit_code,
            "type": self.definition.error_type,
            "error_type": self.definition.error_type,
            "message": self.message,
            "detail": self.details,
            "details": self.details,
        }


class ProcessingInterrupted(Exception):
    """Raised after a committed block when a graceful stop is requested."""
