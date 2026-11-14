from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BakeError(Exception):
    business_code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    recovery_suggestions: list[str] = field(default_factory=list)
    exit_code: int = 2

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def as_report(self, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema": "offline-pbr-bake-error/v1",
            "status": "error",
            "business_code": self.business_code,
            "exit_code": self.exit_code,
            "message": self.message,
            "context": self.context,
            "recovery_suggestions": self.recovery_suggestions,
            "parameters": parameters or {},
        }

