"""Error and issue types shared by loading, baking, and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "context": self.context,
        }


class InputFatal(Exception):
    """An input problem that makes producing a new pack unsafe."""

    def __init__(
        self,
        code: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.issue = Issue(code=code, message=message, context=context or {})


class BakeFailure(Exception):
    """A failure while rendering or committing outputs."""

    def __init__(
        self,
        code: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.issue = Issue(code=code, message=message, context=context or {})

