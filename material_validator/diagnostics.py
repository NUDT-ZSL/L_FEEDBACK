from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass
class Diagnostic:
    severity: str
    code: str
    message: str
    path: str | None = None
    material: str | None = None
    channels: tuple[str, ...] = ()
    stage: str | None = None
    line: int | None = None
    column: int | None = None
    raw: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.severity == "error"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            result["path"] = self.path
        if self.material is not None:
            result["material"] = self.material
        if self.channels:
            result["channels"] = list(self.channels)
        if self.stage is not None:
            result["stage"] = self.stage
        if self.line is not None:
            result["line"] = self.line
        if self.column is not None:
            result["column"] = self.column
        if self.raw is not None:
            result["raw"] = self.raw
        if self.details:
            result["details"] = self.details
        return result


def has_errors(diagnostics: Iterable[Diagnostic]) -> bool:
    return any(item.severity == "error" for item in diagnostics)


def count_by_severity(diagnostics: Iterable[Diagnostic]) -> tuple[int, int]:
    errors = 0
    warnings = 0
    for item in diagnostics:
        if item.severity == "error":
            errors += 1
        elif item.severity == "warning":
            warnings += 1
    return errors, warnings


def stable_diagnostics(diagnostics: Iterable[Diagnostic]) -> list[Diagnostic]:
    return sorted(
        diagnostics,
        key=lambda item: (
            item.material or "",
            item.channels,
            item.path or "",
            item.stage or "",
            item.line if item.line is not None else -1,
            item.code,
            item.message,
        ),
    )
