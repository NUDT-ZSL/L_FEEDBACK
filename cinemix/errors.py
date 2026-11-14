from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

EXIT_CODES = {
    "E_USAGE": 2,
    "E_CONFIG_JSON": 10,
    "E_CONFIG_SCHEMA": 11,
    "E_CONFIG_FIELD": 12,
    "E_INPUT_MISSING": 20,
    "E_INPUT_FORMAT": 21,
    "E_INPUT_SAMPLE_RATE": 22,
    "E_INPUT_SILENT": 23,
    "E_CROSSFADE_CONFLICT": 30,
    "E_LIMIT_UNREACHABLE": 40,
    "E_OUTPUT_WRITE": 50,
    "E_INTERNAL": 99,
}


@dataclass(frozen=True)
class Issue:
    level: str
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "level": self.level,
            "code": self.code,
            "message": self.message,
        }
        if self.context:
            payload["context"] = self.context
        return payload


class CinemixError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = context or {}

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, EXIT_CODES["E_INTERNAL"])

    def as_issue(self) -> Issue:
        return Issue("error", self.code, self.message, self.context)


class Diagnostics:
    def __init__(self) -> None:
        self.issues: list[Issue] = []

    def warning(
        self, code: str, message: str, **context: Any
    ) -> None:
        self.issues.append(Issue("warning", code, message, context))

    def info(self, code: str, message: str, **context: Any) -> None:
        self.issues.append(Issue("info", code, message, context))

    def emit(self, stream: Any | None = None) -> None:
        target = stream or sys.stderr
        for issue in self.issues:
            target.write(
                json.dumps(
                    issue.as_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
        target.flush()

    def as_list(self) -> list[dict[str, Any]]:
        return [issue.as_dict() for issue in self.issues]

