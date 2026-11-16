"""Business error hierarchy and process exit codes.

Every recoverable failure inside the build pipeline raises
:class:`BuildError` carrying a stable integer code.  The CLI layer converts
the exception into a JSON document on stderr and the matching exit code so
that external tooling can rely on a machine-readable contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


class ExitCode:
    """Stable integer exit codes documented in ``docs/TECHNICAL.md``."""

    OK = 0
    CHART_SYNTAX = 2
    MISSING_SOURCE = 3
    UNSUPPORTED_FORMAT = 4
    MAPPING_CONFLICT = 5
    DURATION_MISMATCH = 6
    LOCKED_OUTPUT = 7
    ARGUMENT_ERROR = 8
    INTERNAL_ERROR = 1


@dataclass
class BuildError(Exception):
    """Raised for every expected, business-level failure.

    Attributes:
        code: One of the :class:`ExitCode` integer constants.
        message: Human readable explanation, safe to show to chart authors.
        details: Optional structured context (paths, beat positions, ...).
    """

    code: int
    message: str
    details: dict | None = None

    def __str__(self) -> str:
        return f"[E{self.code}] {self.message}"

    def to_dict(self) -> dict:
        payload = {"ok": False, "error": {"code": self.code, "message": self.message}}
        if self.details:
            payload["error"]["details"] = self.details
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
