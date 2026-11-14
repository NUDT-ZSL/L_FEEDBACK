"""Stable business error codes and process exit statuses."""

from __future__ import annotations

from typing import Dict


class PipelineError(Exception):
    """A user-facing pipeline failure with a stable business error code."""

    EXIT_CODES: Dict[str, int] = {
        "E_USAGE": 2,
        "E_FORMAT": 3,
        "E_EMPTY": 4,
        "E_SAMPLERATE": 5,
        "E_WRITE": 6,
        "E_SILENT": 7,
        "E_INTERNAL": 8,
    }

    def __init__(self, code: str, message: str) -> None:
        if code not in self.EXIT_CODES:
            raise ValueError("unknown pipeline error code: {0}".format(code))
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def exit_code(self) -> int:
        return self.EXIT_CODES[self.code]

    def __str__(self) -> str:
        return "[{0}] {1}".format(self.code, self.message)

