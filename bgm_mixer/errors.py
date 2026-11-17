"""Business errors and their stable process exit codes."""

from __future__ import annotations

from typing import Dict

E_INVALID_INPUT = "E_INVALID_INPUT"
E_RESAMPLE = "E_RESAMPLE"
E_MIX = "E_MIX"
E_IO = "E_IO"

ERROR_EXIT_CODES: Dict[str, int] = {
    E_INVALID_INPUT: 2,
    E_RESAMPLE: 3,
    E_MIX: 4,
    E_IO: 5,
}


class MixerError(Exception):
    """An expected failure that can be reported as a business error."""

    def __init__(self, code: str, message: str) -> None:
        if code not in ERROR_EXIT_CODES:
            raise ValueError(f"unknown business error code: {code}")
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def exit_code(self) -> int:
        return ERROR_EXIT_CODES[self.code]
