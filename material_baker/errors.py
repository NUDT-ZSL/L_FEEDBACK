"""Business errors and exit-code mapping."""

from __future__ import annotations


EXIT_CODES = {
    "E_MISSING_FILE": 2,
    "E_INVALID_FORMAT": 3,
    "E_INVALID_CONFIG": 4,
    "E_IO": 5,
    "E_INTERNAL": 1,
}


class ToolError(Exception):
    """An expected command-line failure with a stable business error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, 1)


def missing_file(message: str) -> ToolError:
    return ToolError("E_MISSING_FILE", message)


def invalid_format(message: str) -> ToolError:
    return ToolError("E_INVALID_FORMAT", message)


def invalid_config(message: str) -> ToolError:
    return ToolError("E_INVALID_CONFIG", message)


def io_error(message: str) -> ToolError:
    return ToolError("E_IO", message)

