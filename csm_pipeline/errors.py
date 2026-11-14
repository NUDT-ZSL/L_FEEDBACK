"""Fixed error codes and the pipeline exception type.

Error codes are part of the JSON contract and must never be renamed so that
downstream tooling can rely on them.
"""

from __future__ import annotations


ERR_CONFIG = "E_CONFIG"
ERR_PATH_MISSING = "E_PATH_MISSING"
ERR_PATH_FORMAT = "E_PATH_FORMAT"
ERR_ARCHIVE_MISSING = "E_ARCHIVE_MISSING"
ERR_ARCHIVE_HEADER = "E_ARCHIVE_HEADER"
ERR_ARCHIVE_TRUNCATED = "E_ARCHIVE_TRUNCATED"
ERR_ARCHIVE_DTYPE = "E_ARCHIVE_DTYPE"
ERR_INVALID_MATRIX = "E_INVALID_MATRIX"
ERR_INVALID_DEPTH = "E_INVALID_DEPTH"
ERR_IO = "E_IO"
ERR_UNKNOWN = "E_UNKNOWN"

# Stable process exit codes mapped from the error codes above.
EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_CONFIG = 2
EXIT_ARCHIVE = 3
EXIT_INVALID_DATA = 4

_EXIT_BY_CODE = {
    ERR_CONFIG: EXIT_CONFIG,
    ERR_PATH_MISSING: EXIT_CONFIG,
    ERR_PATH_FORMAT: EXIT_CONFIG,
    ERR_ARCHIVE_MISSING: EXIT_ARCHIVE,
    ERR_ARCHIVE_HEADER: EXIT_ARCHIVE,
    ERR_ARCHIVE_TRUNCATED: EXIT_ARCHIVE,
    ERR_ARCHIVE_DTYPE: EXIT_ARCHIVE,
    ERR_INVALID_MATRIX: EXIT_INVALID_DATA,
    ERR_INVALID_DEPTH: EXIT_INVALID_DATA,
    ERR_IO: EXIT_GENERIC,
    ERR_UNKNOWN: EXIT_GENERIC,
}


class PipelineError(Exception):
    """Raised for every expected, reportable pipeline failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    @property
    def exit_code(self) -> int:
        return _EXIT_BY_CODE.get(self.code, EXIT_GENERIC)

