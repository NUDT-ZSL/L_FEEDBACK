"""Domain exceptions and process exit codes.

Exit codes are plain integers so ``sys.exit(code)`` follows Windows
process semantics (a 32-bit signed integer exit status).
"""

EXIT_OK = 0
EXIT_BAD_ARGS = 2
EXIT_BAKE_FAILED = 3
EXIT_IO_FAILED = 4
EXIT_PREVIEW_UNAVAILABLE = 6


class BakerError(Exception):
    """Base class for all baker failures."""

    exit_code = EXIT_BAKE_FAILED


class ParameterError(BakerError):
    """An input parameter failed validation."""

    exit_code = EXIT_BAD_ARGS


class IOFailure(BakerError):
    """A filesystem write or atomic commit failed."""

    exit_code = EXIT_IO_FAILED


class PreviewUnavailable(BakerError):
    """No usable offscreen software rendering backend could be created."""

    exit_code = EXIT_PREVIEW_UNAVAILABLE


class MemoryBudgetExceeded(BakerError):
    """Active working data exceeded the configured memory budget."""

    exit_code = EXIT_BAKE_FAILED
