"""Error/warning codes and the structured pipeline exception type."""

# Fatal error codes (any E_ code forces exit code 1).
E_TIMELINE_FORMAT = "E_TIMELINE_FORMAT"
E_MISSING_CLIP = "E_MISSING_CLIP"
E_RANGE = "E_RANGE"
E_CORRUPT_CLIP = "E_CORRUPT_CLIP"
E_MEMORY_LIMIT = "E_MEMORY_LIMIT"

# Non-fatal warning codes (exit code stays 0).
W_SHORT_CLIP = "W_SHORT_CLIP"
W_LOW_LOUDNESS = "W_LOW_LOUDNESS"

ERROR_CODES = frozenset(
    {
        E_TIMELINE_FORMAT,
        E_MISSING_CLIP,
        E_RANGE,
        E_CORRUPT_CLIP,
        E_MEMORY_LIMIT,
    }
)


class PipelineError(Exception):
    """A structured error carrying a stable machine-readable code.

    ``details`` is a JSON-serializable object describing the problem.
    """

    def __init__(self, code: str, details: object) -> None:
        super().__init__(f"{code}: {details}")
        self.code = code
        self.details = details
