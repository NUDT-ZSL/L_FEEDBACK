"""Business error codes and process exit codes.

Every business failure is reported as a JSON report entry carrying one of
these codes, and mapped to a stable non-zero process exit code.
"""


class ExitCode:
    OK = 0
    INTERNAL = 1
    MANIFEST_INVALID = 2
    INPUT_MISSING = 3
    RENDER_ERROR = 4
    VERSION_CONFLICT = 5


class ErrorCode:
    E_INTERNAL = "E_INTERNAL"
    E_MANIFEST_INVALID = "E_MANIFEST_INVALID"
    E_INPUT_MISSING = "E_INPUT_MISSING"
    E_UNSUPPORTED_FORMAT = "E_UNSUPPORTED_FORMAT"
    E_FADE_NEGATIVE = "E_FADE_NEGATIVE"
    E_FADE_TOO_LONG = "E_FADE_TOO_LONG"
    E_OVERLAP_EXCESSIVE = "E_OVERLAP_EXCESSIVE"
    E_STAGE_VERSION_CONFLICT = "E_STAGE_VERSION_CONFLICT"
    E_STATE_LOCK = "E_STATE_LOCK"


class BusinessError(Exception):
    """An expected, user-facing failure with a stable business code."""

    def __init__(self, code, message, exit_code=ExitCode.RENDER_ERROR, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.details = details or {}

    def to_dict(self):
        return {"code": self.code, "message": self.message, "details": self.details}
