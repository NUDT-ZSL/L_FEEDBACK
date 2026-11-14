"""Structured validation errors and business exit codes."""

EXIT_OK = 0
EXIT_VALIDATION = 2
EXIT_IO = 3
EXIT_RESUME_REJECTED = 4
EXIT_INTERRUPTED = 10
EXIT_RUNTIME = 1

E_SCHEMA = "E_SCHEMA"
E_TICK_ORDER = "E_TICK_ORDER"
E_UNKNOWN_REF = "E_UNKNOWN_REF"
E_SAMPLE_RATE = "E_SAMPLE_RATE"
E_EVENT = "E_EVENT"
E_IO = "E_IO"
E_RESUME = "E_RESUME"


class StructuredError(Exception):
    """Error carrying a machine-readable structured object."""

    def __init__(self, code, message, path=None, index=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path
        self.index = index

    def to_obj(self):
        obj = {"code": self.code, "message": self.message}
        if self.path is not None:
            obj["path"] = self.path
        if self.index is not None:
            obj["event_index"] = self.index
        return obj


class ProjectValidationError(StructuredError):
    """Raised for schema / reference / timeline validation failures."""


def make_error(code, message, path=None, index=None):
    obj = {"code": code, "message": message}
    if path is not None:
        obj["path"] = path
    if index is not None:
        obj["event_index"] = index
    return obj
