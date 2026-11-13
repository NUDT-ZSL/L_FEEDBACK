"""Business error codes. These never become process exit codes; the process
exit code is always 0 (success / graceful stop) or 2 (business error)."""
import json
import sys


class BakeError(Exception):
    code = "E_INTERNAL"

    def __init__(self, message, **extra):
        super().__init__(message)
        self.message = message
        self.extra = extra

    def payload(self):
        out = {"error": self.code, "message": self.message}
        out.update(self.extra)
        return out


class HeaderMismatch(BakeError):
    code = "E_HEADER_MISMATCH"


class HeightInvalid(BakeError):
    code = "E_HEIGHT_INVALID"


class MemoryLimit(BakeError):
    code = "E_MEMORY_LIMIT"


class ReconstructMismatch(BakeError):
    code = "E_RECONSTRUCT_MISMATCH"


class NonDeterministic(BakeError):
    code = "E_NONDETERMINISTIC"


class StateCorrupt(BakeError):
    code = "E_STATE_CORRUPT"


def emit_error(err: BakeError) -> int:
    """Print the business error as JSON on stderr and return exit code 2."""
    sys.stderr.write(json.dumps(err.payload(), ensure_ascii=False) + "\n")
    return 2
