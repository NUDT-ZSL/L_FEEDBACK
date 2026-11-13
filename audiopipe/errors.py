"""Business error codes and a single JSON-on-stderr reporting convention."""

# Exit / business codes
EXIT_OK = 0
EXIT_VALIDATION = 2
EXIT_FATAL_CLIP = 3
EXIT_INTERRUPTED = 130
EXIT_USAGE = 64
EXIT_UNEXPECTED = 70


class AudioPipeError(Exception):
    """Base class carrying a stable machine-readable business code."""

    code = "E_INTERNAL"
    exit_code = EXIT_UNEXPECTED

    def __init__(self, message, code=None, exit_code=None, details=None):
        super().__init__(message)
        if code is not None:
            self.code = code
        if exit_code is not None:
            self.exit_code = exit_code
        self.details = details or {}

    def to_record(self):
        rec = {"code": self.code, "message": str(self)}
        if self.details:
            rec["details"] = self.details
        return rec


class WavFormatError(AudioPipeError):
    """Fatal structural/audio-format error for a single WAV file."""

    code = "E_WAV_FORMAT"
    exit_code = EXIT_VALIDATION


class CheckpointVersionError(AudioPipeError):
    code = "E_CHECKPOINT_VERSION"
    exit_code = EXIT_VALIDATION


class InterruptedError2(AudioPipeError):
    code = "E_INTERRUPTED"
    exit_code = EXIT_INTERRUPTED


def emit_error(err):
    """Serialize an error record as a single JSON line on stderr."""
    import json
    import sys

    record = err.to_record() if isinstance(err, AudioPipeError) else {
        "code": "E_INTERNAL",
        "message": str(err),
    }
    sys.stderr.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    sys.stderr.flush()
