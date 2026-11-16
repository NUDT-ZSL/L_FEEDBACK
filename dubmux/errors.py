"""Stable business error codes and process exit codes."""

# Business error codes (stable, written to report and process output).
E_INPUT_UNREADABLE = "E1001"      # input file missing / undecodable
E_SEGMENT_MISSING = "E1002"       # dub has fewer dialogue segments than reference
E_SEGMENT_EXTRA = "E1003"         # dub has more dialogue segments than reference
E_SUBTITLE_OVERFLOW = "E1004"     # subtitle entry beyond video total duration
E_SAMPLE_RATE_MISMATCH = "E1005"  # dub sample rate differs from required rate
E_STRETCH_TOLERANCE = "E1006"     # aligned boundary exceeds configured tolerance
E_FFMPEG_FAILURE = "E1007"        # ffmpeg/ffprobe invocation failed

# Process exit codes.
EXIT_OK = 0        # every language succeeded
EXIT_PARTIAL = 1   # at least one language failed, at least one succeeded
EXIT_FATAL = 2     # all languages failed, or input/config error, or aborted


class DubError(Exception):
    """Business error carrying a stable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message}
