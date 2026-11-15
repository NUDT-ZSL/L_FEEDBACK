"""Business error codes used in the JSON report."""


class Code:
    OK = "OK"
    E_MISSING_FIELD = "E_MISSING_FIELD"        # manifest entry misses a required field
    E_FILE_NOT_FOUND = "E_FILE_NOT_FOUND"      # source wav does not exist
    E_INVALID_RANGE = "E_INVALID_RANGE"        # start/end seconds outside the file
    E_UNSUPPORTED_FORMAT = "E_UNSUPPORTED_FORMAT"  # e.g. IEEE float wav
    E_UNSUPPORTED_LAYOUT = "E_UNSUPPORTED_LAYOUT"  # channel count not in {1, 2, 6}
    E_DECODE_ERROR = "E_DECODE_ERROR"          # unreadable / corrupt wav
    E_INTERNAL = "E_INTERNAL"                  # unexpected processing failure


class PipelineError(Exception):
    """Error carrying a business error code for the report."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
