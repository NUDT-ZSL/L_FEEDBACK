class BusinessError(Exception):
    """Error carrying a stable business error code for CLI reporting."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


ERR_MISSING_FILE = "ERR_MISSING_FILE"
ERR_BAD_SAMPLE_RATE = "ERR_BAD_SAMPLE_RATE"
ERR_INVALID_BPM = "ERR_INVALID_BPM"
