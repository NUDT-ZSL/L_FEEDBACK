"""Business errors and their stable process exit codes."""


class AudioToolError(Exception):
    code = "ERR_AUDIO_TOOL"
    exit_code = 1

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


class LowConfidenceError(AudioToolError):
    code = "ERR_LOW_CONFIDENCE"
    exit_code = 1


class BadFormatError(AudioToolError):
    code = "ERR_BAD_FORMAT"
    exit_code = 2

