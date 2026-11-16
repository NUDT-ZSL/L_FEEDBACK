"""Business errors and stable process exit codes."""


class MixerError(Exception):
    def __init__(self, code, message, details=None, exit_code=3):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
        self.exit_code = exit_code

    def as_json(self):
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }
