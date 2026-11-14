"""Structured pipeline errors with stable error codes."""

from . import EXIT_ERROR


class PipelineError(Exception):
    """Error carrying a machine-readable ``error_code``."""

    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        if error_code not in EXIT_ERROR:
            raise ValueError(f"unknown error code: {error_code}")
        self.error_code = error_code
        self.message = message

    @property
    def exit_code(self) -> int:
        return EXIT_ERROR[self.error_code]

    def to_dict(self) -> dict:
        return {"error_code": self.error_code, "message": self.message}
