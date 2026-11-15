"""Structured input validation errors."""


class RenderError(Exception):
    """Raised for invalid renderer input.

    ``param`` is the canonical parameter name that fails validation so the CLI
    can emit the required single-line JSON payload on stderr.
    """

    def __init__(self, param: str, message: str):
        super().__init__(message)
        self.param = param
        self.message = message

    def to_payload(self) -> dict:
        return {
            "error_code": "E_INVALID_INPUT",
            "param": self.param,
            "message": self.message,
        }
