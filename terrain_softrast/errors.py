class ValidationError(ValueError):
    """Raised before any output is written when an input is invalid."""


class RenderError(RuntimeError):
    """Raised when an otherwise valid scene cannot be rendered."""
