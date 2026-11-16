"""Tool-specific exception types."""


class SceneValidationError(ValueError):
    """The scene JSON or one of its referenced assets is invalid."""


class ProcessingError(RuntimeError):
    """Audio processing or reporting failed after validation."""


class ParameterError(ValueError):
    """Command-line input or output parameters are invalid."""
