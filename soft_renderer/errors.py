class RenderError(Exception):
    """Base class for errors that should be shown without a traceback."""


class SceneError(RenderError):
    """The scene file is malformed or contains invalid values."""


class UsageError(RenderError):
    """Command-line arguments are invalid."""


class OutputError(RenderError):
    """An output could not be written atomically."""
