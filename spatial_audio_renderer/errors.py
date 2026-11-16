"""Application error types mapped to stable command-line exit codes."""


class ParameterError(Exception):
    """Invalid command-line argument or scene description (exit code 1)."""


class RenderRuntimeError(Exception):
    """I/O or rendering failure after parameters were accepted (exit code 2)."""


class TrackLoadError(Exception):
    """A source WAV could not be opened or decoded."""
