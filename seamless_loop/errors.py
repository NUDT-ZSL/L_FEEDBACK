"""Exception hierarchy and the documented process exit codes."""

# Exit codes mandated by the specification.
EXIT_CONFIG = 2       # JSON project / configuration problem.
EXIT_FILE = 3         # Input file missing or unreadable / unsupported format.
EXIT_RANGE = 4        # Parameter range problem (loop region, fade length...).
EXIT_OVERWRITE = 5    # Output exists while overwrite is disabled.


class LoopProcessorError(Exception):
    """Base class for every processor error.

    Each concrete subclass carries the process exit code that the command line
    entry point must return when the error is raised.
    """

    exit_code = EXIT_CONFIG

    def __init__(self, message, exit_code=None):
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class ConfigError(LoopProcessorError):
    """The JSON project file is missing fields or holds invalid values."""

    exit_code = EXIT_CONFIG


class AudioFileError(LoopProcessorError):
    """An input audio file cannot be read or uses an unsupported format."""

    exit_code = EXIT_FILE


class ParameterRangeError(LoopProcessorError):
    """A numeric parameter falls outside the allowed range."""

    exit_code = EXIT_RANGE


class OverwriteError(LoopProcessorError):
    """The output file already exists and overwrite is disabled."""

    exit_code = EXIT_OVERWRITE
