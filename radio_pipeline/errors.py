class PipelineError(Exception):
    """Base error for expected pipeline failures."""


class ConfigError(PipelineError):
    """The supplied configuration or command line is invalid."""


class ProcessingError(PipelineError):
    """The pipeline could not complete processing."""


class VerificationFailure(PipelineError):
    """An output failed a verification check."""

