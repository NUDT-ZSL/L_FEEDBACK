"""Application-level errors with stable process exit codes."""


class AppError(Exception):
    exit_code = 1


class ConfigError(AppError):
    """Invalid or missing scene configuration."""

    exit_code = 2


class RenderError(AppError):
    """Rendering or output failure."""

    exit_code = 4


class VerificationError(AppError):
    """Substantive difference found while verifying a reference report."""

    exit_code = 3
