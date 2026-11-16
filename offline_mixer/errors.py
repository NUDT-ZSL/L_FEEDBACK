"""Business errors used by the command-line application."""

from __future__ import annotations


class MixerError(Exception):
    """Base class for errors that have a stable business error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def as_stderr(self) -> str:
        return f"{self.code}: {self.message}"


class ManifestError(MixerError):
    """Invalid or unreadable input manifest."""


class ParameterError(ManifestError):
    """A parameter is outside its declared range or has an invalid type."""


class ToolchainError(MixerError):
    """A required local executable is unavailable."""


class OutputError(MixerError):
    """Output could not be created."""


class AllTracksUnreadable(MixerError):
    """No input track could be decoded and converted."""
