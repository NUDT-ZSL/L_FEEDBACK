"""Typed errors used by the command line and the per-entry report."""

from __future__ import annotations


class DialogueSyncError(Exception):
    """Base class for fatal command errors."""

    code = "E_INTERNAL"
    exit_code = 2

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class ConfigError(DialogueSyncError):
    code = "E_CONFIG"


class TimecodeError(DialogueSyncError):
    code = "E_TIMECODE"


class SrtError(DialogueSyncError):
    code = "E_SRT"


class ExternalToolError(DialogueSyncError):
    code = "E_EXTERNAL"


class StateError(DialogueSyncError):
    code = "E_STATE"


class StateIntegrityError(StateError):
    code = "E_STATE_INTEGRITY"


class EntryError(Exception):
    """A recoverable error that is confined to one dialogue entry."""

    def __init__(self, code: str, message: str, *, warning: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.warning = warning

