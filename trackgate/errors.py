"""Structured business error codes and exit-code mapping.

Exit codes:
  0  success (snapshots written)
  2  deterministic business error (structured JSON payload emitted)
  1  unexpected internal failure
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_BUSINESS = 2


class GateError(Exception):
    """Base class for deterministic business errors."""

    code = "E_INTERNAL"
    exit_code = EXIT_BUSINESS

    def __init__(self, message: str = "", **context):
        super().__init__(message or self.code)
        self.message = message or self.code
        self.context = dict(context)

    def payload(self) -> dict:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "context": self.context,
            },
        }


class EmptySceneError(GateError):
    code = "E_EMPTY_SCENE"


class DegenerateFovError(GateError):
    code = "E_DEGENERATE_FOV"


class BadTileWidthError(GateError):
    code = "E_BAD_TILE_WIDTH"


class BadImageSizeError(GateError):
    code = "E_BAD_IMAGE_SIZE"


class StaleCacheError(GateError):
    code = "E_STALE_CACHE"
