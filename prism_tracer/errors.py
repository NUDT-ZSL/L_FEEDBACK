"""Business-error types and stable process exit codes."""

from __future__ import annotations


class BusinessError(Exception):
    """An expected user/scene/IO error with a stable machine-readable code."""

    def __init__(self, code: str, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code


ERROR_CODES = {
    "E_USAGE": 2,
    "E_SCENE": 10,
    "E_SCENE_FIELDS": 11,
    "E_SCENE_VALUE": 12,
    "E_RESOLUTION": 13,
    "E_SPP": 14,
    "E_SEED": 15,
    "E_TILE": 16,
    "E_DEPTH": 17,
    "E_CAUSTIC_PHOTONS": 18,
    "E_CAMERA_MEDIUM": 20,
    "E_IO": 21,
    "E_CHECKPOINT": 22,
}


def business_error(code: str, message: str) -> BusinessError:
    return BusinessError(code, message, ERROR_CODES[code])

