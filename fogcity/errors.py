"""Structured business errors used by the command-line interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BusinessError(Exception):
    """An expected, user-facing failure.

    ``code`` is the stable business error code written to stderr.  ``exit_code``
    is the process exit status.  Context contains only JSON-serializable values.
    """

    code: str
    message: str
    exit_code: int
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "exit_code": self.exit_code,
                "context": self.context,
            },
        }


SCENE_NOT_FOUND = "E_SCENE_NOT_FOUND"
SCENE_IO = "E_SCENE_IO"
SCENE_JSON = "E_SCENE_JSON"
SCENE_FIELD = "E_SCENE_FIELD"
KEYFRAME_TIME_ORDER = "E_KEYFRAME_TIME_ORDER"
KEYFRAME_TIME_DUPLICATE = "E_KEYFRAME_TIME_DUPLICATE"
GL_CONTEXT = "E_GL_CONTEXT"
GL_RESOURCE = "E_GL_RESOURCE"
FRAME_RENDER = "E_FRAME_RENDER"
OUTPUT_WRITE = "E_OUTPUT_WRITE"
VERIFY_OUTPUT_MISSING = "E_VERIFY_OUTPUT_MISSING"
VERIFY_MISMATCH = "E_VERIFY_MISMATCH"
INTERNAL = "E_INTERNAL"

EXIT_SCENE_NOT_FOUND = 2
EXIT_SCENE_IO = 3
EXIT_SCENE_JSON = 4
EXIT_SCENE_FIELD = 10
EXIT_KEYFRAME_TIME_ORDER = 11
EXIT_KEYFRAME_TIME_DUPLICATE = 12
EXIT_GL_CONTEXT = 20
EXIT_GL_RESOURCE = 21
EXIT_FRAME_RENDER = 22
EXIT_OUTPUT_WRITE = 30
EXIT_VERIFY_OUTPUT_MISSING = 40
EXIT_VERIFY_MISMATCH = 41
EXIT_INTERNAL = 99


def scene_field_error(path: str, message: str) -> BusinessError:
    return BusinessError(
        SCENE_FIELD,
        message,
        EXIT_SCENE_FIELD,
        {"field": path},
    )

