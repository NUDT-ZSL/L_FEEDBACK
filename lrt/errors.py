"""Error types shared by the CLI, parser, and renderer."""

from __future__ import annotations


class SceneError(Exception):
    """A user-facing scene/input error with a stable process exit code."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)


ERROR_IO_OR_JSON = 2
ERROR_SCHEMA = 3
ERROR_GEOMETRY = 4
ERROR_MATERIAL_TYPE = 5
ERROR_LIGHTS = 6
ERROR_COATING_RANGE = 7
ERROR_RENDER_PARAM = 8
ERROR_OUTPUT = 9

