"""Error types with stable process exit codes."""


class SoftRenderError(Exception):
    """Base error. ``exit_code`` is used as the process exit status."""

    exit_code = 3

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ParameterError(SoftRenderError):
    """Invalid command line parameter (threads, resolution, ...)."""

    exit_code = 3


class AssetError(SoftRenderError):
    """Missing or invalid asset (OBJ, material JSON, scene JSON, textures)."""

    exit_code = 4


class RenderError(SoftRenderError):
    """Failure during rendering."""

    exit_code = 5
