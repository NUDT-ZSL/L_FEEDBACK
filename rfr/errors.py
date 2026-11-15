"""Structured business error codes used across the renderer.

Every failure that should be reported on the command line raises
:class:`RenderError` with one of the stable codes below.  The CLI catches the
error, emits a single structured JSON line on stderr and exits non-zero while
preserving whatever statistics have already been gathered.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


ERROR_CODES = {
    "E_SCENE_INVALID": "Scene file is missing, unreadable or malformed.",
    "E_SCENE_MESH": "A node references a mesh that is not defined.",
    "E_MESH_INVALID": "Mesh geometry is missing or has inconsistent arrays.",
    "E_BACKEND_UNAVAILABLE": "The requested render backend could not initialise.",
    "E_SHADER_COMPILE": "A GPU shader program failed to compile or link.",
    "E_SIZE_MISMATCH": "Framebuffer or geometry dimensions do not match.",
    "E_DIMENSION": "A matrix or vector has an unsupported dimensionality.",
    "E_RENDER": "An unspecified failure happened while rendering a frame.",
    "E_IO": "An image or statistics file could not be written.",
}


class RenderError(Exception):
    """An error carrying a stable business ``code`` and optional context."""

    def __init__(
        self,
        code: str,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"Unknown error code: {code}")
        self.code = code
        self.context = dict(context or {})
        full_message = message or ERROR_CODES[code]
        super().__init__(full_message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": True,
            "code": self.code,
            "message": str(self),
            "context": self.context,
        }
