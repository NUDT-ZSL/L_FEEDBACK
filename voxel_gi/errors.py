"""Business errors with stable, machine-readable codes."""

from __future__ import annotations

from typing import Any, Dict, Optional


class ErrorCode:
    INVALID_JSON = "INVALID_JSON"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    INVALID_MESH_NAN = "INVALID_MESH_NAN"
    INVALID_MESH_DEGENERATE = "INVALID_MESH_DEGENERATE"
    INVALID_MATERIAL = "INVALID_MATERIAL"
    INVALID_LIGHT = "INVALID_LIGHT"
    INVALID_PROBE_SPACING = "INVALID_PROBE_SPACING"
    INVALID_PROBE_BOUNDS = "INVALID_PROBE_BOUNDS"
    INVALID_BAKE_CONFIG = "INVALID_BAKE_CONFIG"
    PROBE_INSIDE_OCCLUDER = "PROBE_INSIDE_OCCLUDER"
    CACHE_NOT_FOUND = "CACHE_NOT_FOUND"
    CACHE_BAD_MAGIC = "CACHE_BAD_MAGIC"
    CACHE_VERSION_UNSUPPORTED = "CACHE_VERSION_UNSUPPORTED"
    CACHE_CORRUPT = "CACHE_CORRUPT"
    CACHE_CHECKSUM_MISMATCH = "CACHE_CHECKSUM_MISMATCH"
    CACHE_GEOMETRY_MISMATCH = "CACHE_GEOMETRY_MISMATCH"
    CACHE_SCENE_MISMATCH = "CACHE_SCENE_MISMATCH"
    CACHE_LAYOUT_MISMATCH = "CACHE_LAYOUT_MISMATCH"
    IO_ERROR = "IO_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class BusinessError(Exception):
    """An expected failure that is safe to expose as structured JSON."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": "error",
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }
