from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ErrorCode:
    """Stable business codes written to diagnostics JSON."""

    SCENE_JSON_INVALID = "SCENE_JSON_INVALID"
    SCENE_FIELD_MISSING = "SCENE_FIELD_MISSING"
    SCENE_FIELD_INVALID = "SCENE_FIELD_INVALID"
    SCENE_ID_COLLISION = "SCENE_ID_COLLISION"
    UV_LAYOUT_INVALID = "UV_LAYOUT_INVALID"
    CAMERA_INVALID = "CAMERA_INVALID"

    MATERIAL_FIELD_MISSING = "MATERIAL_FIELD_MISSING"
    MATERIAL_LAYER_INVALID = "MATERIAL_LAYER_INVALID"
    MAT_MASK_SIZE_MISSING = "MAT_MASK_SIZE_MISSING"
    MAT_MASK_SIZE_MISMATCH = "MAT_MASK_SIZE_MISMATCH"
    MAT_BLEND_UNSUPPORTED = "MAT_BLEND_UNSUPPORTED"
    MAT_PATTERN_UNSUPPORTED = "MAT_PATTERN_UNSUPPORTED"
    MAT_PATTERN_FIELD_MISSING = "MAT_PATTERN_FIELD_MISSING"

    LIGHT_FIELD_MISSING = "LIGHT_FIELD_MISSING"
    LIGHT_VALUE_INVALID = "LIGHT_VALUE_INVALID"
    TONE_MAP_OPERATOR_UNSUPPORTED = "TONE_MAP_OPERATOR_UNSUPPORTED"

    CONSISTENCY_UV_MISMATCH = "CONSISTENCY_UV_MISMATCH"
    CONSISTENCY_METALLIC_JUMP = "CONSISTENCY_METALLIC_JUMP"
    CONSISTENCY_ENERGY_VIOLATION = "CONSISTENCY_ENERGY_VIOLATION"

    IO_OUTPUT_FAILED = "IO_OUTPUT_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class PipelineError(Exception):
    code: str
    message: str
    scope: str
    path: str = ""
    context: dict[str, Any] | None = None

    def as_diagnostic(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": "error",
            "scope": self.scope,
            "path": self.path,
            "message": self.message,
            "context": self.context or {},
        }


def diagnostic(
    code: str,
    message: str,
    scope: str,
    path: str = "",
    **context: Any,
) -> dict[str, Any]:
    """Build a diagnostic in a single stable key order."""

    return {
        "code": code,
        "severity": "error",
        "scope": scope,
        "path": path,
        "message": message,
        "context": context,
    }
