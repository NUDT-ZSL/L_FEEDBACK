"""Structured business errors used by the synchronization core."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class ErrorCode(str, Enum):
    """Stable error codes suitable for logs, tests and automation."""

    STATE_INVALID = "E_STATE_INVALID"
    UNKNOWN_LAYER = "E_UNKNOWN_LAYER"
    INVALID_LAYER_ORDER = "E_INVALID_LAYER_ORDER"
    INVALID_ANIMATION_LAYER = "E_INVALID_ANIMATION_LAYER"
    ANIMATION_WEIGHT_RANGE = "E_ANIMATION_WEIGHT_RANGE"
    MATERIAL_PARAM_LENGTH = "E_MATERIAL_PARAM_LENGTH"
    MATERIAL_PARAM_TYPE = "E_MATERIAL_PARAM_TYPE"
    MATERIAL_BLEND_CONFLICT = "E_MATERIAL_BLEND_CONFLICT"
    EMISSIVE_TRANSPARENT_CONFLICT = "E_EMISSIVE_TRANSPARENT_CONFLICT"
    DUPLICATE_LIGHT = "E_DUPLICATE_LIGHT"
    LIGHT_NOT_FOUND = "E_LIGHT_NOT_FOUND"
    INVALID_LIGHT = "E_INVALID_LIGHT"
    LOGIC_STEP_RANGE = "E_LOGIC_STEP_RANGE"
    RENDER_TIME_INVALID = "E_RENDER_TIME_INVALID"
    PATCH_BASE_MISMATCH = "E_PATCH_BASE_MISMATCH"
    PATCH_REPLAY_FAILED = "E_PATCH_REPLAY_FAILED"
    LIGHT_SNAPSHOT_VERSION_MISMATCH = "E_LIGHT_SNAPSHOT_VERSION_MISMATCH"
    LIGHT_SNAPSHOT_REVISION_MISMATCH = "E_LIGHT_SNAPSHOT_REVISION_MISMATCH"
    LIGHT_SNAPSHOT_CONTENT_MISMATCH = "E_LIGHT_SNAPSHOT_CONTENT_MISMATCH"
    COUPLING_VERSION_MISMATCH = "E_COUPLING_VERSION_MISMATCH"
    RECOVERABLE_ERROR_ACTIVE = "E_RECOVERABLE_ERROR_ACTIVE"
    ROLLBACK_UNAVAILABLE = "E_ROLLBACK_UNAVAILABLE"
    INTERNAL = "E_INTERNAL"


@dataclass(frozen=True)
class BusinessError:
    """Serializable error description.

    Validation failures do not enter a recoverable error state. Runtime chain
    failures set ``recoverable`` and preserve the last valid snapshot.
    """

    code: ErrorCode
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)
    recoverable: bool = False
    state_version: Optional[int] = None
    last_valid_version: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": dict(self.details),
            "recoverable": self.recoverable,
            "state_version": self.state_version,
            "last_valid_version": self.last_valid_version,
        }


class SyncError(Exception):
    """Exception carrying a :class:`BusinessError`."""

    def __init__(self, error: BusinessError):
        super().__init__(f"{error.code.value}: {error.message}")
        self.error = error

    @property
    def code(self) -> ErrorCode:
        return self.error.code

    def to_dict(self) -> Dict[str, Any]:
        return self.error.to_dict()


def make_error(
    code: ErrorCode,
    message: str,
    *,
    details: Optional[Mapping[str, Any]] = None,
    recoverable: bool = False,
    state_version: Optional[int] = None,
    last_valid_version: Optional[int] = None,
) -> SyncError:
    return SyncError(
        BusinessError(
            code=code,
            message=message,
            details=dict(details or {}),
            recoverable=recoverable,
            state_version=state_version,
            last_valid_version=last_valid_version,
        )
    )
