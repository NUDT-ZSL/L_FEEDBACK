"""Structured business error codes and the exception type used across the exporter.

Every validation/runtime failure is represented by a stable string code plus a
context payload (segment index, millisecond offset, ...) so callers can persist
the failure to JSON instead of catching an unstructured exception.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class ErrorCode:
    # Generic input / IO
    INVALID_JSON = "E1000_INVALID_JSON"
    FILE_NOT_FOUND = "E1001_FILE_NOT_FOUND"
    FILE_NOT_WRITABLE = "E1002_FILE_NOT_WRITABLE"
    INVALID_WAV = "E1003_INVALID_WAV"

    # Timeline / metadata validation
    NEGATIVE_DURATION = "E2001_NEGATIVE_DURATION"
    SEGMENT_ORDER_CONFLICT = "E2002_SEGMENT_ORDER_CONFLICT"
    FADE_LONGER_THAN_SEGMENT = "E2003_FADE_LONGER_THAN_SEGMENT"
    MISSING_DIALOGUE_RANGE = "E2004_MISSING_DIALOGUE_RANGE"
    INVALID_DIALOGUE_RANGE = "E2005_INVALID_DIALOGUE_RANGE"
    MISSING_METADATA_FIELD = "E2006_MISSING_METADATA_FIELD"
    NEGATIVE_GAP = "E2007_NEGATIVE_GAP"
    INVALID_TIMELINE = "E2008_INVALID_TIMELINE"

    # Audio format validation
    SAMPLE_RATE_MISMATCH = "E3001_SAMPLE_RATE_MISMATCH"
    NON_INTEGER_FRAME = "E3002_NON_INTEGER_FRAME"
    UNEXPECTED_CHANNELS = "E3003_UNEXPECTED_CHANNELS"
    UNEXPECTED_SAMPLE_WIDTH = "E3004_UNEXPECTED_SAMPLE_WIDTH"

    # Snapshot / resume
    SNAPSHOT_VERSION_MISMATCH = "E4001_SNAPSHOT_VERSION_MISMATCH"
    BLOCK_HASH_MISMATCH = "E4002_BLOCK_HASH_MISMATCH"
    CHECKPOINT_CORRUPT = "E4003_CHECKPOINT_CORRUPT"

    # Concurrency
    OUTPUT_DIR_BUSY = "E5001_OUTPUT_DIR_BUSY"


class ExporterError(Exception):
    """Business exception carrying a stable code and structured context."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        segment_index: Optional[int] = None,
        offset_ms: Optional[float] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.segment_index = segment_index
        self.offset_ms = offset_ms
        self.detail = dict(detail or {})

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.segment_index is not None:
            payload["segment_index"] = self.segment_index
        if self.offset_ms is not None:
            payload["offset_ms"] = self.offset_ms
        if self.detail:
            payload["detail"] = self.detail
        return payload


def json_default(obj: Any) -> Any:
    """Fallback serializer for objects not natively supported by json."""
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)
