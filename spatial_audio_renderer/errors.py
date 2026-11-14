"""Business error codes and JSON-line reporting."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


EXIT_RECOVERABLE = 1
EXIT_UNRECOVERABLE = 2

EXIT_CODE_BY_ERROR = {
    "SNAP_VERSION_MISMATCH": EXIT_RECOVERABLE,
    "SNAP_CORRUPTED": EXIT_RECOVERABLE,
    "SNAP_NOT_FOUND": EXIT_RECOVERABLE,
    "RENDER_INTERRUPTED": EXIT_RECOVERABLE,
    "INVALID_SCENE": EXIT_UNRECOVERABLE,
    "AUDIO_ASSET_MISSING": EXIT_UNRECOVERABLE,
    "INVALID_OUTPUT_DIR": EXIT_UNRECOVERABLE,
    "INVALID_ARGUMENT": EXIT_UNRECOVERABLE,
    "INTERNAL_ERROR": EXIT_UNRECOVERABLE,
}


@dataclass
class BusinessError(Exception):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)
        if self.code not in EXIT_CODE_BY_ERROR:
            raise ValueError("unknown business error code: %s" % self.code)

    @property
    def exit_code(self) -> int:
        return EXIT_CODE_BY_ERROR[self.code]

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "exit_code": self.exit_code,
        }
        if self.details is not None:
            payload["details"] = self.details
        payload.update(self.extra)
        return payload


def report_error(error: BusinessError) -> None:
    """Write one machine-readable error line to stderr."""

    sys.stderr.write(json.dumps(error.as_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    sys.stderr.flush()


def invalid_scene(message: str, **details: Any) -> BusinessError:
    return BusinessError("INVALID_SCENE", message, details or None)


def asset_missing(message: str, **details: Any) -> BusinessError:
    return BusinessError("AUDIO_ASSET_MISSING", message, details or None)


def invalid_output_dir(message: str, **details: Any) -> BusinessError:
    return BusinessError("INVALID_OUTPUT_DIR", message, details or None)


def snapshot_error(code: str, message: str, **details: Any) -> BusinessError:
    if code not in {"SNAP_VERSION_MISMATCH", "SNAP_CORRUPTED", "SNAP_NOT_FOUND"}:
        raise ValueError("not a snapshot error code: %s" % code)
    return BusinessError(code, message, details or None)

