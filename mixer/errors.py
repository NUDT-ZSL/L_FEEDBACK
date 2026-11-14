"""Business errors used by the offline mixer."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    E_SHOT_DESC = "E_SHOT_DESC"
    E_MEDIA_LOAD = "E_MEDIA_LOAD"
    E_TIMELINE_CONFLICT = "E_TIMELINE_CONFLICT"
    E_FADE_OVERFLOW = "E_FADE_OVERFLOW"
    E_PATH_INVALID = "E_PATH_INVALID"


class MixerError(Exception):
    """A stable, user-facing mixer failure.

    ``context`` is intentionally JSON-friendly so the CLI can persist a partial
    report even when processing stops before a complete mix exists.
    """

    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.code = ErrorCode(code) if not isinstance(code, ErrorCode) else code
        self.message = message
        self.context = context or {}
        super().__init__(f"{self.code.value}: {message}")

    @property
    def error_code(self) -> str:
        return self.code.value
