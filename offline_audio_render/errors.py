"""Business error codes and exception types.

Fatal validation failures raise :class:`SceneError` carrying a stable
business error code. The CLI serialises the error as JSON to ``stderr`` and
exits with a non-zero status. Per-event source problems are not fatal; they
become warnings in the JSON report instead.
"""

from __future__ import annotations

from dataclasses import dataclass


# Fatal scene / configuration error codes.
E_MISSING_FIELD = "E1001"
E_INVALID_LAYOUT = "E1002"
E_EVENT_OVERLAP = "E1003"
E_INVALID_VALUE = "E1004"
E_TRAJECTORY = "E1005"
E_CURVE = "E1006"
E_BUS_CONFIG = "E1007"
E_SAMPLE_RATE = "E1008"

# Non-fatal warning codes (kept in the report, never raised).
W_SOURCE_UNREADABLE = "W2001"
W_SAMPLE_RATE_MISMATCH = "W2002"
W_DAMAGED_SOURCE = "W2003"
W_DOWNMIX = "W2004"
W_UNSUPPORTED_CHANNELS = "W2005"


@dataclass
class SceneError(Exception):
    """Fatal error raised while parsing or validating a scene description."""

    code: str
    message: str
    path: str | None = None

    def __str__(self) -> str:
        location = f" ({self.path})" if self.path else ""
        return f"[{self.code}] {self.message}{location}"

    def to_dict(self) -> dict:
        payload = {"error": True, "code": self.code, "message": self.message}
        if self.path is not None:
            payload["path"] = self.path
        return payload
