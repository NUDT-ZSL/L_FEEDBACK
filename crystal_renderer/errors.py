"""Business error codes surfaced on stderr and in report JSON."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

E_DEGENERATE_POLYGON = "E101"
E_ZERO_AREA = "E102"
E_INVALID_IOR = "E103"
E_INVALID_COLOR = "E104"
E_KEYFRAME_RANGE = "E105"
E_INVALID_TRANSFORM = "E106"
E_INVALID_CONFIG = "E107"
E_SNAPSHOT_VERSION = "E108"
E_SNAPSHOT_CHECKSUM = "E109"

_MESSAGES = {
    E_DEGENERATE_POLYGON: "degenerate polygon: fewer than 3 vertices",
    E_ZERO_AREA: "zero-area crystal polygon",
    E_INVALID_IOR: "invalid index of refraction (need finite 1.0 < ior <= 3.0)",
    E_INVALID_COLOR: "invalid color (need 4 finite components in [0, 1])",
    E_KEYFRAME_RANGE: "keyframe step out of range or not an integer",
    E_INVALID_TRANSFORM: "invalid transform (need finite translate/rotate/scale)",
    E_INVALID_CONFIG: "invalid top-level config",
    E_SNAPSHOT_VERSION: "snapshot version mismatch, discarding snapshot",
    E_SNAPSHOT_CHECKSUM: "snapshot checksum mismatch, discarding snapshot",
}


@dataclass
class ErrorLog:
    errors: list[dict] = field(default_factory=list)

    def emit(self, code: str, crystal_id: str | None, detail: str = "") -> None:
        message = _MESSAGES.get(code, "unknown error")
        if detail:
            message = f"{message}: {detail}"
        entry = {"code": code, "crystal_id": crystal_id, "message": message}
        self.errors.append(entry)
        target = crystal_id if crystal_id is not None else "-"
        print(f"ERROR {code} crystal={target}: {message}", file=sys.stderr)

    def has(self, code: str) -> bool:
        return any(e["code"] == code for e in self.errors)
