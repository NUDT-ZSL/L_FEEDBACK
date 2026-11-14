"""Business errors and stable exit codes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ERROR_DEFINITIONS: dict[str, dict[str, Any]] = {
    "E001": {"exit_code": 11, "kind": "input_missing"},
    "E002": {"exit_code": 12, "kind": "image_decode"},
    "E003": {"exit_code": 13, "kind": "channel_mismatch"},
    "E004": {"exit_code": 14, "kind": "invalid_scene"},
    "E005": {"exit_code": 15, "kind": "non_unit_normal"},
    "E006": {"exit_code": 16, "kind": "texture_size_mismatch"},
    "E007": {"exit_code": 17, "kind": "invalid_uv_layout"},
    "E008": {"exit_code": 18, "kind": "io_error"},
    "E009": {"exit_code": 19, "kind": "invalid_output"},
    "E010": {"exit_code": 20, "kind": "recovery_mismatch"},
    "E011": {"exit_code": 21, "kind": "invalid_argument"},
}


@dataclass
class MaterialError(Exception):
    """An expected business failure with a stable report payload."""

    code: str
    message: str
    asset: str | None = None
    location: dict[str, Any] | None = None
    details: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.code not in ERROR_DEFINITIONS:
            raise ValueError(f"unknown material error code: {self.code}")
        super().__init__(self.message)

    @property
    def exit_code(self) -> int:
        return int(ERROR_DEFINITIONS[self.code]["exit_code"])

    @property
    def kind(self) -> str:
        return str(ERROR_DEFINITIONS[self.code]["kind"])

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "kind": self.kind,
            "exit_code": self.exit_code,
            "message": self.message,
        }
        if self.asset is not None:
            payload["asset"] = self.asset
        if self.location is not None:
            payload["location"] = self.location
        if self.details is not None:
            payload["details"] = self.details
        return payload


class SimulatedInterrupt(RuntimeError):
    """Test-only interruption after a checkpoint has been committed."""

    def __init__(self, completed_chunks: int, total_chunks: int) -> None:
        super().__init__(
            f"simulated interruption after {completed_chunks}/{total_chunks} chunks"
        )
        self.completed_chunks = completed_chunks
        self.total_chunks = total_chunks
