"""Business error types and stable process exit codes."""

from __future__ import annotations

from dataclasses import dataclass


EXIT_CODES = {
    "E_SCHEMA": 2,
    "E_ASSET_MISSING": 3,
    "E_FORMAT_UNSUPPORTED": 4,
    "E_CLIP_RANGE": 5,
    "E_CHAPTER_RANGE": 6,
    "E_SILENCE": 7,
    "E_PROCESSING": 8,
}


@dataclass
class BusinessError(Exception):
    code: str
    message: str

    def __post_init__(self) -> None:
        super().__init__(f"{self.code}: {self.message}")

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, 1)

