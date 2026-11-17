"""Business errors shared by the parser, renderer and CLI."""

from __future__ import annotations

from dataclasses import dataclass


EXIT_CODES = {
    "E_INPUT_READ": 2,
    "E_JSON_PARSE": 3,
    "E_SCHEMA": 4,
    "E_VALUE": 5,
    "E_TEXTURE": 6,
    "E_OUTPUT": 7,
    "E_RENDER": 8,
}


@dataclass
class BusinessError(Exception):
    code: str
    message: str
    path: str = "$"

    def __post_init__(self) -> None:
        super().__init__(self.message)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, EXIT_CODES["E_RENDER"])

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "path": self.path,
            },
        }
