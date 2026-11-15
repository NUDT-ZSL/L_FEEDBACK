from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


EXIT_FORMAT = 2
EXIT_TIMECODE = 3
EXIT_OVERFLOW = 4
EXIT_TRANSCODE = 5
EXIT_INPUT = 6
EXIT_STATE = 7
EXIT_VERIFY = 8
EXIT_INTERNAL = 10
EXIT_INTERRUPTED = 130


EXIT_CODES = {
    "E_FORMAT": EXIT_FORMAT,
    "E_TIMECODE": EXIT_TIMECODE,
    "E_OVERFLOW": EXIT_OVERFLOW,
    "E_TRANSCODE": EXIT_TRANSCODE,
    "E_INPUT": EXIT_INPUT,
    "E_STATE": EXIT_STATE,
    "E_VERIFY": EXIT_VERIFY,
    "E_INTERNAL": EXIT_INTERNAL,
    "E_INTERRUPTED": EXIT_INTERRUPTED,
}


@dataclass
class BusinessError(Exception):
    code: str
    stage: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code} [{self.stage}]: {self.message}"

    def as_json(self) -> str:
        return json.dumps(
            {
                "error": {
                    "code": self.code,
                    "stage": self.stage,
                    "message": self.message,
                    "details": self.details,
                }
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, EXIT_INTERNAL)


class InterruptedRun(Exception):
    pass
