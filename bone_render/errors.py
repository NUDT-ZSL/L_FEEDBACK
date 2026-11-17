from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


EXIT_CODES = {
    "E_IO": 3,
    "E_SCHEMA": 10,
    "E_MATRIX_DIM": 11,
    "E_BONE_PARENT": 12,
    "E_BONE_CYCLE": 13,
    "E_WEIGHT_COUNT": 14,
    "E_WEIGHT_SUM": 15,
    "E_WEIGHT_BONE_RANGE": 16,
    "E_WEIGHT_RANGE": 17,
    "E_INDEX": 18,
    "E_MATERIAL_RANGE": 19,
    "E_CAMERA": 20,
    "E_SHADOW": 21,
    "E_LIMIT": 22,
    "E_KEYFRAME": 23,
    "E_OUTPUT": 24,
    "E_THREADS": 31,
    "E_USAGE": 32,
    "E_RENDER": 40,
}


@dataclass
class SceneError(Exception):
    code: str
    path: str
    message: str

    def __post_init__(self) -> None:
        super().__init__(self.message)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "path": self.path,
                "message": self.message,
            }
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def emit_error(error: SceneError) -> None:
    import sys

    sys.stderr.write(error.to_json() + "\n")
    sys.stderr.flush()
