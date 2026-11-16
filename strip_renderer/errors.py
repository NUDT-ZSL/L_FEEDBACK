from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EXIT_SCENE = 2
EXIT_ASSET_OR_ENVIRONMENT = 3


@dataclass
class RenderError(Exception):
    code: str
    message: str
    exit_code: int
    report_dir: Path | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "error",
            "error_code": self.code,
            "message": self.message,
            "exit_code": self.exit_code,
        }
        if self.details:
            payload["details"] = self.details
        return payload


def scene_error(path: str, message: str) -> RenderError:
    return RenderError(
        code="E_SCENE_FIELD",
        message=f"{path}: {message}",
        exit_code=EXIT_SCENE,
        details={"field": path},
    )


def asset_error(code: str, message: str, **details: Any) -> RenderError:
    return RenderError(
        code=code,
        message=message,
        exit_code=EXIT_ASSET_OR_ENVIRONMENT,
        details=details,
    )
