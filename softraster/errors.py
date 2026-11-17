"""Business errors and exit codes used by the command line renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


RENDER_ERR_SCHEMA = "RENDER_ERR_SCHEMA"
RENDER_ERR_INDEX = "RENDER_ERR_INDEX"
RENDER_ERR_LAYER_LIMIT = "RENDER_ERR_LAYER_LIMIT"
RENDER_ERR_IO = "RENDER_ERR_IO"
RENDER_ERR_DETERMINISM = "RENDER_ERR_DETERMINISM"

EXIT_CODES = {
    RENDER_ERR_SCHEMA: 2,
    RENDER_ERR_INDEX: 3,
    RENDER_ERR_LAYER_LIMIT: 4,
    RENDER_ERR_IO: 5,
    RENDER_ERR_DETERMINISM: 6,
}


@dataclass
class RenderError(Exception):
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def as_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


def schema_error(message: str, **details: Any) -> RenderError:
    return RenderError(RENDER_ERR_SCHEMA, message, details)


def index_error(message: str, **details: Any) -> RenderError:
    return RenderError(RENDER_ERR_INDEX, message, details)


def layer_limit_error(message: str, **details: Any) -> RenderError:
    return RenderError(RENDER_ERR_LAYER_LIMIT, message, details)
