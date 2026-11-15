from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PipelineError(Exception):
    """A deterministic, machine-readable pipeline failure."""

    code: str
    message: str
    stage: str
    node_index: Optional[int] = None
    node_id: Optional[str] = None
    field: Optional[str] = None

    def __str__(self) -> str:
        location = ""
        if self.node_index is not None:
            location = f" node_index={self.node_index}"
            if self.node_id is not None:
                location += f" node_id={self.node_id!r}"
        return f"[{self.code}]{location}: {self.message}"

    def as_error_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "node_index": self.node_index,
            "node_id": self.node_id,
            "field": self.field,
        }

    def as_envelope(self) -> Dict[str, Any]:
        return {"ok": False, "error": self.as_error_dict()}


def graph_error(
    code: str,
    message: str,
    *,
    node_index: Optional[int] = None,
    node_id: Optional[str] = None,
    field: Optional[str] = None,
) -> PipelineError:
    return PipelineError(
        code=code,
        message=message,
        stage="graph",
        node_index=node_index,
        node_id=node_id,
        field=field,
    )


def scene_error(code: str, message: str, *, field: Optional[str] = None) -> PipelineError:
    return PipelineError(code=code, message=message, stage="scene", field=field)


def output_error(code: str, message: str, *, field: Optional[str] = None) -> PipelineError:
    return PipelineError(code=code, message=message, stage="output", field=field)
