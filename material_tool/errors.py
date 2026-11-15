"""Structured errors shared by the parser, renderer and CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StructuredError:
    code: str
    message: str
    graph_id: str | None = None
    node_id: str | None = None
    channel: str | None = None
    path: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "error_code": self.code,
            "message": self.message,
        }
        if self.graph_id is not None:
            result["graph_id"] = self.graph_id
        if self.node_id is not None:
            result["node_id"] = self.node_id
        if self.channel is not None:
            result["channel"] = self.channel
        if self.path:
            result["node_path"] = list(self.path)
        if self.details:
            result["details"] = self.details
        return result


class MaterialToolError(Exception):
    """Base class for recoverable tool failures."""

    code = "TOOL_ERROR"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def as_structured(self, graph_id: str | None = None) -> StructuredError:
        return StructuredError(
            code=self.code,
            message=self.message,
            graph_id=graph_id,
            details=self.details,
        )


class GLContextError(MaterialToolError):
    code = "GL_CONTEXT_ERROR"


class ShaderCompileError(MaterialToolError):
    code = "GL_SHADER_COMPILE_ERROR"


class ShaderLinkError(MaterialToolError):
    code = "GL_SHADER_LINK_ERROR"


class RenderError(MaterialToolError):
    code = "GL_RENDER_ERROR"

