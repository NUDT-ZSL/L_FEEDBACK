from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AppError(Exception):
    """An expected, user-facing failure with a stable error code."""

    code: str
    message: str
    detail: str | None = None

    def __str__(self) -> str:
        text = f"[{self.code}] {self.message}"
        if self.detail:
            text += f": {self.detail}"
        return text


class ModelError(AppError):
    pass


class ConfigError(AppError):
    pass


class RenderError(AppError):
    pass


class ContextError(AppError):
    pass


class GLProgramError(RenderError):
    pass

