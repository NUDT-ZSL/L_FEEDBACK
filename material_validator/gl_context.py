from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ContextInfo:
    backend: str
    version: str
    renderer: str
    vendor: str

    def to_dict(self) -> dict[str, str]:
        return {
            "backend": self.backend,
            "version": self.version,
            "renderer": self.renderer,
            "vendor": self.vendor,
        }


def create_headless_context(require: int = 330) -> tuple[Any, ContextInfo]:
    import moderngl

    requested = os.environ.get("MATERIAL_VALIDATOR_GL_BACKEND", "").strip().lower()
    candidates = [requested] if requested else ["egl", "wgl", "auto"]
    failures: list[str] = []
    for backend in candidates:
        kwargs: dict[str, Any] = {"standalone": True, "require": require}
        if backend != "auto":
            kwargs["backend"] = backend
        try:
            context = moderngl.create_context(**kwargs)
        except Exception as exc:
            failures.append(f"{backend}: {type(exc).__name__}: {exc}")
            continue
        info = context.info
        context_info = ContextInfo(
            backend=backend,
            version=str(info.get("GL_VERSION", "unknown")),
            renderer=str(info.get("GL_RENDERER", "unknown")),
            vendor=str(info.get("GL_VENDOR", "unknown")),
        )
        return context, context_info
    raise RuntimeError(
        "Unable to create an OpenGL 3.3+ headless context. "
        "Install EGL/WGL support and verify the graphics driver. "
        + " | ".join(failures)
    )
