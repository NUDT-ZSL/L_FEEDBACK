"""Headless OpenGL 4.1 core context creation."""

from __future__ import annotations

import sys
from typing import Any

import moderngl

from .errors import gl_context_error


def _candidate_backends(requested: str) -> list[str]:
    if requested != "auto":
        return [requested]
    if sys.platform == "win32":
        return ["wgl"]
    if sys.platform == "darwin":
        return ["cgl"]
    return ["egl", "glx"]


def _profile_is_core(info: dict[str, Any]) -> bool | None:
    mask = info.get("GL_CONTEXT_PROFILE_MASK")
    if mask is None:
        return None
    return (int(mask) & 0x1) == 0x1


def create_headless_context(
    requested_backend: str = "auto",
) -> tuple[moderngl.Context, dict[str, Any]]:
    failures: list[str] = []
    for backend in _candidate_backends(requested_backend):
        try:
            ctx = moderngl.create_standalone_context(require=410, backend=backend)
        except Exception as exc:
            failures.append(f"{backend}: {type(exc).__name__}: {exc}")
            continue

        info = {str(key): value for key, value in ctx.info.items()}
        profile_core = _profile_is_core(info)
        version = int(ctx.version_code)
        if version < 410:
            ctx.release()
            failures.append(f"{backend}: OpenGL version {version} is below 410")
            continue
        if profile_core is False:
            ctx.release()
            failures.append(
                f"{backend}: got a compatibility profile ({info.get('GL_VERSION', 'unknown')})"
            )
            continue

        metadata = {
            "backend": backend,
            "version_code": version,
            "version": str(info.get("GL_VERSION", "")),
            "vendor": str(info.get("GL_VENDOR", "")),
            "renderer": str(info.get("GL_RENDERER", "")),
            "shading_language": str(info.get("GL_SHADING_LANGUAGE_VERSION", "")),
            "profile": "core" if profile_core else "reported-core",
            "context_profile_mask": info.get("GL_CONTEXT_PROFILE_MASK"),
            "max_color_attachments": info.get("GL_MAX_COLOR_ATTACHMENTS"),
        }
        return ctx, metadata

    joined = "; ".join(failures) if failures else "no backend attempted"
    raise gl_context_error(
        "unable to create a headless OpenGL 4.1 core context: " + joined,
        requested_backend=requested_backend,
        failures=failures,
    )

