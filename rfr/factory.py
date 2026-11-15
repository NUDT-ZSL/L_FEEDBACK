"""Backend selection with automatic software fallback on GL failure."""

from __future__ import annotations

from typing import List, Tuple

from .backends.gl_backend import GLBackend
from .backends.software import SoftwareBackend
from .errors import RenderError


def create_backend(name: str, width: int, height: int,
                   warnings: List[str]):
    """Return ``(backend, effective_name)`` and append fallback warnings."""
    name = name.lower()
    if name in ("software", "cpu", "numpy"):
        return SoftwareBackend(width, height), "software"
    if name not in ("opengl", "gl", "gpu"):
        raise RenderError("E_BACKEND_UNAVAILABLE", f"Unknown backend: {name}",
                          {"backend": name})
    try:
        backend = GLBackend(width, height)
        return backend, "opengl"
    except RenderError as exc:
        warnings.append(
            f"OpenGL backend unavailable ({exc.code}: {exc}); falling back to software.")
        return SoftwareBackend(width, height), "software"
