"""Headless OpenGL 4.5 Core context backed by Mesa EGL/surfaceless."""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

from OpenGL import EGL
from OpenGL.EGL import (  # noqa: E402
    EGL_ALPHA_SIZE,
    EGL_BLUE_SIZE,
    EGLConfig,
    EGL_CONTEXT_MAJOR_VERSION,
    EGL_CONTEXT_MINOR_VERSION,
    EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
    EGL_CONTEXT_OPENGL_PROFILE_MASK,
    EGL_DEFAULT_DISPLAY,
    EGL_DEPTH_SIZE,
    EGL_GREEN_SIZE,
    EGL_NO_CONTEXT,
    EGL_NO_SURFACE,
    EGL_NONE,
    EGL_OPENGL_API,
    EGL_OPENGL_BIT,
    EGL_PBUFFER_BIT,
    EGL_RED_SIZE,
    EGL_RENDERABLE_TYPE,
    EGL_SURFACE_TYPE,
    eglBindAPI,
    eglChooseConfig,
    eglCreateContext,
    eglDestroyContext,
    eglGetDisplay,
    eglGetError,
    eglInitialize,
    eglMakeCurrent,
    eglQueryString,
    eglTerminate,
)
from OpenGL.GL import (  # noqa: E402
    GL_MAX_TEXTURE_SIZE,
    GL_RENDERER,
    GL_SHADING_LANGUAGE_VERSION,
    GL_VERSION,
    glGetIntegerv,
    glGetString,
)

from .errors import RenderError


@dataclass(frozen=True)
class GLInfo:
    version: str
    renderer: str
    glsl_version: str
    vendor: str
    max_texture_size: int


class EGLContext:
    """A minimal surfaceless EGL context; no window system or X server required."""

    def __init__(self) -> None:
        self.display = None
        self.context = None
        self.info: GLInfo | None = None

    def __enter__(self) -> "EGLContext":
        try:
            self.display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
            if not self.display:
                raise RenderError(f"eglGetDisplay failed: 0x{eglGetError():04x}")
            major = EGL.EGLint()
            minor = EGL.EGLint()
            if not eglInitialize(self.display, major, minor):
                raise RenderError(f"eglInitialize failed: 0x{eglGetError():04x}")
            if not eglBindAPI(EGL_OPENGL_API):
                raise RenderError(f"eglBindAPI(OpenGL) failed: 0x{eglGetError():04x}")
            config_attributes = (
                EGL_SURFACE_TYPE,
                EGL_PBUFFER_BIT,
                EGL_RENDERABLE_TYPE,
                EGL_OPENGL_BIT,
                EGL_RED_SIZE,
                8,
                EGL_GREEN_SIZE,
                8,
                EGL_BLUE_SIZE,
                8,
                EGL_ALPHA_SIZE,
                8,
                EGL_DEPTH_SIZE,
                24,
                EGL_NONE,
            )
            configs = (EGLConfig * 1)()
            count = EGL.EGLint()
            if not eglChooseConfig(self.display, config_attributes, configs, 1, count) or count.value < 1:
                raise RenderError("no EGL OpenGL framebuffer configuration is available")
            context_attributes = (
                EGL_CONTEXT_MAJOR_VERSION,
                4,
                EGL_CONTEXT_MINOR_VERSION,
                5,
                EGL_CONTEXT_OPENGL_PROFILE_MASK,
                EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                EGL_NONE,
            )
            self.context = eglCreateContext(self.display, configs[0], EGL_NO_CONTEXT, context_attributes)
            if not self.context:
                raise RenderError(f"eglCreateContext(OpenGL 4.5 core) failed: 0x{eglGetError():04x}")
            if not eglMakeCurrent(self.display, EGL_NO_SURFACE, EGL_NO_SURFACE, self.context):
                raise RenderError(f"eglMakeCurrent failed: 0x{eglGetError():04x}")
            version = _gl_string(GL_VERSION)
            renderer = _gl_string(GL_RENDERER)
            glsl_version = _gl_string(GL_SHADING_LANGUAGE_VERSION)
            vendor = eglQueryString(self.display, EGL.EGL_VENDOR)
            vendor_text = vendor.decode("utf-8", "replace") if vendor else "unknown"
            if not version.startswith("4.5") and not _version_at_least(version, 4, 5):
                raise RenderError(f"OpenGL >= 4.5 is required, got {version}")
            max_texture_size = int(glGetIntegerv(GL_MAX_TEXTURE_SIZE))
            self.info = GLInfo(version, renderer, glsl_version, vendor_text, max_texture_size)
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.display is not None:
            try:
                eglMakeCurrent(self.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT)
            except Exception:
                pass
        if self.context is not None and self.display is not None:
            try:
                eglDestroyContext(self.display, self.context)
            except Exception:
                pass
        if self.display is not None:
            try:
                eglTerminate(self.display)
            except Exception:
                pass
        self.context = None
        self.display = None


def _gl_string(enum: int) -> str:
    value = glGetString(enum)
    return value.decode("utf-8", "replace") if value else "unknown"


def _version_at_least(version: str, major: int, minor: int) -> bool:
    try:
        prefix = version.split(" ", 1)[0]
        parts = prefix.split(".")
        return (int(parts[0]), int(parts[1])) >= (major, minor)
    except (ValueError, IndexError):
        return False
