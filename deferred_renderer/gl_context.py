"""Hidden-window, offscreen OpenGL 3.3 core context built on GLFW."""

from __future__ import annotations

from dataclasses import dataclass

import OpenGL

# We deliberately handle GL errors ourselves with glGetError so that the
# forced-format failure path can be observed and logged structurally.
OpenGL.ERROR_CHECKING = False
OpenGL.ERROR_LOGGING = False

import glfw  # noqa: E402
import OpenGL.GL as gl  # noqa: E402
from OpenGL import platform as _gl_platform  # noqa: E402


@dataclass
class GlInfo:
    vendor: str
    renderer: str
    version: str
    glsl_version: str
    glfw_version: str


class OffscreenContext:
    """Owns a hidden GLFW window that exists only to host a GL context."""

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height
        self._window = None
        self.info: GlInfo | None = None
        self._saved_get_current_context = None

    def __enter__(self) -> "OffscreenContext":
        if not glfw.init():
            raise RuntimeError("glfw.init() failed: no usable platform backend")

        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.DECORATED, glfw.FALSE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)

        self._window = glfw.create_window(self.width, self.height, "offscreen", None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed to create an offscreen context")

        glfw.make_context_current(self._window)
        glfw.swap_interval(0)

        # GLFW on Linux/Mesa may create the context through EGL even under the
        # X11 platform, in which case PyOpenGL's GLX-based GetCurrentContext()
        # returns NULL and its per-context array tracking (used by
        # glVertexAttribPointer) raises "no valid context". Provide a stable,
        # non-zero sentinel while our single offscreen context is current.
        self._saved_get_current_context = _gl_platform.GetCurrentContext
        _gl_platform.GetCurrentContext = lambda: 1

        self.info = GlInfo(
            vendor=gl.glGetString(gl.GL_VENDOR).decode("utf-8", "replace"),
            renderer=gl.glGetString(gl.GL_RENDERER).decode("utf-8", "replace"),
            version=gl.glGetString(gl.GL_VERSION).decode("utf-8", "replace"),
            glsl_version=gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION).decode(
                "utf-8", "replace"
            ),
            glfw_version=glfw.get_version_string().decode("utf-8", "replace"),
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._saved_get_current_context is not None:
            _gl_platform.GetCurrentContext = self._saved_get_current_context
        if self._window is not None:
            glfw.destroy_window(self._window)
            self._window = None
        glfw.terminate()
