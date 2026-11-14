"""Thin PyOpenGL helpers: shaders, error decoding and state snapshots."""

from __future__ import annotations

from typing import Any

import numpy as np
import OpenGL.GL as gl


GL_ERROR_NAMES: dict[int, str] = {
    0: "GL_NO_ERROR",
    0x0500: "GL_INVALID_ENUM",
    0x0501: "GL_INVALID_VALUE",
    0x0502: "GL_INVALID_OPERATION",
    0x0505: "GL_OUT_OF_MEMORY",
    0x0506: "GL_INVALID_FRAMEBUFFER_OPERATION",
}

FBO_STATUS_NAMES: dict[int, str] = {
    gl.GL_FRAMEBUFFER_COMPLETE: "GL_FRAMEBUFFER_COMPLETE",
    gl.GL_FRAMEBUFFER_UNDEFINED: "GL_FRAMEBUFFER_UNDEFINED",
    gl.GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
        "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT",
    gl.GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
        "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT",
    gl.GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:
        "GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER",
    gl.GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:
        "GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER",
    gl.GL_FRAMEBUFFER_UNSUPPORTED: "GL_FRAMEBUFFER_UNSUPPORTED",
    gl.GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:
        "GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE",
}


def drain_gl_errors() -> list[dict[str, Any]]:
    """Collect every queued GL error via glGetError (never raises)."""
    errors: list[dict[str, Any]] = []
    seen: set[int] = set()
    for _ in range(32):
        code = int(gl.glGetError())
        if code == 0:
            break
        if code in seen:
            break
        seen.add(code)
        errors.append({"code": code, "name": GL_ERROR_NAMES.get(code, "GL_UNKNOWN")})
    return errors


def _compile_shader(kind: int, source: str, label: str) -> int:
    shader = gl.glCreateShader(kind)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    if gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS) != gl.GL_TRUE:
        log = gl.glGetShaderInfoLog(shader).decode("utf-8", "replace")
        gl.glDeleteShader(shader)
        raise RuntimeError(f"Shader compile failed ({label}):\n{log}\n--- source ---\n{source}")
    return shader


class ShaderProgram:
    def __init__(self, vert_source: str, frag_source: str, defines: dict[str, str] | None = None):
        prefix = "#version 330 core\n"
        if defines:
            prefix += "".join(f"#define {key} {value}\n" for key, value in defines.items())

        def inject(src: str) -> str:
            if src.startswith("#version"):
                newline = src.find("\n")
                body = src[newline + 1:]
                return prefix + "\n" + body
            return prefix + src

        vert = _compile_shader(gl.GL_VERTEX_SHADER, inject(vert_source), "vertex")
        frag = _compile_shader(gl.GL_FRAGMENT_SHADER, inject(frag_source), "fragment")
        self.program = gl.glCreateProgram()
        gl.glAttachShader(self.program, vert)
        gl.glAttachShader(self.program, frag)
        gl.glLinkProgram(self.program)
        if gl.glGetProgramiv(self.program, gl.GL_LINK_STATUS) != gl.GL_TRUE:
            log = gl.glGetProgramInfoLog(self.program).decode("utf-8", "replace")
            raise RuntimeError(f"Program link failed:\n{log}")
        gl.glDeleteShader(vert)
        gl.glDeleteShader(frag)
        self._uniform_cache: dict[str, int] = {}

    def use(self) -> None:
        gl.glUseProgram(self.program)

    def location(self, name: str) -> int:
        loc = self._uniform_cache.get(name)
        if loc is None:
            loc = int(gl.glGetUniformLocation(self.program, name))
            self._uniform_cache[name] = loc
        return loc

    def set_mat4(self, name: str, value: np.ndarray) -> None:
        # NumPy stores row-major (C-order) matrices while GLSL expects
        # column-major; GL_TRUE transposes on upload to match.
        gl.glUniformMatrix4fv(self.location(name), 1, gl.GL_TRUE,
                              np.ascontiguousarray(value, dtype=np.float32))

    def set_mat3(self, name: str, value: np.ndarray) -> None:
        gl.glUniformMatrix3fv(self.location(name), 1, gl.GL_TRUE,
                              np.ascontiguousarray(value, dtype=np.float32))

    def set_vec3(self, name: str, value) -> None:
        gl.glUniform3fv(self.location(name), 1, np.asarray(value, dtype=np.float32))

    def set_float(self, name: str, value: float) -> None:
        gl.glUniform1f(self.location(name), float(value))

    def set_int(self, name: str, value: int) -> None:
        gl.glUniform1i(self.location(name), int(value))

    def set_float_array(self, name: str, values) -> None:
        arr = np.asarray(values, dtype=np.float32)
        gl.glUniform1fv(self.location(name), arr.size, arr)

    def set_vec3_array(self, name: str, values) -> None:
        arr = np.asarray(values, dtype=np.float32).reshape(-1, 3)
        gl.glUniform3fv(self.location(name), arr.shape[0], arr)

    def close(self) -> None:
        gl.glDeleteProgram(self.program)


def binding_state_snapshot() -> dict[str, Any]:
    """Capture binding state used to prove save/restore around G-Buffer rebuild."""
    return {
        "framebuffer_binding": int(gl.glGetIntegerv(gl.GL_FRAMEBUFFER_BINDING)),
        "draw_framebuffer_binding": int(gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING)),
        "read_framebuffer_binding": int(gl.glGetIntegerv(gl.GL_READ_FRAMEBUFFER_BINDING)),
        "active_texture_unit": int(gl.glGetIntegerv(gl.GL_ACTIVE_TEXTURE) - gl.GL_TEXTURE0),
        "texture_binding_2d": int(gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D)),
        "viewport": [int(x) for x in gl.glGetIntegerv(gl.GL_VIEWPORT)],
        "depth_test_enabled": bool(gl.glIsEnabled(gl.GL_DEPTH_TEST)),
        "cull_face_enabled": bool(gl.glIsEnabled(gl.GL_CULL_FACE)),
        "current_program": int(gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)),
    }
