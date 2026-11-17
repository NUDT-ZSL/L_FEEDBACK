from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ._gl import GL
from .errors import GLProgramError, RenderError

ERROR_NAMES = {
    0x0500: "GL_INVALID_ENUM",
    0x0501: "GL_INVALID_VALUE",
    0x0502: "GL_INVALID_OPERATION",
    0x0503: "GL_STACK_OVERFLOW",
    0x0504: "GL_STACK_UNDERFLOW",
    0x0505: "GL_OUT_OF_MEMORY",
    0x0506: "GL_INVALID_FRAMEBUFFER_OPERATION",
}


def gl_error_name(error: int) -> str:
    return ERROR_NAMES.get(int(error), hex(int(error)))


def clear_gl_errors() -> None:
    while GL.glGetError() != GL.GL_NO_ERROR:
        pass


def check_gl_error(operation: str) -> None:
    errors: list[str] = []
    while True:
        error = GL.glGetError()
        if error == GL.GL_NO_ERROR:
            break
        errors.append(gl_error_name(error))
    if errors:
        raise RenderError("E_GL", f"OpenGL error after {operation}", ", ".join(errors))


def fbo_status_name(status: int) -> str:
    names = {
        GL.GL_FRAMEBUFFER_COMPLETE: "complete",
        GL.GL_FRAMEBUFFER_UNDEFINED: "undefined",
        GL.GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: "incomplete attachment",
        GL.GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: "missing attachment",
        GL.GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER: "incomplete draw buffer",
        GL.GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER: "incomplete read buffer",
        GL.GL_FRAMEBUFFER_UNSUPPORTED: "unsupported",
        GL.GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE: "incomplete multisample",
    }
    return names.get(int(status), hex(int(status)))


def check_framebuffer(label: str) -> None:
    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if status != GL.GL_FRAMEBUFFER_COMPLETE:
        raise RenderError(
            "E_FRAMEBUFFER",
            f"{label} framebuffer is incomplete",
            fbo_status_name(status),
        )


def _compile_shader(shader_type: int, source: str, label: str) -> int:
    shader = GL.glCreateShader(shader_type)
    if not shader:
        raise GLProgramError("E_SHADER_CREATE", f"could not create {label} shader")
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    if GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS) == GL.GL_FALSE:
        log = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        GL.glDeleteShader(shader)
        check_gl_error(f"compile {label} shader")
        raise GLProgramError("E_SHADER_COMPILE", f"{label} shader compilation failed", log.strip())
    return shader


class ShaderProgram:
    def __init__(self, vertex_source: str, fragment_source: str, label: str) -> None:
        self.label = label
        self.program = 0
        vertex_shader = 0
        fragment_shader = 0
        try:
            vertex_shader = _compile_shader(GL.GL_VERTEX_SHADER, vertex_source, f"{label} vertex")
            fragment_shader = _compile_shader(
                GL.GL_FRAGMENT_SHADER, fragment_source, f"{label} fragment"
            )
            self.program = GL.glCreateProgram()
            if not self.program:
                raise GLProgramError("E_PROGRAM_CREATE", f"could not create {label} program")
            GL.glAttachShader(self.program, vertex_shader)
            GL.glAttachShader(self.program, fragment_shader)
            GL.glLinkProgram(self.program)
            if GL.glGetProgramiv(self.program, GL.GL_LINK_STATUS) == GL.GL_FALSE:
                log = GL.glGetProgramInfoLog(self.program).decode("utf-8", errors="replace")
                raise GLProgramError("E_SHADER_LINK", f"{label} program link failed", log.strip())
            GL.glDetachShader(self.program, vertex_shader)
            GL.glDetachShader(self.program, fragment_shader)
            check_gl_error(f"link {label} program")
            self._uniforms: dict[str, int] = {}
            count = GL.glGetProgramiv(self.program, GL.GL_ACTIVE_UNIFORMS)
            for index in range(int(count)):
                name, _, _ = GL.glGetActiveUniform(self.program, index)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                self._uniforms[name] = int(GL.glGetUniformLocation(self.program, name))
        except Exception:
            if self.program:
                GL.glDeleteProgram(self.program)
            raise
        finally:
            if vertex_shader:
                GL.glDeleteShader(vertex_shader)
            if fragment_shader:
                GL.glDeleteShader(fragment_shader)

    def use(self) -> None:
        GL.glUseProgram(self.program)

    def uniform_location(self, name: str) -> int:
        return self._uniforms.get(name, -1)

    def set_int(self, name: str, value: int) -> None:
        location = self.uniform_location(name)
        if location >= 0:
            GL.glUniform1i(location, value)

    def set_float(self, name: str, value: float) -> None:
        location = self.uniform_location(name)
        if location >= 0:
            GL.glUniform1f(location, value)

    def set_vec3(self, name: str, value: Iterable[float]) -> None:
        location = self.uniform_location(name)
        if location >= 0:
            GL.glUniform3f(location, *[float(x) for x in value])

    def set_mat4(self, name: str, matrix: Any) -> None:
        location = self.uniform_location(name)
        if location >= 0:
            GL.glUniformMatrix4fv(location, 1, GL.GL_FALSE, matrix)

    def set_mat3(self, name: str, matrix: Any) -> None:
        location = self.uniform_location(name)
        if location >= 0:
            GL.glUniformMatrix3fv(location, 1, GL.GL_FALSE, matrix)

    def destroy(self) -> None:
        if self.program:
            GL.glDeleteProgram(self.program)
            self.program = 0

