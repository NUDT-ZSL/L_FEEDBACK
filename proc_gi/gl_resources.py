from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from OpenGL.GL import (
    GL_ARRAY_BUFFER,
    GL_BACK,
    GL_CLAMP_TO_EDGE,
    GL_COLOR_ATTACHMENT0,
    GL_COMPILE_STATUS,
    GL_DEPTH24_STENCIL8,
    GL_DEPTH_ATTACHMENT,
    GL_DEPTH_COMPONENT,
    GL_DEPTH_COMPONENT24,
    GL_DEPTH_TEST,
    GL_DRAW_FRAMEBUFFER,
    GL_ELEMENT_ARRAY_BUFFER,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LINEAR,
    GL_LINK_STATUS,
    GL_NONE,
    GL_PACK_ALIGNMENT,
    GL_R32F,
    GL_R32I,
    GL_RED,
    GL_RED_INTEGER,
    GL_RGBA,
    GL_RGBA16F,
    GL_RGBA32F,
    GL_RGBA8,
    GL_RGB,
    GL_RENDERBUFFER,
    GL_STATIC_DRAW,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_CUBE_MAP,
    GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_R,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_UNSIGNED_BYTE,
    GL_UNSIGNED_INT,
    GL_VERTEX_SHADER,
    GL_FRAGMENT_SHADER,
    glActiveTexture,
    glAttachShader,
    glBindBuffer,
    glBindFramebuffer,
    glBindRenderbuffer,
    glBindTexture,
    glBindVertexArray,
    glBlendEquation,
    glBlendFunc,
    glBufferData,
    glCheckFramebufferStatus,
    glClear,
    glClearColor,
    glCompileShader,
    glCreateProgram,
    glCreateShader,
    glDeleteBuffers,
    glDeleteFramebuffers,
    glDeleteProgram,
    glDeleteRenderbuffers,
    glDeleteShader,
    glDeleteTextures,
    glDeleteVertexArrays,
    glDrawBuffer,
    glDrawBuffers,
    glDrawArrays,
    glDrawElements,
    glEnableVertexAttribArray,
    glFramebufferRenderbuffer,
    glFramebufferTexture2D,
    glGenBuffers,
    glGenFramebuffers,
    glGenRenderbuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetUniformLocation,
    glLinkProgram,
    glPixelStorei,
    glReadBuffer,
    glReadPixels,
    glRenderbufferStorage,
    glShaderSource,
    glTexImage2D,
    glTexParameterf,
    glTexParameteri,
    glUniform1f,
    glUniform1fv,
    glUniform1i,
    glUniform1iv,
    glUniform2f,
    glUniform3f,
    glUniform3fv,
    glUniform4fv,
    glUniformMatrix4fv,
    glUseProgram,
    glVertexAttribPointer,
    glViewport,
)
from PIL import Image

from .geometry import MeshData

SHADER_DIR = Path(__file__).resolve().parent / "shaders"

# PyOpenGL's public glVertexAttribPointer wrapper caches a Python array in its
# context dictionary. Some WSLg/EGL-backed contexts are invisible to that
# wrapper's GLX lookup, so use the ctypes entry point for this single call.
from OpenGL.raw.GL.VERSION.GL_2_0 import glVertexAttribPointer as _raw_vertex_attrib_pointer


def _compile_shader(source: str, shader_type: int) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader).decode("utf-8", errors="replace")
        glDeleteShader(shader)
        raise RuntimeError(f"shader compile failed:\n{log}")
    return shader


class ShaderProgram:
    def __init__(
        self,
        vertex_name: str,
        fragment_name: str,
        *,
        defines: dict[str, int | str] | None = None,
    ) -> None:
        prefix = "".join(
            f"#define {key} {value}\n" for key, value in sorted((defines or {}).items())
        )
        vertex_source = (SHADER_DIR / vertex_name).read_text(encoding="utf-8")
        fragment_source = (SHADER_DIR / fragment_name).read_text(encoding="utf-8")
        if prefix:
            vertex_source = vertex_source.replace("#version 330 core\n", f"#version 330 core\n{prefix}", 1)
            fragment_source = fragment_source.replace(
                "#version 330 core\n", f"#version 330 core\n{prefix}", 1
            )
        vertex = _compile_shader(vertex_source, GL_VERTEX_SHADER)
        fragment = _compile_shader(fragment_source, GL_FRAGMENT_SHADER)
        self.program = glCreateProgram()
        glAttachShader(self.program, vertex)
        glAttachShader(self.program, fragment)
        glLinkProgram(self.program)
        glDeleteShader(vertex)
        glDeleteShader(fragment)
        if not glGetProgramiv(self.program, GL_LINK_STATUS):
            log = glGetProgramInfoLog(self.program).decode("utf-8", errors="replace")
            glDeleteProgram(self.program)
            raise RuntimeError(f"program link failed:\n{log}")
        self._uniform_cache: dict[str, int] = {}

    def use(self) -> None:
        glUseProgram(self.program)

    def location(self, name: str) -> int:
        if name not in self._uniform_cache:
            self._uniform_cache[name] = glGetUniformLocation(self.program, name)
        return self._uniform_cache[name]

    def set_int(self, name: str, value: int) -> None:
        glUniform1i(self.location(name), int(value))

    def set_float(self, name: str, value: float) -> None:
        glUniform1f(self.location(name), float(value))

    def set_vec2(self, name: str, values: Sequence[float]) -> None:
        glUniform2f(self.location(name), float(values[0]), float(values[1]))

    def set_vec3(self, name: str, values: Sequence[float]) -> None:
        glUniform3f(self.location(name), float(values[0]), float(values[1]), float(values[2]))

    def set_mat4(self, name: str, matrix: np.ndarray) -> None:
        glUniformMatrix4fv(
            self.location(name),
            1,
            GL_FALSE,
            np.asarray(matrix, dtype=np.float32).T,
        )

    def set_int_array(self, name: str, values: Sequence[int]) -> None:
        glUniform1iv(self.location(name), len(values), np.asarray(values, dtype=np.int32))

    def set_float_array(self, name: str, values: Sequence[float]) -> None:
        if not values:
            return
        glUniform1fv(self.location(name), len(values), np.asarray(values, dtype=np.float32))

    def set_vec3_array(self, name: str, values: np.ndarray) -> None:
        values = np.ascontiguousarray(values, dtype=np.float32)
        glUniform3fv(self.location(name), len(values), values)

    def set_vec4_array(self, name: str, values: np.ndarray) -> None:
        values = np.ascontiguousarray(values, dtype=np.float32)
        glUniform4fv(self.location(name), len(values), values)

    def set_texture(self, name: str, unit: int, texture: int, target: int = GL_TEXTURE_2D) -> None:
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(target, texture)
        self.set_int(name, unit)

    def close(self) -> None:
        if self.program:
            glDeleteProgram(self.program)
            self.program = 0


class MeshGPU:
    def __init__(self, mesh: MeshData) -> None:
        self.mesh = mesh
        self.vao = int(glGenVertexArrays(1))
        self.vbo = int(glGenBuffers(1))
        self.ebo = int(glGenBuffers(1))
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, mesh.vertices.nbytes, mesh.vertices, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.nbytes, mesh.indices, GL_STATIC_DRAW)
        stride = 7 * np.dtype(np.float32).itemsize
        glEnableVertexAttribArray(0)
        _raw_vertex_attrib_pointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        _raw_vertex_attrib_pointer(
            1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4)
        )
        glEnableVertexAttribArray(2)
        _raw_vertex_attrib_pointer(
            2, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4)
        )
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw_groups(self) -> int:
        glBindVertexArray(self.vao)
        draw_calls = 0
        for group in self.mesh.groups:
            glDrawElements(
                GL_TRIANGLES,
                group.index_count,
                GL_UNSIGNED_INT,
                ctypes.c_void_p(group.index_byte_offset),
            )
            draw_calls += 1
        glBindVertexArray(0)
        return draw_calls

    def close(self) -> None:
        glDeleteBuffers(1, [self.vbo])
        glDeleteBuffers(1, [self.ebo])
        glDeleteVertexArrays(1, [self.vao])


def _new_texture(
    width: int,
    height: int,
    internal_format: int,
    pixel_format: int,
    pixel_type: int,
    *,
    filter_mode: int = GL_LINEAR,
) -> int:
    texture = int(glGenTextures(1))
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        internal_format,
        width,
        height,
        0,
        pixel_format,
        pixel_type,
        None,
    )
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter_mode)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter_mode)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    return texture


def _new_framebuffer() -> int:
    return int(glGenFramebuffers(1))


def _check_framebuffer(label: str) -> None:
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"{label} framebuffer incomplete: 0x{status:x}")


class GBuffer:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.position = _new_texture(width, height, GL_RGBA32F, GL_RGBA, GL_FLOAT)
        self.normal_material = _new_texture(width, height, GL_RGBA16F, GL_RGBA, GL_FLOAT)
        self.depth = _new_texture(
            width,
            height,
            GL_DEPTH_COMPONENT24,
            GL_DEPTH_COMPONENT,
            GL_FLOAT,
        )
        self.fbo = _new_framebuffer()
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.position, 0
        )
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + 1, GL_TEXTURE_2D, self.normal_material, 0
        )
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.depth, 0
        )
        from OpenGL.GL import glDrawBuffers
        glDrawBuffers(2, np.asarray([GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT0 + 1], dtype=np.uint32))
        _check_framebuffer("G-buffer")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind(self) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)

    def close(self) -> None:
        glDeleteTextures(3, [self.position, self.normal_material, self.depth])
        glDeleteFramebuffers(1, [self.fbo])


class ColorTarget:
    def __init__(
        self,
        width: int,
        height: int,
        internal_format: int = GL_RGBA16F,
        pixel_format: int = GL_RGBA,
        pixel_type: int = GL_FLOAT,
    ) -> None:
        self.width = width
        self.height = height
        self.texture = _new_texture(width, height, internal_format, pixel_format, pixel_type)
        self.fbo = _new_framebuffer()
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture, 0
        )
        _check_framebuffer("color target")
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind(self) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.width, self.height)

    def read_rgb(self) -> np.ndarray:
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_FLOAT)
        pixels = np.frombuffer(data, dtype=np.float32).reshape(self.height, self.width, 3)
        return pixels[::-1].copy()

    def close(self) -> None:
        glDeleteTextures(1, [self.texture])
        glDeleteFramebuffers(1, [self.fbo])


class FinalTarget:
    def __init__(self, width: int, height: int) -> None:
        self.target = ColorTarget(width, height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE)
        self.width = width
        self.height = height

    def save_png(self, path: str | Path) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, self.target.fbo)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        pixels = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)[::-1]
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels, mode="RGB").save(target)

    def close(self) -> None:
        self.target.close()


class DepthCube:
    def __init__(self, size: int) -> None:
        self.size = size
        self.texture = int(glGenTextures(1))
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture)
        for face in range(6):
            glTexImage2D(
                GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                0,
                GL_R32F,
                size,
                size,
                0,
                GL_RED,
                GL_FLOAT,
                None,
            )
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
        self.fbo = _new_framebuffer()
        self.depth_buffer = int(glGenRenderbuffers(1))
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_buffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, size, size)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_buffer
        )
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind_face(self, face: int) -> None:
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
            self.texture,
            0,
        )
        glDrawBuffer(GL_COLOR_ATTACHMENT0)
        _check_framebuffer("VPL cube face")
        glViewport(0, 0, self.size, self.size)

    def close(self) -> None:
        glDeleteTextures(1, [self.texture])
        glDeleteRenderbuffers(1, [self.depth_buffer])
        glDeleteFramebuffers(1, [self.fbo])


class FullscreenTriangle:
    def __init__(self) -> None:
        self.vao = int(glGenVertexArrays(1))

    def draw(self) -> None:
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)

    def close(self) -> None:
        glDeleteVertexArrays(1, [self.vao])

