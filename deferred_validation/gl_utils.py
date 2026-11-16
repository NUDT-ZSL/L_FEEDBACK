"""OpenGL resource wrappers with strict cleanup and error reporting."""

from __future__ import annotations

import ctypes
from typing import Iterable

import numpy as np
from OpenGL.GL import *

from .errors import RenderError


def check_gl(path: str) -> None:
    error = glGetError()
    if error != GL_NO_ERROR:
        names = {
            GL_INVALID_ENUM: "GL_INVALID_ENUM",
            GL_INVALID_VALUE: "GL_INVALID_VALUE",
            GL_INVALID_OPERATION: "GL_INVALID_OPERATION",
            GL_INVALID_FRAMEBUFFER_OPERATION: "GL_INVALID_FRAMEBUFFER_OPERATION",
            GL_OUT_OF_MEMORY: "GL_OUT_OF_MEMORY",
        }
        raise RenderError(f"{path}: OpenGL error {names.get(error, hex(error))}")


def compile_program(name: str, vertex_source: str, fragment_source: str | None = None) -> int:
    vertex = _compile_shader(name, GL_VERTEX_SHADER, vertex_source)
    fragment = None
    try:
        if fragment_source is not None:
            fragment = _compile_shader(name, GL_FRAGMENT_SHADER, fragment_source)
        program = glCreateProgram()
        glAttachShader(program, vertex)
        if fragment is not None:
            glAttachShader(program, fragment)
        glLinkProgram(program)
        if not glGetProgramiv(program, GL_LINK_STATUS):
            log = glGetProgramInfoLog(program).decode("utf-8", "replace")
            raise RenderError(f"shader program '{name}' link failed:\n{log}")
        return program
    finally:
        glDeleteShader(vertex)
        if fragment is not None:
            glDeleteShader(fragment)


def _compile_shader(name: str, shader_type: int, source: str) -> int:
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader).decode("utf-8", "replace")
        glDeleteShader(shader)
        kind = "vertex" if shader_type == GL_VERTEX_SHADER else "fragment"
        raise RenderError(f"{kind} shader '{name}' compile failed:\n{log}")
    return shader


def uniform(program: int, name: str) -> int:
    location = glGetUniformLocation(program, name)
    if location < 0:
        raise RenderError(f"uniform '{name}' is missing from program {program}")
    return location


def optional_uniform(program: int, name: str) -> int:
    return glGetUniformLocation(program, name)


def create_texture_2d(
    width: int,
    height: int,
    internal_format: int,
    pixel_format: int,
    pixel_type: int,
    *,
    data=None,
    min_filter: int = GL_LINEAR,
    mag_filter: int = GL_LINEAR,
    wrap_s: int = GL_CLAMP_TO_EDGE,
    wrap_t: int = GL_CLAMP_TO_EDGE,
) -> int:
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    glTexImage2D(
        GL_TEXTURE_2D, 0, internal_format, width, height, 0, pixel_format, pixel_type, data
    )
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture


def create_depth_texture_2d(width: int, height: int) -> int:
    texture = create_texture_2d(
        width,
        height,
        GL_DEPTH_COMPONENT24,
        GL_DEPTH_COMPONENT,
        GL_FLOAT,
        min_filter=GL_LINEAR,
        mag_filter=GL_LINEAR,
        wrap_s=GL_CLAMP_TO_BORDER,
        wrap_t=GL_CLAMP_TO_BORDER,
    )
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, np.asarray([1.0, 1.0, 1.0, 1.0], np.float32))
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)
    glBindTexture(GL_TEXTURE_2D, 0)
    return texture


def create_color_fbo(textures: Iterable[int], width: int, height: int, depth_renderbuffer: bool = True):
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    attachments = []
    for index, texture in enumerate(textures):
        attachment = GL_COLOR_ATTACHMENT0 + index
        glFramebufferTexture2D(GL_FRAMEBUFFER, attachment, GL_TEXTURE_2D, texture, 0)
        attachments.append(attachment)
    if attachments:
        glDrawBuffers(len(attachments), attachments)
    depth_buffer = None
    if depth_renderbuffer:
        depth_buffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height)
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer
        )
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    if status != GL_FRAMEBUFFER_COMPLETE:
        raise RenderError(f"framebuffer incomplete: 0x{status:04x}")
    return fbo, depth_buffer


def create_depth_fbo(texture: int, target: int = GL_TEXTURE_2D):
    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, target, texture, 0)
    glDrawBuffer(GL_NONE)
    glReadBuffer(GL_NONE)
    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    if status != GL_FRAMEBUFFER_COMPLETE:
        raise RenderError(f"shadow framebuffer incomplete: 0x{status:04x}")
    return fbo


def create_cube_depth_texture(size: int) -> int:
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_CUBE_MAP, texture)
    for face in range(6):
        glTexImage2D(
            GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
            0,
            GL_DEPTH_COMPONENT24,
            size,
            size,
            0,
            GL_DEPTH_COMPONENT,
            GL_FLOAT,
            None,
        )
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE)
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL)
    glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
    return texture


class FullscreenTriangle:
    def __init__(self) -> None:
        self.vao = glGenVertexArrays(1)

    def draw(self) -> None:
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)

    def delete(self) -> None:
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = 0


class MeshGPU:
    def __init__(self, mesh) -> None:
        count = mesh.positions.shape[0]
        interleaved = np.empty((count, 12), dtype=np.float32)
        interleaved[:, 0:3] = mesh.positions
        interleaved[:, 3:6] = mesh.normals
        interleaved[:, 6:8] = mesh.uvs
        interleaved[:, 8:11] = mesh.tangents
        interleaved[:, 11] = mesh.bitangent_signs
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)
        self.index_count = int(mesh.indices.size)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)
        stride = 12 * 4
        attribute_specs = ((0, 3, 0), (1, 3, 12), (2, 2, 24), (3, 3, 32), (4, 1, 44))
        for location, components, offset in attribute_specs:
            glEnableVertexAttribArray(location)
            glVertexAttribPointer(
                location, components, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(offset)
            )
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh.indices.nbytes, mesh.indices, GL_STATIC_DRAW)
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw(self) -> None:
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.index_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def delete(self) -> None:
        if self.ebo:
            glDeleteBuffers(1, [self.ebo])
            self.ebo = 0
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
            self.vbo = 0
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
            self.vao = 0


def read_rgba_float(width: int, height: int) -> np.ndarray:
    data = np.empty((height, width, 4), dtype=np.float32)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT, data)
    return np.flipud(data)


def read_rgba_u8(width: int, height: int) -> np.ndarray:
    data = np.empty((height, width, 4), dtype=np.uint8)
    glPixelStorei(GL_PACK_ALIGNMENT, 1)
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data)
    return np.flipud(data)
