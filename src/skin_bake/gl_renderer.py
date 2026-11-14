"""OpenGL 3.3 core renderer shared by the WGL viewer and EGL check path."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import numpy as np

from .errors import gl_runtime
from .runtime import BakedAsset
from .software import camera_matrices


VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aTangent;
layout(location = 3) in vec3 aBitangent;
layout(location = 4) in vec2 aBlockOffset;
layout(location = 5) in vec2 aBlockSize;

uniform mat4 uViewProjection;
uniform sampler2D uDisplacement;
uniform vec2 uAtlasSize;
uniform int uSampleIndex;

out vec3 vNormal;
out vec3 vWorldPosition;

void main() {
    int width = max(1, int(aBlockSize.x + 0.5));
    int row = uSampleIndex / width;
    int column = uSampleIndex - row * width;
    vec2 texel = aBlockOffset + vec2(float(column) + 0.5, float(row) + 0.5);
    vec2 texCoord = texel / uAtlasSize;
    vec2 displacement = texture(uDisplacement, texCoord).rg;
    vec3 world = aPosition + aTangent * displacement.x + aBitangent * displacement.y;
    vWorldPosition = world;
    vNormal = normalize(aNormal);
    gl_Position = uViewProjection * vec4(world, 1.0);
}
"""


FRAGMENT_SHADER = """#version 330 core
in vec3 vNormal;
in vec3 vWorldPosition;
out vec4 fragColor;

void main() {
    vec3 n = normalize(vNormal);
    vec3 light = normalize(vec3(0.28, 0.42, 0.86));
    float diffuse = max(dot(n, light), 0.0);
    float rim = pow(1.0 - abs(n.z), 2.0);
    vec3 base = vec3(0.12, 0.48, 0.78);
    vec3 color = base * (0.26 + 0.74 * diffuse) + vec3(0.16, 0.25, 0.30) * rim;
    fragColor = vec4(color, 1.0);
}
"""


@dataclass(frozen=True)
class GLRenderResult:
    image: np.ndarray
    sample_index: int
    frame_time: float
    version: str
    renderer: str


def _import_gl():
    try:
        from OpenGL import GL
    except ImportError as exc:
        raise gl_runtime(f"PyOpenGL is required for the viewer: {exc}") from exc
    return GL


class GLRenderer:
    def __init__(self, asset: BakedAsset, width: int, height: int):
        self.asset = asset
        self.width = int(width)
        self.height = int(height)
        self.GL = _import_gl()
        self.GL.glViewport(0, 0, self.width, self.height)
        version = self.GL.glGetString(self.GL.GL_VERSION)
        self.version = version.decode("ascii", "replace") if version else "unknown"
        if self._version_tuple(self.version) < (3, 3):
            raise gl_runtime(f"OpenGL 3.3 core is required, got {self.version}")
        renderer = self.GL.glGetString(self.GL.GL_RENDERER)
        self.renderer = renderer.decode("ascii", "replace") if renderer else "unknown"
        self._program = self._make_program()
        self._vao, self._dynamic_vbo, self._static_vbo, self._ebo = self._make_buffers()
        self._texture = self._make_texture()
        self._upload_static_attributes()
        self._uniform_view_projection = self.GL.glGetUniformLocation(
            self._program, "uViewProjection"
        )
        self._uniform_displacement = self.GL.glGetUniformLocation(
            self._program, "uDisplacement"
        )
        self._uniform_atlas_size = self.GL.glGetUniformLocation(
            self._program, "uAtlasSize"
        )
        self._uniform_sample_index = self.GL.glGetUniformLocation(
            self._program, "uSampleIndex"
        )
        self.GL.glEnable(self.GL.GL_DEPTH_TEST)
        self.GL.glEnable(self.GL.GL_CULL_FACE)
        self.GL.glCullFace(self.GL.GL_BACK)

    @staticmethod
    def _version_tuple(version: str) -> tuple[int, int]:
        try:
            major_text = version.split()[1] if version.startswith("OpenGL ES") else version.split()[0]
            major, minor = major_text.split(".")[:2]
            return int(major), int(minor)
        except (IndexError, ValueError):
            return (0, 0)

    def _compile(self, stage: int, source: str) -> int:
        GL = self.GL
        shader = GL.glCreateShader(stage)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
            log = GL.glGetShaderInfoLog(shader)
            if isinstance(log, bytes):
                log = log.decode("utf-8", "replace")
            GL.glDeleteShader(shader)
            raise gl_runtime(f"shader compilation failed: {log}")
        return shader

    def _make_program(self) -> int:
        GL = self.GL
        vertex = self._compile(GL.GL_VERTEX_SHADER, VERTEX_SHADER)
        fragment = self._compile(GL.GL_FRAGMENT_SHADER, FRAGMENT_SHADER)
        program = GL.glCreateProgram()
        GL.glAttachShader(program, vertex)
        GL.glAttachShader(program, fragment)
        GL.glLinkProgram(program)
        GL.glDeleteShader(vertex)
        GL.glDeleteShader(fragment)
        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            log = GL.glGetProgramInfoLog(program)
            if isinstance(log, bytes):
                log = log.decode("utf-8", "replace")
            GL.glDeleteProgram(program)
            raise gl_runtime(f"shader link failed: {log}")
        return program

    def _make_buffers(self) -> tuple[int, int, int, int]:
        GL = self.GL
        vao = GL.glGenVertexArrays(1)
        dynamic_vbo = GL.glGenBuffers(1)
        static_vbo = GL.glGenBuffers(1)
        ebo = GL.glGenBuffers(1)
        GL.glBindVertexArray(vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, dynamic_vbo)
        stride = 12 * 4
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            self.asset.scene.vertex_count * stride,
            None,
            GL.GL_DYNAMIC_DRAW,
        )
        for location, offset in ((0, 0), (1, 3 * 4), (2, 6 * 4), (3, 9 * 4)):
            GL.glEnableVertexAttribArray(location)
            GL.glVertexAttribPointer(
                location, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, ctypes.c_void_p(offset)
            )

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, static_vbo)
        static_stride = 4 * 4
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            self.asset.scene.vertex_count * static_stride,
            None,
            GL.GL_STATIC_DRAW,
        )
        for location, offset in ((4, 0), (5, 2 * 4)):
            GL.glEnableVertexAttribArray(location)
            GL.glVertexAttribPointer(
                location, 2, GL.GL_FLOAT, GL.GL_FALSE, static_stride, ctypes.c_void_p(offset)
            )

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
        indices = np.ascontiguousarray(self.asset.scene.triangles.reshape(-1), dtype=np.uint32)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            indices.nbytes,
            indices,
            GL.GL_STATIC_DRAW,
        )
        GL.glBindVertexArray(0)
        return vao, dynamic_vbo, static_vbo, ebo

    def _make_texture(self) -> int:
        GL = self.GL
        texture = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        data = np.ascontiguousarray(self.asset.texture.pixels, dtype="<f2")
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RG16F,
            self.asset.texture.width,
            self.asset.texture.height,
            0,
            GL.GL_RG,
            GL.GL_HALF_FLOAT,
            data,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        return texture

    def _upload_static_attributes(self) -> None:
        GL = self.GL
        values = np.empty((self.asset.scene.vertex_count, 4), dtype=np.float32)
        for vertex in range(self.asset.scene.vertex_count):
            block = self.asset.blocks[vertex]
            values[vertex, 0:2] = block.pixel_offset
            values[vertex, 2:4] = block.size
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._static_vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, values.nbytes, values)

    def _view_projection(self) -> np.ndarray:
        view, projection = camera_matrices(self.asset, self.width / self.height)
        return (projection @ view).astype(np.float32)

    def render(self, time: float) -> GLRenderResult:
        GL = self.GL
        skin, _displacement, position = self.asset.displaced_frame(time)
        interleaved = np.empty((self.asset.scene.vertex_count, 12), dtype=np.float32)
        interleaved[:, 0:3] = position
        interleaved[:, 3:6] = skin.normals
        interleaved[:, 6:9] = skin.tangents
        interleaved[:, 9:12] = skin.bitangents

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._dynamic_vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, interleaved.nbytes, interleaved)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClearColor(0.012, 0.018, 0.028, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glUseProgram(self._program)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glUniform1i(self._uniform_displacement, 0)
        matrix = np.ascontiguousarray(self._view_projection().T)
        GL.glUniformMatrix4fv(self._uniform_view_projection, 1, GL.GL_FALSE, matrix)
        GL.glUniform2f(
            self._uniform_atlas_size,
            float(self.asset.texture.width),
            float(self.asset.texture.height),
        )
        sample_index = self.asset.curve_sample_index(time)
        GL.glUniform1i(self._uniform_sample_index, sample_index)
        GL.glBindVertexArray(self._vao)
        GL.glDrawElements(
            GL.GL_TRIANGLES,
            int(self.asset.scene.triangles.size),
            GL.GL_UNSIGNED_INT,
            None,
        )
        GL.glBindVertexArray(0)
        GL.glFinish()
        GL.glReadBuffer(GL.GL_BACK)

        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        raw = GL.glReadPixels(
            0, 0, self.width, self.height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE
        )
        image = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)[::-1].copy()
        return GLRenderResult(image, sample_index, time, self.version, self.renderer)

