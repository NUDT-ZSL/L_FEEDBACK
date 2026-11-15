"""OpenGL 3.3 offscreen renderer with tracked GPU resources."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import glfw
import numpy as np
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_ATTACHMENT0,
    GL_COLOR_ATTACHMENT1,
    GL_COLOR_ATTACHMENT2,
    GL_COLOR_ATTACHMENT3,
    GL_COLOR_BUFFER_BIT,
    GL_COMPILE_STATUS,
    GL_DEPTH_TEST,
    GL_DRAW_FRAMEBUFFER,
    GL_DYNAMIC_DRAW,
    GL_FALSE,
    GL_FLOAT,
    GL_FRAGMENT_SHADER,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_LINEAR,
    GL_LINK_STATUS,
    GL_MAJOR_VERSION,
    GL_MINOR_VERSION,
    GL_NEAREST,
    GL_READ_FRAMEBUFFER,
    GL_REPEAT,
    GL_RGBA,
    GL_RGBA8,
    GL_RGBA32F,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRIANGLES,
    GL_TRUE,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    GL_UNIFORM_BUFFER,
    GL_VERTEX_SHADER,
    glActiveTexture,
    glAttachShader,
    glBindBuffer,
    glBindBufferBase,
    glBindFramebuffer,
    glBindTexture,
    glBindVertexArray,
    glBlitFramebuffer,
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
    glDeleteShader,
    glDeleteTextures,
    glDeleteVertexArrays,
    glDisable,
    glDrawBuffers,
    glDrawArrays,
    glEnable,
    glFinish,
    glFramebufferTexture2D,
    glGenBuffers,
    glGenFramebuffers,
    glGenTextures,
    glGenVertexArrays,
    glGetError,
    glGetIntegerv,
    glGetProgramInfoLog,
    glGetProgramiv,
    glGetShaderInfoLog,
    glGetShaderiv,
    glGetString,
    glGetUniformBlockIndex,
    glGetUniformLocation,
    glLinkProgram,
    glReadBuffer,
    glReadPixels,
    glShaderSource,
    glTexImage2D,
    glTexParameteri,
    glUniform1i,
    glUniformBlockBinding,
    glUseProgram,
    glViewport,
    GL_NO_ERROR,
    GL_RENDERER,
    GL_VENDOR,
    GL_VERSION,
)
from PIL import Image

from .errors import GLContextError, RenderError, ShaderCompileError, ShaderLinkError
from .glsl import ShaderSources, generate_shaders, material_uniform_layout
from .graph import (
    BUILTIN_TEXTURES,
    CHANNEL_INDEX,
    CHANNELS,
    CameraSpec,
    LightSpec,
    ParsedGraph,
)


@dataclass
class ChannelRenderResult:
    channel: str
    image_path: Path
    stats: dict[str, Any]
    roughness: np.ndarray
    metallic: np.ndarray
    normal: np.ndarray


def _gl_id(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        value = value[0]
    return int(value)


class ResourceLedger:
    """Counts only resources for which allocation and release were recorded."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, int], str] = {}

    def allocate(self, kind: str, gl_id: Any, label: str) -> int:
        identifier = _gl_id(gl_id)
        key = (kind, identifier)
        if key in self._records:
            raise RuntimeError(f"Duplicate resource record for {kind}:{identifier}")
        self._records[key] = label
        return identifier

    def release(self, kind: str, gl_id: Any) -> None:
        identifier = _gl_id(gl_id)
        key = (kind, identifier)
        if key not in self._records:
            raise RuntimeError(f"Release of untracked resource {kind}:{identifier}")
        del self._records[key]

    def snapshot(self) -> dict[str, int]:
        counts = Counter(kind for kind, _ in self._records)
        return {kind: int(counts[kind]) for kind in sorted(counts)}

    def total(self) -> int:
        return len(self._records)

    def entries(self) -> list[dict[str, Any]]:
        return [
            {"kind": kind, "id": identifier, "label": label}
            for (kind, identifier), label in sorted(self._records.items())
        ]

    def difference(
        self, before: dict[str, int], after: dict[str, int] | None = None
    ) -> dict[str, int]:
        after = self.snapshot() if after is None else after
        keys = set(before) | set(after)
        return {
            key: int(after.get(key, 0) - before.get(key, 0))
            for key in sorted(keys)
            if after.get(key, 0) != before.get(key, 0)
        }


class OpenGLContext:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.window: Any = None
        self.version = ""
        self.major_version = 0
        self.minor_version = 0
        self.renderer = ""
        self.vendor = ""
        if not glfw.init():
            error = glfw.get_error()
            raise GLContextError(
                "GLFW initialization failed.",
                details={"glfw_error": list(error) if error else None},
            )
        try:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
            self.window = glfw.create_window(width, height, "material-graph-offscreen", None, None)
            if not self.window:
                error = glfw.get_error()
                raise GLContextError(
                    "Failed to create an OpenGL 3.3 core context.",
                    details={"glfw_error": list(error) if error else None},
                )
            glfw.make_context_current(self.window)
            if glfw.get_current_context() is None:
                error = glfw.get_error()
                raise GLContextError(
                    "Failed to make the OpenGL context current.",
                    details={"glfw_error": list(error) if error else None},
                )
            glfw.swap_interval(0)
            self.version = self._string(GL_VERSION)
            self.renderer = self._string(GL_RENDERER)
            self.vendor = self._string(GL_VENDOR)
            self.major_version = int(glGetIntegerv(GL_MAJOR_VERSION))
            self.minor_version = int(glGetIntegerv(GL_MINOR_VERSION))
            if (self.major_version, self.minor_version) < (3, 3):
                raise GLContextError(
                    "OpenGL 3.3 or newer is required.",
                    details={
                        "major": self.major_version,
                        "minor": self.minor_version,
                        "version_string": self.version,
                    },
                )
        except Exception:
            self.close()
            raise

    @staticmethod
    def _string(name: int) -> str:
        value = glGetString(name)
        return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)

    def close(self) -> None:
        if self.window is not None:
            glfw.destroy_window(self.window)
            self.window = None
        glfw.terminate()

    def __enter__(self) -> "OpenGLContext":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _check_gl(stage: str) -> None:
    errors: list[str] = []
    while True:
        error = glGetError()
        if error == GL_NO_ERROR:
            break
        errors.append(hex(int(error)))
    if errors:
        raise RenderError(
            f"OpenGL error after {stage}: {', '.join(errors)}",
            details={"stage": stage, "gl_errors": errors},
        )


def _diagnostic_records(log: str, source: str) -> list[dict[str, Any]]:
    source_lines = source.splitlines()
    records: list[dict[str, Any]] = []
    line_patterns = (
        re.compile(r":(\d+)\(\d+\)\s*:"),
        re.compile(r":(\d+)\s*:\s*(?:error|warning)"),
        re.compile(r"\((\d+)\)\s*:\s*(?:error|warning)"),
    )
    for raw_line in log.splitlines():
        text = raw_line.strip()
        if not text:
            continue
        line_number: int | None = None
        for pattern in line_patterns:
            match = pattern.search(text)
            if match:
                line_number = int(match.group(1))
                break
        source_line = None
        if line_number is not None and 1 <= line_number <= len(source_lines):
            source_line = source_lines[line_number - 1]
        records.append(
            {
                "line": line_number,
                "source": source_line,
                "message": text,
            }
        )
    return records


class GLProgram:
    def __init__(
        self,
        ledger: ResourceLedger,
        sources: ShaderSources,
        *,
        label: str,
    ):
        self.ledger = ledger
        self.sources = sources
        self.label = label
        self.program = 0
        self.uniform_locations: dict[str, int] = {}
        self.sampler_locations: dict[str, int] = {}
        self.material_block_index = -1
        self.scene_block_index = -1
        self._build()

    def _build(self) -> None:
        vertex = self._compile(GL_VERTEX_SHADER, self.sources.vertex, "vertex")
        try:
            fragment = self._compile(
                GL_FRAGMENT_SHADER, self.sources.fragment, "fragment"
            )
        except Exception:
            glDeleteShader(vertex)
            self.ledger.release("shader", vertex)
            raise
        program = _gl_id(glCreateProgram())
        self.ledger.allocate("program", program, self.label)
        try:
            glAttachShader(program, vertex)
            glAttachShader(program, fragment)
            glLinkProgram(program)
            linked = glGetProgramiv(program, GL_LINK_STATUS)
            if not linked:
                log = _decode_log(glGetProgramInfoLog(program))
                raise ShaderLinkError(
                    f"Shader link failed for {self.label}.",
                    details={
                        "stage": "link",
                        "raw_log": log,
                        "diagnostics": _diagnostic_records(log, self.sources.fragment),
                        "vertex_source": self.sources.vertex,
                        "fragment_source": self.sources.fragment,
                    },
                )
            self.program = program
            self._discover_uniforms()
        except Exception:
            glDeleteProgram(program)
            self.ledger.release("program", program)
            raise
        finally:
            for shader in (vertex, fragment):
                glDeleteShader(shader)
                self.ledger.release("shader", shader)

    def _compile(self, stage: int, source: str, stage_name: str) -> int:
        shader = _gl_id(glCreateShader(stage))
        self.ledger.allocate("shader", shader, f"{self.label}:{stage_name}")
        glShaderSource(shader, source)
        glCompileShader(shader)
        compiled = glGetShaderiv(shader, GL_COMPILE_STATUS)
        if not compiled:
            log = _decode_log(glGetShaderInfoLog(shader))
            glDeleteShader(shader)
            self.ledger.release("shader", shader)
            raise ShaderCompileError(
                f"{stage_name} shader compilation failed for {self.label}.",
                details={
                    "stage": stage_name,
                    "raw_log": log,
                    "diagnostics": _diagnostic_records(log, source),
                    "source": source,
                },
            )
        return shader

    def _discover_uniforms(self) -> None:
        for name in ("u_channel",):
            self.uniform_locations[name] = glGetUniformLocation(self.program, name)
        self.material_block_index = glGetUniformBlockIndex(
            self.program, "MaterialParameters"
        )
        self.scene_block_index = glGetUniformBlockIndex(self.program, "SceneParameters")
        if int(self.material_block_index) != 0xFFFFFFFF:
            glUniformBlockBinding(self.program, self.material_block_index, 0)
        if int(self.scene_block_index) != 0xFFFFFFFF:
            glUniformBlockBinding(self.program, self.scene_block_index, 1)

    def register_sampler(self, uniform_name: str) -> int:
        if uniform_name not in self.sampler_locations:
            self.sampler_locations[uniform_name] = glGetUniformLocation(
                self.program, uniform_name
            )
        return self.sampler_locations[uniform_name]

    def release(self) -> None:
        if self.program:
            glDeleteProgram(self.program)
            self.ledger.release("program", self.program)
            self.program = 0


def _decode_log(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


class GLTexture:
    def __init__(
        self,
        ledger: ResourceLedger,
        *,
        width: int,
        height: int,
        internal_format: int,
        data: bytes | None,
        label: str,
        filter_mode: int = GL_LINEAR,
    ):
        self.ledger = ledger
        self.width = width
        self.height = height
        self.texture = _gl_id(glGenTextures(1))
        ledger.allocate("texture", self.texture, label)
        try:
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter_mode)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter_mode)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                internal_format,
                width,
                height,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE if data is not None else GL_FLOAT,
                data,
            )
            glBindTexture(GL_TEXTURE_2D, 0)
            _check_gl(f"texture allocation {label}")
        except Exception:
            glBindTexture(GL_TEXTURE_2D, 0)
            glDeleteTextures(1, [self.texture])
            self.ledger.release("texture", self.texture)
            self.texture = 0
            raise

    def release(self) -> None:
        if self.texture:
            glDeleteTextures(1, [self.texture])
            self.ledger.release("texture", self.texture)
            self.texture = 0


class GLBuffer:
    def __init__(
        self,
        ledger: ResourceLedger,
        data: bytes,
        *,
        target: int = GL_UNIFORM_BUFFER,
        label: str,
    ):
        self.ledger = ledger
        self.target = target
        self.buffer = _gl_id(glGenBuffers(1))
        ledger.allocate("buffer", self.buffer, label)
        try:
            glBindBuffer(target, self.buffer)
            glBufferData(target, len(data), data, GL_DYNAMIC_DRAW)
            glBindBuffer(target, 0)
            _check_gl(f"buffer allocation {label}")
        except Exception:
            glBindBuffer(target, 0)
            glDeleteBuffers(1, [self.buffer])
            self.ledger.release("buffer", self.buffer)
            self.buffer = 0
            raise

    def update(self, data: bytes) -> None:
        glBindBuffer(self.target, self.buffer)
        glBufferData(self.target, len(data), data, GL_DYNAMIC_DRAW)
        glBindBuffer(self.target, 0)

    def release(self) -> None:
        if self.buffer:
            glDeleteBuffers(1, [self.buffer])
            self.ledger.release("buffer", self.buffer)
            self.buffer = 0


class GLFramebuffer:
    def __init__(
        self,
        ledger: ResourceLedger,
        *,
        width: int,
        height: int,
        label: str,
        with_aux: bool,
    ):
        self.ledger = ledger
        self.width = width
        self.height = height
        self.textures: list[GLTexture] = []
        self.framebuffer = _gl_id(glGenFramebuffers(1))
        ledger.allocate("framebuffer", self.framebuffer, label)
        try:
            glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer)
            channel = GLTexture(
                ledger,
                width=width,
                height=height,
                internal_format=GL_RGBA8,
                data=bytes(width * height * 4),
                label=f"{label}:channel",
                filter_mode=GL_NEAREST,
            )
            self.textures.append(channel)
            glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D,
                channel.texture,
                0,
            )
            attachments = [GL_COLOR_ATTACHMENT0]
            if with_aux:
                for index, name in enumerate(
                    ("roughness", "metallic", "normal"), start=1
                ):
                    texture = GLTexture(
                        ledger,
                        width=width,
                        height=height,
                        internal_format=GL_RGBA32F,
                        data=None,
                        label=f"{label}:{name}",
                        filter_mode=GL_NEAREST,
                    )
                    self.textures.append(texture)
                    glFramebufferTexture2D(
                        GL_FRAMEBUFFER,
                        GL_COLOR_ATTACHMENT0 + index,
                        GL_TEXTURE_2D,
                        texture.texture,
                        0,
                    )
                    attachments.append(GL_COLOR_ATTACHMENT0 + index)
            glDrawBuffers(len(attachments), attachments)
            status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            if status != GL_FRAMEBUFFER_COMPLETE:
                raise RenderError(
                    f"Framebuffer {label} is incomplete: {hex(int(status))}",
                    details={"status": hex(int(status))},
                )
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            _check_gl(f"framebuffer allocation {label}")
        except Exception:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            self.release()
            raise

    def release(self) -> None:
        for texture in self.textures:
            texture.release()
        self.textures.clear()
        if self.framebuffer:
            glDeleteFramebuffers(1, [self.framebuffer])
            self.ledger.release("framebuffer", self.framebuffer)
            self.framebuffer = 0


class TextureCache:
    def __init__(self, ledger: ResourceLedger):
        self.ledger = ledger
        self.textures: dict[str, GLTexture] = {}
        self._prewarm()

    def _prewarm(self) -> None:
        try:
            for name in BUILTIN_TEXTURES:
                data = _generate_texture(name, 256)
                self.textures[name] = GLTexture(
                    self.ledger,
                    width=256,
                    height=256,
                    internal_format=GL_RGBA8,
                    data=data,
                    label=f"builtin:{name}",
                )
        except Exception:
            self.release()
            raise

    def get(self, name: str) -> GLTexture:
        if name not in self.textures:
            raise RenderError(f"Texture {name!r} is not available.")
        return self.textures[name]

    def release(self) -> None:
        for texture in self.textures.values():
            texture.release()
        self.textures.clear()


def _generate_texture(name: str, size: int) -> bytes:
    if name == "checker":
        block = max(1, size // 16)
        y, x = np.mgrid[0:size, 0:size]
        parity = ((x // block) + (y // block)) % 2
        array = np.empty((size, size, 4), dtype=np.uint8)
        array[..., 0] = np.where(parity == 0, 220, 34)
        array[..., 1] = np.where(parity == 0, 188, 82)
        array[..., 2] = np.where(parity == 0, 110, 132)
        array[..., 3] = 255
        return array.tobytes()
    if name == "noise":
        y, x = np.mgrid[0:size, 0:size]
        xx = x.astype(np.uint64)
        yy = y.astype(np.uint64)
        hashed = xx * np.uint64(374761393) + yy * np.uint64(668265263)
        hashed = (hashed ^ (hashed >> np.uint64(13))) * np.uint64(1274126177)
        hashed ^= hashed >> np.uint64(16)
        low = (hashed & np.uint64(0xFFFF)).astype(np.float64) / 65535.0
        high = ((hashed >> np.uint64(8)) & np.uint64(0xFFFF)).astype(np.float64) / 65535.0
        array = np.empty((size, size, 4), dtype=np.uint8)
        array[..., 0] = np.clip(low * 255.0, 0, 255).astype(np.uint8)
        array[..., 1] = np.clip(high * 255.0, 0, 255).astype(np.uint8)
        array[..., 2] = np.clip((0.35 + 0.65 * low * high) * 255.0, 0, 255).astype(
            np.uint8
        )
        array[..., 3] = 255
        return array.tobytes()
    if name == "normal_tile":
        u = np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float64)
        v = np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
        phase_u = 2.0 * math.pi * 5.0 * uu
        phase_v = 2.0 * math.pi * 4.0 * vv
        height = 0.18 * np.sin(phase_u) * np.cos(phase_v) + 0.08 * np.sin(
            2.0 * phase_u + phase_v
        )
        du = (
            0.18
            * 2.0
            * math.pi
            * 5.0
            * np.cos(phase_u)
            * np.cos(phase_v)
            + 0.08
            * 4.0
            * math.pi
            * np.cos(2.0 * phase_u + phase_v)
        )
        dv = (
            -0.18
            * 2.0
            * math.pi
            * 4.0
            * np.sin(phase_u)
            * np.sin(phase_v)
            + 0.08
            * 2.0
            * math.pi
            * np.cos(2.0 * phase_u + phase_v)
        )
        nx = -du * 0.08
        ny = -dv * 0.08
        nz = np.ones_like(nx)
        length = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / length, ny / length, nz / length
        array = np.empty((size, size, 4), dtype=np.uint8)
        array[..., 0] = np.clip((nx * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        array[..., 1] = np.clip((ny * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        array[..., 2] = np.clip((nz * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        array[..., 3] = 255
        return array.tobytes()
    if name == "uv_grid":
        y, x = np.mgrid[0:size, 0:size]
        array = np.empty((size, size, 4), dtype=np.uint8)
        array[..., 0] = (
            x.astype(np.float64) / max(1, size - 1) * 255.0
        ).astype(np.uint8)
        array[..., 1] = (
            y.astype(np.float64) / max(1, size - 1) * 255.0
        ).astype(np.uint8)
        block = max(1, size // 8)
        parity = ((x // block) + (y // block)) % 2
        array[..., 2] = np.where(parity == 0, 64, 224).astype(np.uint8)
        array[..., 3] = 255
        return array.tobytes()
    raise RenderError(f"Unknown built-in texture {name!r}")


def _camera_basis(camera: CameraSpec) -> tuple[np.ndarray, ...]:
    position = np.asarray(camera.position, dtype=np.float64)
    target = np.asarray(camera.target, dtype=np.float64)
    up = np.asarray(camera.up, dtype=np.float64)
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    normalized_up = np.cross(right, forward)
    normalized_up /= np.linalg.norm(normalized_up)
    return position, forward, right, normalized_up


def _scene_uniform_bytes(camera: CameraSpec, light: LightSpec, aspect: float) -> bytes:
    position, forward, right, up = _camera_basis(camera)
    direction = np.asarray(light.direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    color = np.asarray(light.color, dtype=np.float64)
    tan_half_fov = math.tan(math.radians(camera.fov_degrees) * 0.5)
    data = np.zeros(28, dtype=np.float32)
    data[0:3] = position
    data[3] = 1.0
    data[4:7] = forward
    data[8:11] = right
    data[12:15] = up
    data[16] = tan_half_fov
    data[17] = aspect
    data[20:23] = direction
    data[24:27] = color
    data[27] = light.intensity
    return data.tobytes()


def _material_uniform_bytes(graph: ParsedGraph) -> bytes:
    layout = material_uniform_layout(graph)
    data = np.zeros(layout.size // 4, dtype=np.float32)
    for name, spec in graph.parameters.items():
        offset = layout.offsets[name] // 4
        data[offset : offset + len(spec.value)] = np.asarray(spec.value, dtype=np.float32)
    return data.tobytes()


class MaterialSession:
    def __init__(
        self,
        renderer: "Renderer",
        graph: ParsedGraph,
        *,
        output_dir: Path,
        size: int,
    ):
        self.renderer = renderer
        self.graph = graph
        self.output_dir = output_dir
        self.size = size
        self.program: GLProgram | None = None
        self.framebuffer: GLFramebuffer | None = None
        self.material_ubo: GLBuffer | None = None
        self.scene_ubo: GLBuffer | None = None
        self.sources: ShaderSources | None = None
        self.baseline = renderer.resource_counts()
        try:
            self.sources = generate_shaders(graph)
            self._write_sources(self.sources)
            self._build_initial_program()
            self.framebuffer = GLFramebuffer(
                renderer.ledger,
                width=size,
                height=size,
                label=f"{graph.graph_id}:mrt",
                with_aux=True,
            )
            self.material_ubo = GLBuffer(
                renderer.ledger,
                _material_uniform_bytes(graph),
                label=f"{graph.graph_id}:material-ubo",
            )
            self.scene_ubo = GLBuffer(
                renderer.ledger,
                _scene_uniform_bytes(graph.camera, graph.light, 1.0),
                label=f"{graph.graph_id}:scene-ubo",
            )
        except Exception:
            self.release()
            raise

    def _write_sources(self, sources: ShaderSources) -> None:
        shader_dir = self.output_dir / "shaders"
        shader_dir.mkdir(parents=True, exist_ok=True)
        (shader_dir / "material.vert").write_text(sources.vertex, encoding="utf-8")
        (shader_dir / "material.frag").write_text(sources.fragment, encoding="utf-8")

    def _build_initial_program(self) -> None:
        assert self.sources is not None
        self.program = GLProgram(
            self.renderer.ledger,
            self.sources,
            label=f"{self.graph.graph_id}:program",
        )

    def reload(
        self, graph: ParsedGraph, *, write_sources: bool = True
    ) -> dict[str, Any]:
        before = self.renderer.resource_counts()
        sources = generate_shaders(graph)
        if write_sources:
            self._write_sources(sources)
        new_program: GLProgram | None = None
        try:
            new_program = GLProgram(
                self.renderer.ledger,
                sources,
                label=f"{graph.graph_id}:program",
            )
        except Exception:
            after_failed = self.renderer.resource_counts()
            raise
        old_program = self.program
        self.program = new_program
        self.graph = graph
        self.sources = sources
        if old_program is not None:
            old_program.release()
        if self.material_ubo is not None:
            self.material_ubo.update(_material_uniform_bytes(graph))
        if self.scene_ubo is not None:
            self.scene_ubo.update(
                _scene_uniform_bytes(graph.camera, graph.light, 1.0)
            )
        after = self.renderer.resource_counts()
        return {
            "attempted": True,
            "resource_counts_before": before,
            "resource_counts_after": after,
            "equal": before == after,
            "source_written": bool(write_sources),
        }

    def render_channel(self, channel: str) -> ChannelRenderResult:
        if channel not in CHANNELS:
            raise RenderError(f"Unknown render channel {channel!r}.")
        if self.program is None or self.framebuffer is None:
            raise RenderError("Material session has no active program or framebuffer.")
        if self.material_ubo is None or self.scene_ubo is None:
            raise RenderError("Material session is missing a uniform buffer.")

        base_temp_count = self.renderer.resource_counts().get("framebuffer", 0)
        temp: GLFramebuffer | None = None
        try:
            temp = GLFramebuffer(
                self.renderer.ledger,
                width=self.size,
                height=self.size,
                label=f"{self.graph.graph_id}:{channel}:readback",
                with_aux=False,
            )
            glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer.framebuffer)
            glViewport(0, 0, self.size, self.size)
            glDisable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
            glUseProgram(self.program.program)
            glBindVertexArray(self.renderer.vao)
            glBindBufferBase(GL_UNIFORM_BUFFER, 0, self.material_ubo.buffer)
            glBindBufferBase(GL_UNIFORM_BUFFER, 1, self.scene_ubo.buffer)
            channel_location = self.program.uniform_locations.get("u_channel", -1)
            if channel_location >= 0:
                glUniform1i(channel_location, CHANNEL_INDEX[channel])

            textures = sorted(self._referenced_textures())
            for unit, texture_name in enumerate(textures):
                texture = self.renderer.textures.get(texture_name)
                glActiveTexture(GL_TEXTURE0 + unit)
                glBindTexture(GL_TEXTURE_2D, texture.texture)
                location = self.program.register_sampler(
                    self.graph.texture_uniform(texture_name)
                )
                if location >= 0:
                    glUniform1i(location, unit)

            glDrawArrays(GL_TRIANGLES, 0, 3)
            glFinish()
            _check_gl(f"render {self.graph.graph_id}:{channel}")

            glBindFramebuffer(GL_READ_FRAMEBUFFER, self.framebuffer.framebuffer)
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, temp.framebuffer)
            glReadBuffer(GL_COLOR_ATTACHMENT0)
            glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])
            glBlitFramebuffer(
                0,
                0,
                self.size,
                self.size,
                0,
                0,
                self.size,
                self.size,
                GL_COLOR_BUFFER_BIT,
                GL_NEAREST,
            )
            glBindFramebuffer(GL_FRAMEBUFFER, temp.framebuffer)
            glReadBuffer(GL_COLOR_ATTACHMENT0)
            channel_pixels = np.empty((self.size, self.size, 4), dtype=np.uint8)
            glReadPixels(
                0,
                0,
                self.size,
                self.size,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                channel_pixels,
            )

            glBindFramebuffer(GL_FRAMEBUFFER, self.framebuffer.framebuffer)
            aux_arrays: list[np.ndarray] = []
            for attachment in (GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3):
                glReadBuffer(attachment)
                aux = np.empty((self.size, self.size, 4), dtype=np.float32)
                glReadPixels(
                    0,
                    0,
                    self.size,
                    self.size,
                    GL_RGBA,
                    GL_FLOAT,
                    aux,
                )
                aux_arrays.append(np.ascontiguousarray(np.flipud(aux)))
            _check_gl(f"readback {self.graph.graph_id}:{channel}")
        finally:
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            glUseProgram(0)
            glBindVertexArray(0)
            if temp is not None:
                temp.release()
        after_temp_count = self.renderer.resource_counts().get("framebuffer", 0)
        if after_temp_count != base_temp_count:
            raise RenderError(
                "Temporary framebuffer was not released after rendering.",
                details={
                    "channel": channel,
                    "before": base_temp_count,
                    "after": after_temp_count,
                },
            )

        image_path = self.output_dir / f"{channel}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.flipud(channel_pixels)[..., :3]
        Image.fromarray(image, mode="RGB").save(
            image_path, format="PNG", optimize=False, compress_level=6
        )
        stats = compute_png_stats(image_path)
        return ChannelRenderResult(
            channel=channel,
            image_path=image_path,
            stats=stats,
            roughness=aux_arrays[0],
            metallic=aux_arrays[1],
            normal=aux_arrays[2],
        )

    def _referenced_textures(self) -> set[str]:
        return {
            str(self.graph.nodes[node_id].params.get("texture"))
            for node_id in self.graph.topo_order
            if self.graph.nodes[node_id].node_type == "texture"
        }

    def resource_counts(self) -> dict[str, int]:
        return self.renderer.resource_counts()

    def release(self) -> None:
        if self.program is not None:
            self.program.release()
            self.program = None
        if self.material_ubo is not None:
            self.material_ubo.release()
            self.material_ubo = None
        if self.scene_ubo is not None:
            self.scene_ubo.release()
            self.scene_ubo = None
        if self.framebuffer is not None:
            self.framebuffer.release()
            self.framebuffer = None


class Renderer:
    def __init__(self, *, size: int = 512):
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size
        self.context = OpenGLContext(size, size)
        self.ledger = ResourceLedger()
        try:
            self.textures = TextureCache(self.ledger)
            self.vao = _gl_id(glGenVertexArrays(1))
            self.ledger.allocate("vertex_array", self.vao, "fullscreen-triangle")
            _check_gl("renderer initialization")
        except Exception:
            self.close()
            raise

    @property
    def context_info(self) -> dict[str, str]:
        return {
            "version": self.context.version,
            "major": str(self.context.major_version),
            "minor": str(self.context.minor_version),
            "renderer": self.context.renderer,
            "vendor": self.context.vendor,
        }

    def create_session(
        self, graph: ParsedGraph, *, output_dir: Path, size: int | None = None
    ) -> MaterialSession:
        return MaterialSession(
            self,
            graph,
            output_dir=output_dir,
            size=self.size if size is None else size,
        )

    def resource_counts(self) -> dict[str, int]:
        return self.ledger.snapshot()

    def resource_entries(self) -> list[dict[str, Any]]:
        return self.ledger.entries()

    def close(self) -> None:
        for attribute in ("textures",):
            value = getattr(self, attribute, None)
            if value is not None:
                value.release()
                setattr(self, attribute, None)
        vao = getattr(self, "vao", 0)
        if vao:
            glDeleteVertexArrays(1, [vao])
            try:
                self.ledger.release("vertex_array", vao)
            except RuntimeError:
                pass
            self.vao = 0
        self.context.close()

    def __enter__(self) -> "Renderer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def compute_png_stats(path: Path) -> dict[str, Any]:
    """Re-open the written PNG and compute statistics from its decoded pixels."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    flat = pixels.reshape(-1)
    per_channel = {}
    for index, name in enumerate(("r", "g", "b")):
        channel = pixels[..., index].reshape(-1)
        per_channel[name] = {
            "mean": float(np.mean(channel)),
            "variance": float(np.var(channel)),
            "min": float(np.min(channel)),
            "max": float(np.max(channel)),
        }
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "mean": float(np.mean(flat)),
        "variance": float(np.var(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "per_channel": per_channel,
        "pixel_count": int(pixels.shape[0] * pixels.shape[1]),
        "sha256": digest,
    }

