"""Offscreen software-rendered previews.

Two software backends are attempted in order:

* OSMesa (``osmesa32.dll`` on Windows, ``libOSMesa.so.8`` elsewhere), the
  classic headless Mesa backend;
* EGL with a pbuffer plus ``llvmpipe`` (``LIBGL_ALWAYS_SOFTWARE=1``), the
  equivalent software path available on modern Mesa/Windows builds.

All GL calls go through ``ctypes`` so the exact backend library that was
loaded is known and failures produce a precise diagnostic. If no backend
can create a context, :class:`PreviewUnavailable` is raised with the
diagnostic while core baking artifacts remain untouched.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os

import numpy as np
from PIL import Image

from .errors import PreviewUnavailable
from .lod import INDEX_DTYPE, VERTEX_DTYPE

GL_FALSE = 0
GL_FLOAT = 0x1406
GL_UNSIGNED_INT = 0x1405
GL_TRIANGLES = 0x0004
GL_DEPTH_TEST = 0x0B71
GL_CULL_FACE = 0x0B44
GL_COLOR_BUFFER_BIT = 0x4000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_VERTEX_SHADER = 0x8B31
GL_FRAGMENT_SHADER = 0x8B30
GL_ARRAY_BUFFER = 0x8892
GL_ELEMENT_ARRAY_BUFFER = 0x8893
GL_STATIC_DRAW = 0x88E4

_DIAGNOSTICS: list[str] = []


def diagnostics() -> list[str]:
    return list(_DIAGNOSTICS)


def _load_first(candidates: list[str]) -> ctypes.CDLL | None:
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    resolved = ctypes.util.find_library(candidates[0])
    if resolved:
        try:
            return ctypes.CDLL(resolved)
        except OSError:
            return None
    return None


class _GL:
    """Minimal function bindings shared by the OSMesa and EGL backends."""

    def __init__(self, lib: ctypes.CDLL) -> None:
        self.lib = lib
        for name in (
            "glClearColor", "glClear", "glEnable", "glDisable",
            "glViewport", "glCreateShader", "glShaderSource",
            "glCompileShader", "glGetShaderiv", "glGetShaderInfoLog",
            "glCreateProgram", "glAttachShader", "glLinkProgram",
            "glGetProgramiv", "glGetProgramInfoLog", "glUseProgram",
            "glGenBuffers", "glBindBuffer", "glBufferData",
            "glEnableVertexAttribArray", "glVertexAttribPointer",
            "glGetUniformLocation", "glUniformMatrix4fv", "glUniform3f",
            "glUniform1i", "glDrawElements", "glReadPixels", "glFinish",
            "glGenVertexArrays", "glBindVertexArray",
        ):
            setattr(self, name, getattr(lib, name))
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        lib = self.lib
        lib.glClearColor.argtypes = [ctypes.c_float] * 4
        lib.glClear.argtypes = [ctypes.c_uint]
        lib.glEnable.argtypes = [ctypes.c_uint]
        lib.glDisable.argtypes = [ctypes.c_uint]
        lib.glViewport.argtypes = [ctypes.c_int] * 4
        lib.glCreateShader.restype = ctypes.c_uint
        lib.glShaderSource.argtypes = [
            ctypes.c_uint, ctypes.c_size_t, ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_void_p,
        ]
        lib.glCompileShader.argtypes = [ctypes.c_uint]
        lib.glGetShaderiv.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
        lib.glGetShaderInfoLog.argtypes = [
            ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_char_p,
        ]
        lib.glCreateProgram.restype = ctypes.c_uint
        lib.glAttachShader.argtypes = [ctypes.c_uint, ctypes.c_uint]
        lib.glLinkProgram.argtypes = [ctypes.c_uint]
        lib.glGetProgramiv.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
        lib.glGetProgramInfoLog.argtypes = [
            ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_char_p,
        ]
        lib.glUseProgram.argtypes = [ctypes.c_uint]
        lib.glGenBuffers.argtypes = [ctypes.c_size_t, ctypes.c_void_p]
        lib.glBindBuffer.argtypes = [ctypes.c_uint, ctypes.c_uint]
        lib.glBufferData.argtypes = [
            ctypes.c_uint, ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_uint,
        ]
        lib.glEnableVertexAttribArray.argtypes = [ctypes.c_uint]
        lib.glVertexAttribPointer.argtypes = [
            ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_ubyte,
            ctypes.c_size_t, ctypes.c_void_p,
        ]
        lib.glGetUniformLocation.restype = ctypes.c_int
        lib.glGetUniformLocation.argtypes = [ctypes.c_uint, ctypes.c_char_p]
        lib.glUniformMatrix4fv.argtypes = [
            ctypes.c_int, ctypes.c_size_t, ctypes.c_ubyte, ctypes.c_char_p,
        ]
        lib.glUniform3f.argtypes = [
            ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ]
        lib.glUniform1i.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.glDrawElements.argtypes = [
            ctypes.c_uint, ctypes.c_size_t, ctypes.c_uint, ctypes.c_void_p,
        ]
        lib.glReadPixels.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
        ]
        lib.glFinish.restype = None
        lib.glGenVertexArrays.argtypes = [ctypes.c_size_t, ctypes.c_void_p]
        lib.glBindVertexArray.argtypes = [ctypes.c_uint]


class _OffscreenContext:
    def make_current(self) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


def _try_osmesa(width: int, height: int) -> _OffscreenContext | None:
    osmesa = _load_first(
        ["osmesa32.dll", "libOSMesa.so.8", "libOSMesa.so.6", "OSMesa"]
    )
    if osmesa is None:
        _DIAGNOSTICS.append("OSMesa: shared library not found")
        return None
    gl_lib = _load_first(["opengl32.dll", "libGL.so.1", "libGL.so"])
    if gl_lib is None:
        _DIAGNOSTICS.append("OSMesa: OpenGL runtime library not found")
        return None
    try:
        OSMESA_RGBA = 0x1908
        OSMESA_DEPTH_BITS = 0x0030
        OSMESA_PROFILE = 0x2202
        OSMESA_CORE_PROFILE = 0x2203
        OSMESA_CONTEXT_MAJOR_VERSION = 0x2200
        OSMESA_CONTEXT_MINOR_VERSION = 0x2201
        attrs = (ctypes.c_int * 12)(
            OSMESA_DEPTH_BITS, 24,
            OSMESA_PROFILE, OSMESA_CORE_PROFILE,
            OSMESA_CONTEXT_MAJOR_VERSION, 3,
            OSMESA_CONTEXT_MINOR_VERSION, 3,
            0, 0, 0, 0,
        )
        create_attribs = getattr(osmesa, "OSMesaCreateContextAttribs", None)
        if create_attribs is None:
            context = osmesa.OSMesaCreateContext(OSMESA_RGBA, None)
        else:
            context = create_attribs(attrs, None)
        if not context:
            _DIAGNOSTICS.append("OSMesa: context creation returned null")
            return None
        buffer = (ctypes.c_ubyte * (width * height * 4))()
        if not osmesa.OSMesaMakeCurrent(
            context, buffer, ctypes.c_uint(0x1401), width, height
        ):
            _DIAGNOSTICS.append("OSMesa: OSMesaMakeCurrent failed")
            return None
        gl = _GL(gl_lib)
        return _OSMesaContext(osmesa, context, buffer, gl)
    except (OSError, AttributeError) as exc:
        _DIAGNOSTICS.append(f"OSMesa: {type(exc).__name__}: {exc}")
        return None


class _OSMesaContext(_OffscreenContext):
    def __init__(self, osmesa, context, buffer, gl) -> None:
        self.osmesa = osmesa
        self.context = context
        self.buffer = buffer
        self.gl = gl

    def make_current(self) -> None:
        return None

    def shutdown(self) -> None:
        destroy = getattr(self.osmesa, "OSMesaDestroyContext", None)
        if destroy:
            destroy(self.context)


def _try_egl(width: int, height: int) -> _OffscreenContext | None:
    egl_lib = _load_first(["libEGL.so.1", "libEGL.dll", "libEGL.so"])
    gl_lib = _load_first(["opengl32.dll", "libGL.so.1", "libGL.so"])
    if egl_lib is None or gl_lib is None:
        _DIAGNOSTICS.append("EGL: libEGL or OpenGL runtime not found")
        return None
    egl = egl_lib
    try:
        EGL_DEFAULT_DISPLAY = ctypes.c_void_p(0)
        EGL_OPENGL_API = 0x30A2
        EGL_PBUFFER_BIT = 0x0001
        EGL_OPENGL_BIT = 0x0008
        EGL_NONE = 0x3038
        EGL_SURFACE_TYPE = 0x3033
        EGL_RENDERABLE_TYPE = 0x3040
        EGL_RED_SIZE, EGL_GREEN_SIZE, EGL_BLUE_SIZE = 0x3024, 0x3023, 0x3022
        EGL_DEPTH_SIZE, EGL_WIDTH, EGL_HEIGHT = 0x3025, 0x3057, 0x3056
        egl.eglGetDisplay.restype = ctypes.c_void_p
        display = egl.eglGetDisplay(EGL_DEFAULT_DISPLAY)
        if not display:
            _DIAGNOSTICS.append("EGL: eglGetDisplay returned null")
            return None
        major, minor = ctypes.c_int(), ctypes.c_int()
        if not egl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
            _DIAGNOSTICS.append("EGL: eglInitialize failed")
            return None
        if not egl.eglBindAPI(EGL_OPENGL_API):
            _DIAGNOSTICS.append("EGL: eglBindAPI(OpenGL) failed")
            return None
        config = ctypes.c_void_p()
        count = ctypes.c_int()
        config_attrs = (ctypes.c_int * 13)(
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
            EGL_DEPTH_SIZE, 24, EGL_NONE,
        )
        if not egl.eglChooseConfig(
            display, config_attrs, ctypes.byref(config), 1, ctypes.byref(count)
        ) or count.value < 1:
            _DIAGNOSTICS.append("EGL: eglChooseConfig found no config")
            return None
        surface_attrs = (ctypes.c_int * 5)(
            EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE
        )
        egl.eglCreatePbufferSurface.restype = ctypes.c_void_p
        surface = egl.eglCreatePbufferSurface(display, config, surface_attrs)
        if not surface:
            _DIAGNOSTICS.append("EGL: eglCreatePbufferSurface failed")
            return None
        egl.eglCreateContext.restype = ctypes.c_void_p
        context = egl.eglCreateContext(display, config, None, None)
        if not context:
            _DIAGNOSTICS.append("EGL: eglCreateContext failed")
            return None
        if not egl.eglMakeCurrent(display, surface, surface, context):
            _DIAGNOSTICS.append("EGL: eglMakeCurrent failed")
            return None
        gl = _GL(gl_lib)
        return _EGLContext(egl, display, surface, context, gl)
    except (OSError, AttributeError) as exc:
        _DIAGNOSTICS.append(f"EGL: {type(exc).__name__}: {exc}")
        return None


class _EGLContext(_OffscreenContext):
    def __init__(self, egl, display, surface, context, gl) -> None:
        self.egl = egl
        self.display = display
        self.surface = surface
        self.context = context
        self.gl = gl

    def make_current(self) -> None:
        return None

    def shutdown(self) -> None:
        self.egl.eglMakeCurrent(self.display, None, None, None)
        self.egl.eglDestroyContext(self.display, self.context)
        self.egl.eglDestroySurface(self.display, self.surface)
        self.egl.eglTerminate(self.display)


_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;
uniform mat4 uMVP;
out vec3 vNormal;
out vec2 vUv;
void main() {
    vNormal = aNormal;
    vUv = aUv;
    gl_Position = uMVP * vec4(aPosition, 1.0);
}
"""

_DIFFUSE_FRAGMENT = """
#version 330 core
in vec3 vNormal;
in vec2 vUv;
uniform vec3 uLightDir;
out vec4 fragColor;
void main() {
    vec3 base = mix(vec3(0.16, 0.30, 0.12), vec3(0.55, 0.48, 0.32), vUv.y);
    float diffuse = max(dot(normalize(vNormal), normalize(uLightDir)), 0.0);
    fragColor = vec4(base * (0.25 + 0.75 * diffuse), 1.0);
}
"""

_NORMAL_FRAGMENT = """
#version 330 core
in vec3 vNormal;
in vec2 vUv;
out vec4 fragColor;
void main() {
    fragColor = vec4(normalize(vNormal) * 0.5 + 0.5, 1.0);
}
"""


def _orthographic_matrix(half_extent: float, near: float, far: float) -> np.ndarray:
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 1.0 / half_extent
    m[1, 1] = 1.0 / half_extent
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


def _rotation_matrix() -> np.ndarray:
    angle = np.deg2rad(35.0)
    rot_x = np.eye(4, dtype=np.float32)
    rot_x[1, 1] = np.cos(angle)
    rot_x[1, 2] = -np.sin(angle)
    rot_x[2, 1] = np.sin(angle)
    rot_x[2, 2] = np.cos(angle)
    angle_z = np.deg2rad(25.0)
    rot_z = np.eye(4, dtype=np.float32)
    rot_z[0, 0] = np.cos(angle_z)
    rot_z[0, 1] = -np.sin(angle_z)
    rot_z[1, 0] = np.sin(angle_z)
    rot_z[1, 1] = np.cos(angle_z)
    return rot_z @ rot_x


def _compile_program(gl: _GL, fragment_source: str) -> int:
    def shader(kind: int, source: str) -> int:
        handle = gl.glCreateShader(kind)
        buf = ctypes.c_char_p(source.encode("utf-8"))
        gl.glShaderSource(handle, 1, ctypes.byref(buf), None)
        gl.glCompileShader(handle)
        status = ctypes.c_int(GL_FALSE)
        gl.glGetShaderiv(handle, 0x8B81, ctypes.byref(status))
        if status.value == GL_FALSE:
            log = ctypes.create_string_buffer(4096)
            length = ctypes.c_int()
            gl.glGetShaderInfoLog(handle, 4096, ctypes.byref(length), log)
            raise PreviewUnavailable(f"shader compile failed: {log.value.decode()}")
        return handle

    vs = shader(GL_VERTEX_SHADER, _VERTEX_SHADER)
    fs = shader(GL_FRAGMENT_SHADER, fragment_source)
    program = gl.glCreateProgram()
    gl.glAttachShader(program, vs)
    gl.glAttachShader(program, fs)
    gl.glLinkProgram(program)
    status = ctypes.c_int(GL_FALSE)
    gl.glGetProgramiv(program, 0x8B82, ctypes.byref(status))
    if status.value == GL_FALSE:
        log = ctypes.create_string_buffer(4096)
        length = ctypes.c_int()
        gl.glGetProgramInfoLog(program, 4096, ctypes.byref(length), log)
        raise PreviewUnavailable(f"shader link failed: {log.value.decode()}")
    return program


def _upload_mesh(gl: _GL, vertices: np.ndarray, indices: np.ndarray) -> int:
    vao = ctypes.c_uint()
    gl.glGenVertexArrays(1, ctypes.byref(vao))
    gl.glBindVertexArray(vao)
    buffers = (ctypes.c_uint * 2)()
    gl.glGenBuffers(2, buffers)
    vertex_bytes = np.ascontiguousarray(vertices).tobytes()
    index_bytes = np.ascontiguousarray(indices).tobytes()
    gl.glBindBuffer(GL_ARRAY_BUFFER, buffers[0])
    gl.glBufferData(
        GL_ARRAY_BUFFER, len(vertex_bytes),
        ctypes.cast(ctypes.c_char_p(vertex_bytes), ctypes.c_void_p),
        GL_STATIC_DRAW,
    )
    stride = VERTEX_DTYPE.itemsize
    offsets = {
        0: VERTEX_DTYPE.fields["position"][1],
        1: VERTEX_DTYPE.fields["normal"][1],
        2: VERTEX_DTYPE.fields["uv"][1],
    }
    for location, offset in offsets.items():
        components = 3 if location < 2 else 2
        gl.glEnableVertexAttribArray(location)
        gl.glVertexAttribPointer(
            location, components, GL_FLOAT, GL_FALSE, stride,
            ctypes.c_void_p(offset),
        )
    gl.glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[1])
    gl.glBufferData(
        GL_ELEMENT_ARRAY_BUFFER, len(index_bytes),
        ctypes.cast(ctypes.c_char_p(index_bytes), ctypes.c_void_p),
        GL_STATIC_DRAW,
    )
    gl.glBindVertexArray(0)
    return vao.value


def _read_rgba(gl: _GL, width: int, height: int) -> np.ndarray:
    buffer = (ctypes.c_ubyte * (width * height * 4))()
    gl.glReadPixels(0, 0, width, height, 0x1908, 0x1401, buffer)
    image = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
    return np.flipud(image[:, :, :3]).copy()


def render_previews(
    binary_path: str,
    vertex_count: int,
    index_count: int,
    diffuse_png: str,
    normal_png: str,
    width: int = 512,
    height: int = 512,
) -> dict:
    """Render diffuse and normal previews of one LOD into two PNGs."""
    vertices = np.fromfile(binary_path, dtype=VERTEX_DTYPE, count=vertex_count)
    raw_indices = np.fromfile(
        binary_path,
        dtype=INDEX_DTYPE,
        count=index_count,
        offset=vertex_count * VERTEX_DTYPE.itemsize,
    )
    positions = vertices["position"]
    center = (positions.max(axis=0) + positions.min(axis=0)) * 0.5
    half_extent = float(np.max(positions.max(axis=0) - positions.min(axis=0))) * 0.62

    context = _try_osmesa(width, height) or _try_egl(width, height)
    if context is None:
        raise PreviewUnavailable(
            "no offscreen software GL backend available: " + " | ".join(_DIAGNOSTICS)
        )
    gl = context.gl
    info = {"backend": type(context).__name__, "diagnostics": diagnostics()}
    try:
        context.make_current()
        gl.glViewport(0, 0, width, height)
        gl.glEnable(GL_DEPTH_TEST)
        gl.glDisable(GL_CULL_FACE)
        vao = _upload_mesh(gl, vertices, raw_indices)
        model = _rotation_matrix()
        model[0:3, 3] = -model[0:3, 0:3] @ center.astype(np.float32)
        projection = _orthographic_matrix(half_extent, 0.1, 100000.0)
        mvp = projection @ model
        # NumPy stores row-major; GLSL mat4 is column-major, so transpose.
        mvp_gl = np.ascontiguousarray(mvp.T)

        for output_path, fragment in (
            (diffuse_png, _DIFFUSE_FRAGMENT),
            (normal_png, _NORMAL_FRAGMENT),
        ):
            program = _compile_program(gl, fragment)
            gl.glUseProgram(program)
            mvp_loc = gl.glGetUniformLocation(program, b"uMVP")
            gl.glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp_gl.tobytes())
            if output_path == diffuse_png:
                light_loc = gl.glGetUniformLocation(program, b"uLightDir")
                gl.glUniform3f(light_loc, 0.4, -0.5, 0.75)
            gl.glClearColor(0.08, 0.09, 0.11, 1.0)
            gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            gl.glBindVertexArray(vao)
            gl.glDrawElements(
                GL_TRIANGLES, index_count, GL_UNSIGNED_INT, ctypes.c_void_p(0)
            )
            gl.glFinish()
            Image.fromarray(_read_rgba(gl, width, height), mode="RGB").save(
                output_path, format="PNG"
            )
        info["diffuse_png"] = os.path.basename(diffuse_png)
        info["normal_png"] = os.path.basename(normal_png)
        info["rendered"] = True
    finally:
        context.shutdown()
    return info
