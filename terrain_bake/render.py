"""OpenGL offscreen preview with a deterministic CPU rasterizer fallback."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .coordinates import MATERIAL_COLORS
from .errors import E_RENDER_FAILED, BakeError
from .mesh import TerrainMesh, tangent_basis


@dataclass(frozen=True)
class PreviewResult:
    image: np.ndarray
    backend: str
    attempted_backends: list[str]
    fallback_reason: str | None
    width: int
    height: int


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-15:
        return vector.copy()
    return vector / length


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = _normalize(target - eye)
    side = _normalize(np.cross(forward, up))
    camera_up = np.cross(side, forward)
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, :3] = side
    matrix[1, :3] = camera_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -np.dot(side, eye)
    matrix[1, 3] = -np.dot(camera_up, eye)
    matrix[2, 3] = np.dot(forward, eye)
    return matrix


def _perspective(fov_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    factor = 1.0 / np.tan(np.deg2rad(fov_degrees) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = factor / aspect
    matrix[1, 1] = factor
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def camera_matrices(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    eye = np.array([1.52, 1.24, 1.67], dtype=np.float64)
    target = np.array([0.48, 0.13, 0.46], dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    view = _look_at(eye, target, up)
    projection = _perspective(42.0, width / max(height, 1), 0.05, 8.0)
    return view, projection


def _background(width: int, height: int) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float64)[:, None, None]
    top = np.array([0.36, 0.57, 0.76], dtype=np.float64)
    bottom = np.array([0.78, 0.79, 0.68], dtype=np.float64)
    color = top * (1.0 - y) + bottom * y
    return np.broadcast_to(color, (height, width, 3)).copy()


def _sample_texture_bilinear(
    texture: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    rows, cols = texture.shape[:2]
    px = np.clip(u, 0.0, 1.0) * (cols - 1)
    py = np.clip(v, 0.0, 1.0) * (rows - 1)
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    x0 = np.minimum(x0, cols - 1)
    y0 = np.minimum(y0, rows - 1)
    x1 = np.minimum(x0 + 1, cols - 1)
    y1 = np.minimum(y0 + 1, rows - 1)
    tx = px - x0
    ty = py - y0
    c00 = texture[y0, x0].astype(np.float64) / 255.0
    c10 = texture[y0, x1].astype(np.float64) / 255.0
    c01 = texture[y1, x0].astype(np.float64) / 255.0
    c11 = texture[y1, x1].astype(np.float64) / 255.0
    if c00.ndim == tx.ndim + 1:
        tx = tx[..., None]
        ty = ty[..., None]
    c0 = c00 * (1.0 - tx) + c10 * tx
    c1 = c01 * (1.0 - tx) + c11 * tx
    return c0 * (1.0 - ty) + c1 * ty


def _shade_preview(
    color: np.ndarray,
    normal: np.ndarray,
    tangent: np.ndarray,
    uv: np.ndarray,
    normal_map: np.ndarray,
) -> np.ndarray:
    normal = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-15)
    tangent = tangent - normal * np.sum(normal * tangent, axis=-1, keepdims=True)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=-1, keepdims=True), 1e-15)
    bitangent = np.cross(normal, tangent)
    tangent_encoded = _sample_texture_bilinear(normal_map, uv[..., 0], uv[..., 1])
    tangent_normal = tangent_encoded * 2.0 - 1.0
    tangent_normal /= np.maximum(
        np.linalg.norm(tangent_normal, axis=-1, keepdims=True), 1e-15
    )
    world_normal = (
        tangent * tangent_normal[..., 0:1]
        + bitangent * tangent_normal[..., 1:2]
        + normal * tangent_normal[..., 2:3]
    )
    world_normal /= np.maximum(
        np.linalg.norm(world_normal, axis=-1, keepdims=True), 1e-15
    )
    light = _normalize(np.array([0.46, 0.82, 0.34], dtype=np.float64))
    half_vector = _normalize(light + np.array([0.0, 0.6, 1.0]))
    diffuse = np.clip(np.sum(world_normal * light, axis=-1), 0.0, 1.0)
    specular = np.clip(np.sum(world_normal * half_vector, axis=-1), 0.0, 1.0) ** 28
    sky = 0.20 + 0.12 * np.clip(world_normal[..., 1], 0.0, 1.0)
    shaded = color * (0.26 + 0.78 * diffuse[..., None] + sky[..., None] * 0.18)
    shaded += 0.07 * specular[..., None]
    return np.clip(shaded, 0.0, 1.0)


def render_preview_cpu(
    mesh: TerrainMesh,
    normal_map: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    view, projection = camera_matrices(width, height)
    mvp = projection @ view
    vertices = mesh.vertices
    homogeneous = np.column_stack((vertices, np.ones(len(vertices), dtype=np.float64)))
    clip = homogeneous @ mvp.T
    w_clip = clip[:, 3]
    ndc = clip[:, :3] / np.maximum(w_clip[:, None], 1e-15)
    screen_x = (ndc[:, 0] * 0.5 + 0.5) * width
    screen_y = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * height

    tangent, _, normals = tangent_basis(mesh)
    colors = np.clip(mesh.weights @ MATERIAL_COLORS, 0.0, 1.0)
    uv = vertices[:, [0, 2]]
    image = _background(width, height)
    depth_buffer = np.zeros((height, width), dtype=np.float64)

    for face in mesh.faces:
        i0, i1, i2 = int(face[0]), int(face[1]), int(face[2])
        if min(w_clip[i0], w_clip[i1], w_clip[i2]) <= 0.03:
            continue
        x0, y0 = screen_x[i0], screen_y[i0]
        x1, y1 = screen_x[i1], screen_y[i1]
        x2, y2 = screen_x[i2], screen_y[i2]
        denominator = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(denominator) <= 1e-12:
            continue
        xmin = max(int(np.floor(min(x0, x1, x2))), 0)
        xmax = min(int(np.ceil(max(x0, x1, x2))) + 1, width)
        ymin = max(int(np.floor(min(y0, y1, y2))), 0)
        ymax = min(int(np.ceil(max(y0, y1, y2))) + 1, height)
        if xmin >= xmax or ymin >= ymax:
            continue

        xs = np.arange(xmin, xmax, dtype=np.float64) + 0.5
        ys = np.arange(ymin, ymax, dtype=np.float64) + 0.5
        xx, yy = np.meshgrid(xs, ys)
        b0 = ((x1 - xx) * (y2 - yy) - (x2 - xx) * (y1 - yy)) / denominator
        b1 = ((x2 - xx) * (y0 - yy) - (x0 - xx) * (y2 - yy)) / denominator
        b2 = 1.0 - b0 - b1
        inside = (b0 >= -1e-9) & (b1 >= -1e-9) & (b2 >= -1e-9)
        if not np.any(inside):
            continue

        inverse_w = (
            b0 / w_clip[i0] + b1 / w_clip[i1] + b2 / w_clip[i2]
        )
        sub_depth = depth_buffer[ymin:ymax, xmin:xmax]
        passes = inside & (inverse_w > sub_depth)
        if not np.any(passes):
            continue

        def interpolate(attribute: np.ndarray) -> np.ndarray:
            return (
                b0[..., None] * attribute[i0] / w_clip[i0]
                + b1[..., None] * attribute[i1] / w_clip[i1]
                + b2[..., None] * attribute[i2] / w_clip[i2]
            ) / inverse_w[..., None]

        normal = interpolate(normals)
        tangent_value = interpolate(tangent)
        uv_value = interpolate(uv)
        color_value = interpolate(colors)
        shaded = _shade_preview(
            color_value,
            normal,
            tangent_value,
            uv_value,
            normal_map,
        )
        sub_depth[passes] = inverse_w[passes]
        sub_image = image[ymin:ymax, xmin:xmax]
        sub_image[passes] = shaded[passes]

    return np.ascontiguousarray(np.rint(image * 255.0).astype(np.uint8))


def _compile_shader(gl, source: str, shader_type: int) -> int:
    shader = gl.glCreateShader(shader_type)
    gl.glShaderSource(shader, source)
    gl.glCompileShader(shader)
    if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
        log = gl.glGetShaderInfoLog(shader)
        raise RuntimeError(f"OpenGL shader compilation failed: {log}")
    return shader


def render_preview_opengl(
    mesh: TerrainMesh,
    normal_map: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    try:
        import glfw
        from OpenGL import GL as gl
    except Exception as exc:  # pragma: no cover - depends on local installation
        raise RuntimeError(f"OpenGL Python bindings unavailable: {exc}") from exc

    initialized = False
    window = None
    try:
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed (no display/context)")
        initialized = True
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(width, height, "terrain-bake", None, None)
        if window is None:
            raise RuntimeError("failed to create OpenGL 3.3 offscreen window")
        glfw.make_context_current(window)
        version = (gl.glGetString(gl.GL_VERSION) or b"").decode("utf-8", "replace")
        major = int(gl.glGetIntegerv(gl.GL_MAJOR_VERSION))
        minor = int(gl.glGetIntegerv(gl.GL_MINOR_VERSION))
        if (major, minor) < (3, 3):
            raise RuntimeError(f"OpenGL 3.3 required, got {version}")

        framebuffer = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
        color_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, color_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA8,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            None,
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER,
            gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D,
            color_texture,
            0,
        )
        depth_buffer = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_buffer)
        gl.glRenderbufferStorage(
            gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT24, width, height
        )
        gl.glFramebufferRenderbuffer(
            gl.GL_FRAMEBUFFER,
            gl.GL_DEPTH_ATTACHMENT,
            gl.GL_RENDERBUFFER,
            depth_buffer,
        )
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("OpenGL framebuffer is incomplete")
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)

        normal_texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, normal_texture)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGB8,
            normal_map.shape[1],
            normal_map.shape[0],
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            normal_map.tobytes(),
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        vertex_source = """
        #version 330 core
        layout(location=0) in vec3 aPos;
        layout(location=1) in vec3 aNormal;
        layout(location=2) in vec3 aTangent;
        layout(location=3) in vec2 aUV;
        layout(location=4) in vec3 aColor;
        uniform mat4 uMVP;
        out vec3 vNormal;
        out vec3 vTangent;
        out vec2 vUV;
        out vec3 vColor;
        void main() {
            vNormal = aNormal;
            vTangent = aTangent;
            vUV = aUV;
            vColor = aColor;
            gl_Position = uMVP * vec4(aPos, 1.0);
        }
        """
        fragment_source = """
        #version 330 core
        in vec3 vNormal;
        in vec3 vTangent;
        in vec2 vUV;
        in vec3 vColor;
        uniform sampler2D uNormalMap;
        out vec4 fragColor;
        void main() {
            vec3 N = normalize(vNormal);
            vec3 T = normalize(vTangent - N * dot(N, vTangent));
            vec3 B = cross(N, T);
            vec3 nt = normalize(texture(uNormalMap, vUV).xyz * 2.0 - 1.0);
            vec3 n = normalize(T * nt.x + B * nt.y + N * nt.z);
            vec3 L = normalize(vec3(0.46, 0.82, 0.34));
            vec3 H = normalize(L + vec3(0.0, 0.6, 1.0));
            float diffuse = max(dot(n, L), 0.0);
            float specular = pow(max(dot(n, H), 0.0), 28.0);
            vec3 color = vColor * (0.26 + 0.78 * diffuse + 0.18 * max(n.y, 0.0));
            color += vec3(0.07) * specular;
            fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
        }
        """
        vertex_shader = _compile_shader(gl, vertex_source, gl.GL_VERTEX_SHADER)
        fragment_shader = _compile_shader(gl, fragment_source, gl.GL_FRAGMENT_SHADER)
        program = gl.glCreateProgram()
        gl.glAttachShader(program, vertex_shader)
        gl.glAttachShader(program, fragment_shader)
        gl.glLinkProgram(program)
        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            raise RuntimeError(f"OpenGL program link failed: {gl.glGetProgramInfoLog(program)}")

        tangent, _, normals = tangent_basis(mesh)
        colors = np.clip(mesh.weights @ MATERIAL_COLORS, 0.0, 1.0)
        uv = mesh.vertices[:, [0, 2]]
        interleaved = np.hstack((mesh.vertices, normals, tangent, uv, colors)).astype(
            np.float32
        )
        indices = mesh.faces.reshape(-1).astype(np.uint32)

        vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(vao)
        vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            interleaved.nbytes,
            interleaved.tobytes(),
            gl.GL_STATIC_DRAW,
        )
        ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            indices.nbytes,
            indices.tobytes(),
            gl.GL_STATIC_DRAW,
        )
        stride = 14 * 4
        for location, size, offset in (
            (0, 3, 0),
            (1, 3, 3 * 4),
            (2, 3, 6 * 4),
            (3, 2, 9 * 4),
            (4, 3, 11 * 4),
        ):
            gl.glEnableVertexAttribArray(location)
            gl.glVertexAttribPointer(
                location,
                size,
                gl.GL_FLOAT,
                gl.GL_FALSE,
                stride,
                ctypes.c_void_p(offset),
            )

        gl.glViewport(0, 0, width, height)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.45, 0.62, 0.76, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glUseProgram(program)
        view, projection = camera_matrices(width, height)
        mvp = projection @ view
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(program, "uMVP"),
            1,
            gl.GL_FALSE,
            np.ascontiguousarray(mvp.T, dtype=np.float32),
        )
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, normal_texture)
        gl.glUniform1i(gl.glGetUniformLocation(program, "uNormalMap"), 0)
        gl.glDrawElements(
            gl.GL_TRIANGLES,
            int(indices.size),
            gl.GL_UNSIGNED_INT,
            None,
        )
        gl.glFinish()
        gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
        pixels = gl.glReadPixels(
            0, 0, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE
        )
        array = np.frombuffer(bytes(pixels), dtype=np.uint8).reshape(height, width, 4)
        return np.ascontiguousarray(array[::-1, :, :3])
    finally:
        if initialized:
            glfw.terminate()


def render_preview(
    mesh: TerrainMesh,
    normal_map: np.ndarray,
    width: int,
    height: int,
    backend: str = "auto",
) -> PreviewResult:
    attempted: list[str] = []
    fallback_reason: str | None = None
    if backend in {"auto", "opengl"}:
        attempted.append("opengl")
        try:
            image = render_preview_opengl(mesh, normal_map, width, height)
            return PreviewResult(image, "opengl", attempted, None, width, height)
        except Exception as exc:
            fallback_reason = str(exc) or exc.__class__.__name__
            if backend == "opengl":
                raise BakeError(
                    E_RENDER_FAILED,
                    "OpenGL preview requested but no usable 3.3 context was available",
                    {"reason": fallback_reason},
                ) from exc
    attempted.append("cpu")
    try:
        image = render_preview_cpu(mesh, normal_map, width, height)
    except Exception as exc:
        raise BakeError(
            E_RENDER_FAILED,
            "CPU preview rasterization failed",
            {"reason": str(exc)},
        ) from exc
    return PreviewResult(
        image=image,
        backend="cpu",
        attempted_backends=attempted,
        fallback_reason=fallback_reason,
        width=width,
        height=height,
    )


def save_preview_png(
    result: PreviewResult,
    path: Path,
    seed: int,
) -> None:
    from .normals import _png_metadata

    metadata = _png_metadata(
        seed,
        renderer=result.backend,
        resolution=f"{result.width}x{result.height}",
    )
    Image.fromarray(result.image, mode="RGB").save(path, format="PNG", pnginfo=metadata)
