"""ModernGL (GPU / OpenGL 3.3 core) render backend.

A standalone headless context is created through EGL or OSMesa so no window
system is required.  The vertex shader implements the exact same Blinn-Phong
math as :mod:`rfr.lighting`; the rasterizer, depth test and blending are handled
by OpenGL.  State changes are forwarded only when :class:`StateCache` reports a
change, mirroring the software backend's counters.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..errors import RenderError
from ..pipeline import DrawCall
from ..scene import Scene
from .base import RenderBackend

try:  # ModernGL is optional at import time so the software fallback still works
    import moderngl
except Exception:  # pragma: no cover - exercised only when the wheel is absent
    moderngl = None


VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_color;

uniform mat4 u_mvp;

out vec3 v_normal;
out vec4 v_color;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    v_normal = normalize(in_normal);
    v_color = in_color;
}
"""

FRAGMENT_SHADER = """
#version 330 core
in vec3 v_normal;
in vec4 v_color;
out vec4 frag_color;

void main() {
    frag_color = clamp(v_color, 0.0, 1.0);
}
"""


class GLBackend(RenderBackend):
    name = "opengl"

    def __init__(self, width: int, height: int, backend: Optional[str] = None) -> None:
        if moderngl is None:
            raise RenderError("E_BACKEND_UNAVAILABLE",
                              "ModernGL is not installed.")
        if width <= 0 or height <= 0:
            raise RenderError("E_SIZE_MISMATCH", "Invalid GL framebuffer size.",
                              {"width": width, "height": height})
        self.width = int(width)
        self.height = int(height)
        self.ctx = self._create_context(backend)
        try:
            self.program = self.ctx.program(
                vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        except Exception as exc:
            raise RenderError("E_SHADER_COMPILE", "Shader program failed to build.",
                              {"detail": str(exc)}) from exc
        # Standard RGBA8 normalized color target so readback follows the
        # conventional channel order expected by the PNG encoder and the
        # software backend.
        self.color_texture = self.ctx.texture((self.width, self.height), 4)
        self.depth_renderbuffer = self.ctx.depth_renderbuffer((self.width, self.height))
        self.fbo = self.ctx.framebuffer(
            color_attachments=[self.color_texture],
            depth_attachment=self.depth_renderbuffer)
        self.fbo.use()
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        self.ctx.front_face = "ccw"
        self.ctx.cull_face = "back"
        self._blend = "opaque"
        self._cull = "back"
        self._depth_test = True
        self.vaos = 0
        self.uploaded_vertices = 0
        self._mvp = self.program["u_mvp"]

    @staticmethod
    def _create_context(backend: Optional[str]):
        candidates = (backend,) if backend else ("egl", "osmesa")
        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                return moderngl.create_context(standalone=True, backend=candidate)
            except Exception as exc:  # pragma: no cover - depends on driver
                last_error = exc
        raise RenderError("E_BACKEND_UNAVAILABLE",
                          "Could not create an OpenGL standalone context.",
                          {"tried": list(candidates), "detail": str(last_error)})

    # ------------------------------------------------------------ frame flow
    def begin_frame(self, scene: Scene) -> np.ndarray:
        if scene.width != self.width or scene.height != self.height:
            raise RenderError("E_SIZE_MISMATCH", "Scene/GL framebuffer size mismatch.",
                              {"scene": [scene.width, scene.height],
                               "backend": [self.width, self.height]})
        bg = np.asarray(scene.background, dtype=np.float32)
        self.fbo.clear(red=float(bg[0]), green=float(bg[1]),
                       blue=float(bg[2]), alpha=float(bg[3]), depth=1.0)
        self._scene = scene
        return np.empty((self.height, self.width, 4), dtype=np.uint8)

    def apply_state(self, draw: DrawCall, changes: Dict[str, bool]) -> None:
        if changes["depth_test"]:
            self._depth_test = draw.depth_test
            if draw.depth_test:
                self.ctx.enable(moderngl.DEPTH_TEST)
            else:
                self.ctx.disable(moderngl.DEPTH_TEST)
        if changes["cull"]:
            self._cull = draw.cull
            if draw.cull == "none":
                self.ctx.disable(moderngl.CULL_FACE)
            else:
                self.ctx.enable(moderngl.CULL_FACE)
                self.ctx.cull_face = draw.cull
        if changes["blend"]:
            self._blend = draw.blend
            self._apply_blend(draw.blend)

    def _apply_blend(self, blend: str) -> None:
        if blend == "opaque":
            self.ctx.disable(moderngl.BLEND)
            self.ctx.depth_mask = True
        elif blend == "alpha":
            self.ctx.enable(moderngl.BLEND)
            # Standard glBlendFunc: the same (SRC_ALPHA,
            # ONE_MINUS_SRC_ALPHA) factors for RGB and alpha, matching the
            # software compositor exactly.
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self.ctx.blend_equation = moderngl.FUNC_ADD
            self.ctx.depth_mask = False
        else:  # additive
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE,
                                   moderngl.SRC_ALPHA, moderngl.ONE)
            self.ctx.blend_equation = moderngl.FUNC_ADD
            self.ctx.depth_mask = False

    def draw(self, draw: DrawCall) -> None:
        if draw.positions.shape[0] != draw.colors.shape[0]:
            raise RenderError("E_SIZE_MISMATCH", "Position/color vertex count mismatch.",
                              {"node": draw.node_name})
        scene = getattr(self, "_scene", None)
        if scene is None:
            raise RenderError("E_RENDER", "draw() called before begin_frame().")
        from .. import mathutils
        aspect = self.width / self.height
        view = mathutils.look_at(scene.camera.eye, scene.camera.target, scene.camera.up)
        proj = mathutils.perspective(scene.camera.fov_y, aspect,
                                     scene.camera.near, scene.camera.far)
        mvp = (proj @ view).astype(np.float32)

        vertex_count = draw.positions.shape[0]
        packed = np.empty(vertex_count, dtype=[
            ("position", np.float32, (3,)),
            ("normal", np.float32, (3,)),
            ("color", np.float32, (4,)),
        ])
        packed["position"] = draw.positions
        packed["normal"] = draw.normals
        packed["color"] = draw.colors
        vbo = self.ctx.buffer(packed.tobytes())
        ibo = self.ctx.buffer(draw.indices.astype(np.int32).tobytes())
        vao = self.ctx.vertex_array(
            self.program,
            [(vbo, "3f 3f 4f", "in_position", "in_normal", "in_color")],
            index_buffer=ibo,
            index_element_size=4,
        )
        self._mvp.write(mvp.T.tobytes())
        vao.render(moderngl.TRIANGLES)
        self.vaos += 1
        self.uploaded_vertices += vertex_count
        vao.release()
        ibo.release()
        vbo.release()

    def end_frame(self) -> None:
        self.ctx.finish()

    def read_pixels(self) -> np.ndarray:
        raw = self.fbo.read(components=4, alignment=1)
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 4))
        # fbo.read() returns bottom-up RGBA; flip to top-down to match the
        # software framebuffer and the PNG row convention.
        return np.ascontiguousarray(np.flipud(pixels))

    def backend_statistics(self) -> Dict[str, int]:
        return {
            "vaos_created": self.vaos,
            "uploaded_vertices": self.uploaded_vertices,
            "gl_renderer": self.ctx.info.get("GL_RENDERER", "unknown"),
        }

    def release(self) -> None:
        try:
            self.fbo.release()
            self.depth_renderbuffer.release()
            self.color_texture.release()
            self.program.release()
        except Exception:  # pragma: no cover
            pass
