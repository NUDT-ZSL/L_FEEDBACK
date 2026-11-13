"""ModernGL hardware rasteriser sharing mesh/LOD/stitch logic with software.

Only rasterisation and framebuffer output differ from
:mod:`terrain.software_backend`.  Every GL resource is created inside a
``try`` block and released in ``finally``; context creation failure is mapped
to the ``E_NO_GL`` business code so callers can fall back to the software
backend without leaking resources.
"""

from __future__ import annotations

import numpy as np

from .camera import look_at, perspective
from .config import PipelineConfig
from .errors import E_NO_GL, PipelineError
from .mesh import TileMesh

BACKEND_NAME = "moderngl"

VERTEX_SHADER = """
#version 330
uniform mat4 view_projection;
in vec3 in_position;
in float in_color;
out float v_color;
void main() {
    gl_Position = view_projection * vec4(in_position, 1.0);
    v_color = in_color;
}
"""

FRAGMENT_SHADER = """
#version 330
in float v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, v_color, v_color, 1.0);
}
"""


def render_scene(
    meshes: list[TileMesh],
    vertex_colors: list[np.ndarray],
    config: PipelineConfig,
) -> np.ndarray:
    try:
        import moderngl
    except ImportError as exc:
        raise PipelineError(E_NO_GL, f"moderngl unavailable: {exc}") from exc

    context = None
    program = None
    vertex_array = None
    vertex_buffer = None
    index_buffer = None
    framebuffer = None
    try:
        try:
            context = moderngl.create_standalone_context()
        except Exception as exc:
            raise PipelineError(
                E_NO_GL, f"GL 3.3 context creation failed: {exc}"
            ) from exc
        if int(context.version_code) < 330:
            raise PipelineError(
                E_NO_GL,
                f"GL {context.version_code} below required 3.3",
            )

        program = context.program(
            vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER
        )
        framebuffer = context.framebuffer(
            color_attachments=[
                context.texture((config.image_width, config.image_height), 3)
            ],
            depth_attachment=context.depth_renderbuffer(
                (config.image_width, config.image_height)
            ),
        )

        ordered = sorted(
            meshes, key=lambda mesh: (mesh.key.level, mesh.key.iz, mesh.key.ix)
        )
        # GLSL mat4 uniforms expect column-major storage; transpose the
        # row-major NumPy view-projection matrix before uploading.
        view_projection = (
            (perspective(config.camera) @ look_at(config.camera)).T
        ).astype("f4")

        framebuffer.use()
        context.enable(context.DEPTH_TEST | context.CULL_FACE)
        context.front_face = "ccw"
        context.cull_face = "back"
        context.clear(0.0, 0.0, 0.0, 1.0, depth=1.0)

        color_lookup = {
            mesh.key: values for mesh, values in zip(meshes, vertex_colors)
        }
        for mesh in ordered:
            positions = np.ascontiguousarray(mesh.vertices, dtype="f4")
            colors = np.ascontiguousarray(
                color_lookup[mesh.key], dtype="f4"
            ).reshape(-1, 1)
            packed = np.concatenate([positions, colors], axis=1)
            indices = np.ascontiguousarray(mesh.triangles, dtype="i4")
            vertex_buffer = context.buffer(packed.tobytes())
            index_buffer = context.buffer(indices.tobytes())
            content = [(vertex_buffer, "3f 1f", "in_position", "in_color")]
            vertex_array = context.vertex_array(
                program, content, index_buffer
            )
            program["view_projection"].write(view_projection.tobytes())
            vertex_array.render()
            vertex_array.release()
            vertex_buffer.release()
            index_buffer.release()
            vertex_array = None
            vertex_buffer = None
            index_buffer = None

        raw = framebuffer.read(components=3, alignment=1)
        image = np.frombuffer(raw, dtype=np.uint8).reshape(
            config.image_height, config.image_width, 3
        )
        image = image[::-1, :, :]
        luminance = image[:, :, 0]
        return np.ascontiguousarray(luminance)
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            E_NO_GL, f"GL rendering failed: {exc}"
        ) from exc
    finally:
        for resource in (
            vertex_array,
            vertex_buffer,
            index_buffer,
            program,
            framebuffer,
        ):
            if resource is not None:
                resource.release()
        if context is not None:
            context.release()
