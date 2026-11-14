"""OpenGL 4.1 rendering pipeline."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import glm
import moderngl
import numpy as np

from .errors import (
    EXIT_FRAME_RENDER,
    EXIT_GL_CONTEXT,
    EXIT_GL_RESOURCE,
    FRAME_RENDER,
    GL_CONTEXT,
    GL_RESOURCE,
    BusinessError,
)
from .interpolation import CameraState, interpolate_camera
from .ordering import PanelOrder, order_panels
from .output import rgba16f_bytes_to_rgb8, rgb8_to_png_bytes
from .reporting import make_frame_stats
from .scene import Scene
from .util import sha256_bytes


R32UI_INTERNAL_FORMAT = 0x8236
SHADER_DIRECTORY = Path(__file__).with_name("shaders")


@dataclass(frozen=True)
class FrameArtifact:
    frame_index: int
    png_bytes: bytes
    png_sha256: str
    frame_stats: dict[str, Any]
    gl_info: dict[str, str]


class FrameResources:
    """Transient per-frame GL objects, always released by the caller."""

    def __init__(self, ctx: moderngl.Context, width: int, height: int, shadow_resolution: int):
        self.ctx = ctx
        self.camera_id = ctx.texture(
            (width, height), 1, dtype="u4", internal_format=R32UI_INTERNAL_FORMAT
        )
        self.camera_id.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.camera_depth = ctx.depth_texture((width, height))
        self.camera_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.shadow_id = ctx.texture(
            (shadow_resolution, shadow_resolution),
            1,
            dtype="u4",
            internal_format=R32UI_INTERNAL_FORMAT,
        )
        self.shadow_id.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.shadow_depth = ctx.depth_texture((shadow_resolution, shadow_resolution))
        self.shadow_depth.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.base_color = ctx.texture((width, height), 4, dtype="f2")
        self.base_color.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.volume_color = ctx.texture((width, height), 4, dtype="f2")
        self.volume_color.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.final_color = ctx.texture((width, height), 4, dtype="f2")
        self.final_color.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self.camera_prepass_fbo = ctx.framebuffer([self.camera_id], self.camera_depth)
        self.shadow_fbo = ctx.framebuffer([self.shadow_id], self.shadow_depth)
        # The camera depth attachment is deliberately shared: it is cleared
        # once by the camera prepass and then reused for depth testing.
        self.base_fbo = ctx.framebuffer([self.base_color], self.camera_depth)
        self.volume_fbo = ctx.framebuffer([self.volume_color])
        self.final_fbo = ctx.framebuffer([self.final_color], self.camera_depth)

        self._objects: list[Any] = [
            self.camera_prepass_fbo,
            self.shadow_fbo,
            self.base_fbo,
            self.volume_fbo,
            self.final_fbo,
            self.camera_id,
            self.camera_depth,
            self.shadow_id,
            self.shadow_depth,
            self.base_color,
            self.volume_color,
            self.final_color,
        ]

    def release(self) -> None:
        # Deleting a framebuffer that is still bound leaves a driver-dependent
        # binding residue on some software GL stacks.  Explicitly return to the
        # default framebuffer before deleting the transient attachments.
        try:
            self.ctx.detect_framebuffer(0).use()
        except Exception:
            pass
        for item in self._objects:
            try:
                item.release()
            except Exception:
                pass
        self._objects.clear()


def _context_settings(backend: str) -> dict[str, str]:
    backend = backend.lower()
    if backend == "auto":
        if sys.platform.startswith("linux"):
            return {"backend": "egl"}
        return {}
    if backend == "default":
        return {}
    if backend not in {"egl", "glx", "wgl"}:
        raise BusinessError(
            GL_CONTEXT,
            f"unsupported OpenGL backend {backend!r}",
            EXIT_GL_CONTEXT,
            {"backend": backend},
        )
    return {"backend": backend}


def create_gl_context(backend: str) -> moderngl.Context:
    settings = _context_settings(backend)
    try:
        ctx = moderngl.create_standalone_context(require=410, **settings)
    except Exception as exc:
        raise BusinessError(
            GL_CONTEXT,
            f"could not create an OpenGL 4.1 standalone context: {exc}",
            EXIT_GL_CONTEXT,
            {"backend": backend, "exception_type": type(exc).__name__},
        ) from exc
    if ctx.version_code < 410:
        try:
            ctx.release()
        except Exception:
            pass
        raise BusinessError(
            GL_CONTEXT,
            f"OpenGL 4.1 is required, got {ctx.version_code}",
            EXIT_GL_CONTEXT,
            {"backend": backend, "version_code": ctx.version_code},
        )
    ctx.gc_mode = "auto"
    return ctx


def _gl_info(ctx: moderngl.Context) -> dict[str, str]:
    info = ctx.info
    return {
        "GL_VENDOR": str(info.get("GL_VENDOR", "")),
        "GL_RENDERER": str(info.get("GL_RENDERER", "")),
        "GL_VERSION": str(info.get("GL_VERSION", "")),
    }


def _set_matrix(program: moderngl.Program, name: str, matrix: glm.mat4) -> None:
    program[name].write(matrix.to_bytes())


def _view_projection(camera: CameraState, width: int, height: int) -> tuple[glm.mat4, glm.mat4]:
    position = glm.vec3(*camera.position)
    # Camera state stores xyzw, PyGLM's constructor is wxyz.
    rotation = glm.mat4_cast(
        glm.quat(camera.orientation[3], camera.orientation[0], camera.orientation[1], camera.orientation[2])
    )
    world_to_view = glm.inverse(glm.translate(position) * rotation)
    projection = glm.perspective(
        glm.radians(camera.fov_y_deg),
        width / height,
        camera.near,
        camera.far,
    )
    return projection * world_to_view, world_to_view


def _light_view_projection(scene: Scene) -> glm.mat4:
    all_vertices = np.concatenate([building.vertices for building in scene.buildings], axis=0)
    minimum = all_vertices.min(axis=0)
    maximum = all_vertices.max(axis=0)
    center = (minimum + maximum) * 0.5
    radius = float(np.linalg.norm(maximum - minimum)) * 0.5
    direction = glm.normalize(glm.vec3(*scene.light.direction))
    up = glm.vec3(0.0, 1.0, 0.0)
    if abs(float(glm.dot(direction, up))) > 0.95:
        up = glm.vec3(0.0, 0.0, 1.0)
    eye = glm.vec3(*center) - direction * (radius * 2.0 + 1.0)
    view = glm.lookAt(eye, glm.vec3(*center), up)

    transformed: list[glm.vec4] = []
    for vertex in all_vertices:
        transformed.append(view * glm.vec4(float(vertex[0]), float(vertex[1]), float(vertex[2]), 1.0))
    xs = [float(point.x) for point in transformed]
    ys = [float(point.y) for point in transformed]
    distances = [-float(point.z) for point in transformed]
    margin = max(radius * 0.02, 0.5)
    left = min(xs) - margin
    right = max(xs) + margin
    bottom = min(ys) - margin
    top = max(ys) + margin
    near = max(0.01, min(distances) - margin)
    far = max(near + 0.1, max(distances) + margin)
    projection = glm.ortho(left, right, bottom, top, near, far)
    return projection * view


def _mesh_geometry(scene: Scene) -> tuple[np.ndarray, int]:
    vertices: list[tuple[float, float, float, float, float, float, float, float, float, float]] = []
    triangle_count = 0
    for building in scene.buildings:
        for face in building.faces:
            points = building.vertices[np.asarray(face, dtype=np.int64)]
            for index in range(1, len(points) - 1):
                triangle = np.asarray([points[0], points[index], points[index + 1]])
                normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                length = float(np.linalg.norm(normal))
                if length <= 1e-12:
                    continue
                normal /= length
                face_center = triangle.mean(axis=0)
                if float(np.dot(normal, face_center - np.asarray(building.centroid))) < 0.0:
                    normal = -normal
                for point in triangle:
                    vertices.append(
                        (
                            float(point[0]),
                            float(point[1]),
                            float(point[2]),
                            float(normal[0]),
                            float(normal[1]),
                            float(normal[2]),
                            float(building.color[0]),
                            float(building.color[1]),
                            float(building.color[2]),
                            float(building.id),
                        )
                    )
                triangle_count += 1
    return np.asarray(vertices, dtype=np.float32), triangle_count


def _glass_geometry(scene: Scene) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    vertices: list[tuple[float, float, float, float, float, float, float]] = []
    ranges: dict[int, tuple[int, int]] = {}
    for panel in scene.glass_panels:
        start = len(vertices)
        alpha = 1.0 - panel.transmittance
        for index in range(1, len(panel.vertices) - 1):
            for point in (panel.vertices[0], panel.vertices[index], panel.vertices[index + 1]):
                vertices.append(
                    (
                        float(point[0]),
                        float(point[1]),
                        float(point[2]),
                        float(panel.color[0]),
                        float(panel.color[1]),
                        float(panel.color[2]),
                        float(alpha),
                    )
                )
        ranges[panel.id] = (start, len(vertices) - start)
    return np.asarray(vertices, dtype=np.float32), ranges


class RenderPipeline:
    """A reusable context plus immutable scene GL resources."""

    def __init__(self, scene: Scene, backend: str = "auto"):
        self.scene = scene
        self._closed = False
        try:
            self.ctx = create_gl_context(backend)
            self.gl_info = _gl_info(self.ctx)
            self._create_programs()
            self._create_geometry()
            self.light_view_proj = _light_view_projection(scene)
        except BusinessError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise BusinessError(
                GL_RESOURCE,
                f"failed to initialize OpenGL resources: {exc}",
                EXIT_GL_RESOURCE,
                {"exception_type": type(exc).__name__},
            ) from exc

    def _shader(self, name: str) -> str:
        return (SHADER_DIRECTORY / name).read_text(encoding="utf-8")

    def _program(self, vertex: str, fragment: str) -> moderngl.Program:
        try:
            return self.ctx.program(
                vertex_shader=self._shader(vertex),
                fragment_shader=self._shader(fragment),
            )
        except Exception as exc:
            raise BusinessError(
                GL_RESOURCE,
                f"shader compilation/link failed for {vertex} + {fragment}: {exc}",
                EXIT_GL_RESOURCE,
                {"vertex": vertex, "fragment": fragment},
            ) from exc

    def _create_programs(self) -> None:
        self.prepass_program = self._program("prepass.vert", "prepass.frag")
        self.sky_program = self._program("fullscreen.vert", "sky.frag")
        self.building_program = self._program("building.vert", "building.frag")
        self.volume_program = self._program("fullscreen.vert", "volume.frag")
        self.composite_program = self._program("fullscreen.vert", "composite.frag")
        self.glass_program = self._program("glass.vert", "glass.frag")

    def _create_geometry(self) -> None:
        mesh_data, self.building_triangle_count = _mesh_geometry(self.scene)
        prepass_data = mesh_data[:, [0, 1, 2, 9]].copy()
        building_data = mesh_data[:, :9].copy()
        self.prepass_buffer = self.ctx.buffer(prepass_data.tobytes())
        self.building_buffer = self.ctx.buffer(building_data.tobytes())
        self.mesh_prepass_vao = self.ctx.vertex_array(
            self.prepass_program,
            [(self.prepass_buffer, "3f 1f", "in_position", "in_building_id")],
        )
        self.mesh_building_vao = self.ctx.vertex_array(
            self.building_program,
            [(self.building_buffer, "3f 3f 3f", "in_position", "in_normal", "in_color")],
        )

        glass_data, self.glass_ranges = _glass_geometry(self.scene)
        self.glass_buffer = self.ctx.buffer(glass_data.tobytes())
        self.glass_vao = self.ctx.vertex_array(
            self.glass_program,
            [(self.glass_buffer, "3f 4f", "in_position", "in_color")],
        )
        self.sky_vao = self.ctx.vertex_array(self.sky_program, [])
        self.volume_vao = self.ctx.vertex_array(self.volume_program, [])
        self.composite_vao = self.ctx.vertex_array(self.composite_program, [])

    def _check_gl(self) -> None:
        error = self.ctx.error
        if error != "GL_NO_ERROR":
            raise RuntimeError(f"OpenGL error: {error}")

    def _draw_shadow_pass(self, resources: FrameResources) -> None:
        ctx = self.ctx
        resources.shadow_fbo.use()
        ctx.viewport = (0, 0, self.scene.shadow.resolution, self.scene.shadow.resolution)
        ctx.depth_mask = True
        resources.shadow_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<"
        ctx.depth_mask = True
        ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.CULL_FACE)
        _set_matrix(self.prepass_program, "u_view_proj", self.light_view_proj)
        self.mesh_prepass_vao.render(moderngl.TRIANGLES)

    def _draw_camera_prepass(self, resources: FrameResources, view_proj: glm.mat4) -> None:
        ctx = self.ctx
        resources.camera_prepass_fbo.use()
        ctx.viewport = (0, 0, self.scene.width, self.scene.height)
        ctx.depth_mask = True
        resources.camera_prepass_fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<"
        ctx.depth_mask = True
        ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.CULL_FACE)
        _set_matrix(self.prepass_program, "u_view_proj", view_proj)
        self.mesh_prepass_vao.render(moderngl.TRIANGLES)

    def _draw_base(
        self,
        resources: FrameResources,
        view_proj: glm.mat4,
        inverse_view_proj: glm.mat4,
    ) -> None:
        ctx = self.ctx
        resources.base_fbo.use()
        ctx.viewport = (0, 0, self.scene.width, self.scene.height)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.depth_mask = False
        ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.CULL_FACE)

        sky = self.sky_program
        _set_matrix(sky, "u_inv_view_proj", inverse_view_proj)
        sky["u_light_direction"].value = self.scene.light.direction
        sky["u_light_color"].value = self.scene.light.color
        self.sky_vao.render(moderngl.TRIANGLES, vertices=3)

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<="
        ctx.depth_mask = False
        building = self.building_program
        _set_matrix(building, "u_view_proj", view_proj)
        building["u_light_direction"].value = self.scene.light.direction
        building["u_light_color"].value = self.scene.light.color
        building["u_light_intensity"].value = self.scene.light.intensity
        self.mesh_building_vao.render(moderngl.TRIANGLES)

    def _draw_volume(
        self,
        resources: FrameResources,
        camera: CameraState,
        inverse_view_proj: glm.mat4,
    ) -> None:
        ctx = self.ctx
        resources.volume_fbo.use()
        ctx.viewport = (0, 0, self.scene.width, self.scene.height)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.depth_mask = False
        ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.CULL_FACE)

        resources.camera_depth.use(0)
        resources.shadow_depth.use(1)
        resources.shadow_id.use(2)
        program = self.volume_program
        program["u_camera_depth"].value = 0
        program["u_shadow_depth"].value = 1
        program["u_shadow_id"].value = 2
        _set_matrix(program, "u_inv_view_proj", inverse_view_proj)
        _set_matrix(program, "u_light_view_proj", self.light_view_proj)
        program["u_camera_position"].value = camera.position
        program["u_light_direction"].value = self.scene.light.direction
        program["u_light_color"].value = self.scene.light.color
        program["u_fog_color"].value = self.scene.fog.color
        program["u_light_intensity"].value = self.scene.light.intensity
        program["u_fog_density"].value = self.scene.fog.density
        program["u_fog_extinction"].value = self.scene.fog.extinction
        program["u_fog_scattering"].value = self.scene.fog.scattering
        program["u_fog_anisotropy"].value = self.scene.fog.anisotropy
        program["u_fog_ambient"].value = self.scene.fog.ambient
        program["u_near"].value = camera.near
        program["u_far"].value = camera.far
        program["u_shadow_bias"].value = self.scene.shadow.bias
        program["u_steps"].value = self.scene.fog.steps
        program["u_shadow_resolution"].value = self.scene.shadow.resolution
        program["u_resolution"].value = (float(self.scene.width), float(self.scene.height))
        self.volume_vao.render(moderngl.TRIANGLES, vertices=3)

    def _draw_composite(self, resources: FrameResources) -> None:
        ctx = self.ctx
        resources.final_fbo.use()
        ctx.viewport = (0, 0, self.scene.width, self.scene.height)
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.depth_mask = False
        ctx.disable(moderngl.BLEND)
        ctx.disable(moderngl.CULL_FACE)
        resources.base_color.use(0)
        resources.volume_color.use(1)
        program = self.composite_program
        program["u_base_color"].value = 0
        program["u_volume_color"].value = 1
        self.composite_vao.render(moderngl.TRIANGLES, vertices=3)

    def _draw_glass(
        self,
        resources: FrameResources,
        view_proj: glm.mat4,
        orders: tuple[PanelOrder, ...],
    ) -> None:
        ctx = self.ctx
        resources.final_fbo.use()
        ctx.viewport = (0, 0, self.scene.width, self.scene.height)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = "<="
        ctx.depth_mask = False
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        ctx.disable(moderngl.CULL_FACE)

        program = self.glass_program
        _set_matrix(program, "u_view_proj", view_proj)
        for order in orders:
            if not order.rendered:
                continue
            first, count = self.glass_ranges[order.panel_id]
            self.glass_vao.render(moderngl.TRIANGLES, vertices=count, first=first)

    def render_frame(self, frame_index: int) -> FrameArtifact:
        camera = interpolate_camera(self.scene, frame_index)
        orders = order_panels(self.scene, camera)
        view_proj, _view = _view_projection(camera, self.scene.width, self.scene.height)
        inverse_view_proj = glm.inverse(view_proj)

        self.ctx.clear_errors()
        resources = FrameResources(
            self.ctx,
            self.scene.width,
            self.scene.height,
            self.scene.shadow.resolution,
        )
        try:
            self._check_gl()
            self._draw_shadow_pass(resources)
            self._draw_camera_prepass(resources, view_proj)
            self._draw_base(resources, view_proj, inverse_view_proj)
            self._draw_volume(resources, camera, inverse_view_proj)
            self._draw_composite(resources)
            self._draw_glass(resources, view_proj, orders)
            self.ctx.finish()
            self._check_gl()
            readback = resources.final_color.read(alignment=1)
            self._check_gl()
            rgb = rgba16f_bytes_to_rgb8(
                readback, self.scene.width, self.scene.height
            )
            png_bytes = rgb8_to_png_bytes(rgb)
        finally:
            resources.release()
            self.ctx.gc()

        png_sha = sha256_bytes(png_bytes)
        stats = make_frame_stats(
            self.scene,
            camera,
            orders,
            png_sha,
            building_triangle_count=self.building_triangle_count,
        )
        return FrameArtifact(
            frame_index=frame_index,
            png_bytes=png_bytes,
            png_sha256=png_sha,
            frame_stats=stats,
            gl_info=self.gl_info,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name in (
            "mesh_prepass_vao",
            "mesh_building_vao",
            "glass_vao",
            "sky_vao",
            "volume_vao",
            "composite_vao",
            "prepass_buffer",
            "building_buffer",
            "glass_buffer",
            "prepass_program",
            "sky_program",
            "building_program",
            "volume_program",
            "composite_program",
            "glass_program",
        ):
            item = getattr(self, name, None)
            if item is None:
                continue
            try:
                item.release()
            except Exception:
                pass
        context = getattr(self, "ctx", None)
        if context is not None:
            try:
                context.release()
            except Exception:
                pass


def frame_failure(frame_index: int, exception: BaseException) -> BusinessError:
    if isinstance(exception, BusinessError):
        detail = exception.message
        cause_code = exception.code
    else:
        detail = f"{type(exception).__name__}: {exception}"
        cause_code = type(exception).__name__
    return BusinessError(
        FRAME_RENDER,
        f"frame {frame_index} failed: {detail}",
        EXIT_FRAME_RENDER,
        {
            "frame_index": frame_index,
            "cause_code": cause_code,
            "cause_type": type(exception).__name__,
        },
    )

