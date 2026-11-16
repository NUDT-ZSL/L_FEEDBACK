"""OpenGL 4.5 deferred rendering pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import glm
import numpy as np
from OpenGL.GL import *

from .errors import RenderError
from .gbuffer import GBUFFER_ATTACHMENTS
from .gl_context import GLInfo
from .gl_utils import (
    FullscreenTriangle,
    MeshGPU,
    check_gl,
    compile_program,
    create_color_fbo,
    create_cube_depth_texture,
    create_depth_fbo,
    create_depth_texture_2d,
    create_texture_2d,
    read_rgba_float,
    read_rgba_u8,
)
from .mathutil import downsample_box, normalize
from .scene import Light, Scene
from .shaders import (
    COMPOSITE_FRAGMENT,
    DEPTH_FRAGMENT,
    DEPTH_VERTEX,
    DIRECT_LIGHT_FRAGMENT,
    DOWNSAMPLE_FRAGMENT,
    FULLSCREEN_VERTEX,
    GBUFFER_FRAGMENT,
    GBUFFER_VERTEX,
    INDIRECT_FRAGMENT,
    TONEMAP_FRAGMENT,
)


@dataclass
class ColorTarget:
    name: str
    width: int
    height: int
    texture: int
    fbo: int
    depth_buffer: int | None = None

    def delete(self) -> None:
        if self.depth_buffer:
            glDeleteRenderbuffers(1, [self.depth_buffer])
        if self.fbo:
            glDeleteFramebuffers(1, [self.fbo])
        if self.texture:
            glDeleteTextures(1, [self.texture])
        self.depth_buffer = 0
        self.fbo = 0
        self.texture = 0


@dataclass
class GBuffer:
    width: int
    height: int
    textures: list[int]
    fbo: int
    depth_buffer: int

    def delete(self) -> None:
        if self.depth_buffer:
            glDeleteRenderbuffers(1, [self.depth_buffer])
        if self.fbo:
            glDeleteFramebuffers(1, [self.fbo])
        if self.textures:
            glDeleteTextures(len(self.textures), self.textures)
        self.textures = []
        self.fbo = 0
        self.depth_buffer = 0


@dataclass
class Shadow2D:
    light_name: str
    size: int
    texture: int
    fbo: int
    matrix: np.ndarray

    def delete(self) -> None:
        if self.fbo:
            glDeleteFramebuffers(1, [self.fbo])
        if self.texture:
            glDeleteTextures(1, [self.texture])
        self.fbo = 0
        self.texture = 0


@dataclass
class ShadowCube:
    light_name: str
    size: int
    texture: int
    fbo: int
    matrices: list[np.ndarray]

    def delete(self) -> None:
        if self.fbo:
            glDeleteFramebuffers(1, [self.fbo])
        if self.texture:
            glDeleteTextures(1, [self.texture])
        self.fbo = 0
        self.texture = 0


@dataclass
class RenderResult:
    width: int
    height: int
    ssaa: int
    internal_width: int
    internal_height: int
    gbuffer_target: dict[str, np.ndarray]
    gbuffer_internal: dict[str, np.ndarray]
    direct: np.ndarray
    indirect: np.ndarray
    composite: np.ndarray
    final_rgb: np.ndarray
    gl_info: dict[str, Any]
    shadow_resources: dict[str, Any] = field(default_factory=dict)

    @property
    def coverage(self) -> np.ndarray:
        return self.gbuffer_target["normal"][..., 3]


class DeferredRenderer:
    """Reusable renderer; resolution/scene changes invalidate GL resources."""

    def __init__(self, gl_info: GLInfo) -> None:
        self.gl_info = gl_info
        self._programs: dict[str, int] = {}
        self._uniform_cache: dict[int, dict[str, int]] = {}
        self.fullscreen = FullscreenTriangle()
        self._scene_signature: tuple | None = None
        self.resource_generation = 0
        self._gbuffer: GBuffer | None = None
        self._internal_targets: dict[str, ColorTarget] = {}
        self._target_targets: dict[str, ColorTarget] = {}
        self._final_target: ColorTarget | None = None
        self._shadows_2d: dict[str, Shadow2D] = {}
        self._shadows_cube: dict[str, ShadowCube] = {}
        self._mesh_gpu: dict[str, MeshGPU] = {}
        self._normal_textures: dict[str, int] = {}
        self._normal_map_dummy = 0
        self._shadow_dummy_2d = 0
        self._shadow_dummy_cube = 0
        self._compiled = False

    @property
    def resource_signature(self) -> tuple | None:
        return self._scene_signature

    def _compile(self) -> None:
        if self._compiled:
            return
        self._programs = {
            "gbuffer": compile_program("g_buffer", GBUFFER_VERTEX, GBUFFER_FRAGMENT),
            "depth": compile_program("shadow_depth", DEPTH_VERTEX, DEPTH_FRAGMENT),
            "direct": compile_program("direct_light", FULLSCREEN_VERTEX, DIRECT_LIGHT_FRAGMENT),
            "indirect": compile_program("indirect_probe", FULLSCREEN_VERTEX, INDIRECT_FRAGMENT),
            "composite": compile_program("composite", FULLSCREEN_VERTEX, COMPOSITE_FRAGMENT),
            "downsample": compile_program("ssaa_downsample", FULLSCREEN_VERTEX, DOWNSAMPLE_FRAGMENT),
            "tonemap": compile_program("tone_map", FULLSCREEN_VERTEX, TONEMAP_FRAGMENT),
        }
        self._normal_map_dummy = create_texture_2d(
            1,
            1,
            GL_RGBA8,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            data=np.asarray([[[128, 128, 255, 255]]], dtype=np.uint8),
            min_filter=GL_NEAREST,
            mag_filter=GL_NEAREST,
        )
        self._shadow_dummy_2d = create_depth_texture_2d(1, 1)
        self._shadow_dummy_cube = create_cube_depth_texture(1)
        self._compiled = True
        check_gl("compile programs")

    def _u(self, program: int, name: str) -> int:
        locations = self._uniform_cache.setdefault(program, {})
        if name not in locations:
            locations[name] = glGetUniformLocation(program, name)
        return locations[name]

    def _set_int(self, program: int, name: str, value: int) -> None:
        location = self._u(program, name)
        if location >= 0:
            glUniform1i(location, int(value))

    def _set_float(self, program: int, name: str, value: float) -> None:
        location = self._u(program, name)
        if location >= 0:
            glUniform1f(location, float(value))

    def _set_vec3(self, program: int, name: str, value: np.ndarray) -> None:
        location = self._u(program, name)
        if location >= 0:
            glUniform3fv(location, 1, np.ascontiguousarray(value, dtype=np.float32))

    def _set_vec2(self, program: int, name: str, value: np.ndarray) -> None:
        location = self._u(program, name)
        if location >= 0:
            glUniform2fv(location, 1, np.ascontiguousarray(value, dtype=np.float32))

    def _set_vec4(self, program: int, name: str, value: np.ndarray) -> None:
        location = self._u(program, name)
        if location >= 0:
            glUniform4fv(location, 1, np.ascontiguousarray(value, dtype=np.float32))

    def _set_mat4(self, program: int, name: str, matrix: np.ndarray) -> None:
        location = self._u(program, name)
        if location >= 0:
            transpose = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32).T)
            glUniformMatrix4fv(location, 1, GL_FALSE, transpose)

    def _set_mat3(self, program: int, name: str, matrix: np.ndarray) -> None:
        location = self._u(program, name)
        if location >= 0:
            transpose = np.ascontiguousarray(np.asarray(matrix, dtype=np.float32).T)
            glUniformMatrix3fv(location, 1, GL_FALSE, transpose)

    def _bind_texture(self, unit: int, target: int, texture: int) -> None:
        glActiveTexture(GL_TEXTURE0 + unit)
        glBindTexture(target, texture)

    def _ensure_resources(self, scene: Scene) -> None:
        self._compile()
        settings = scene.rendering
        internal_width, internal_height = scene.internal_size
        signature = (
            scene.checksums["scene"],
            internal_width,
            internal_height,
            settings.ssaa,
            settings.shadows.enabled,
            settings.shadows.directional_resolution,
            settings.shadows.spot_resolution,
            settings.shadows.point_resolution,
        )
        if signature == self._scene_signature:
            return
        self._destroy_scene_resources()
        if internal_width > self.gl_info.max_texture_size or internal_height > self.gl_info.max_texture_size:
            raise RenderError(
                f"internal resolution {internal_width}x{internal_height} exceeds "
                f"GL_MAX_TEXTURE_SIZE={self.gl_info.max_texture_size}"
            )
        try:
            self._create_gbuffer(scene)
            self._create_targets(scene)
            self._create_scene_gpu(scene)
            self._create_shadows(scene)
            self._scene_signature = signature
            self.resource_generation += 1
            check_gl("resource creation")
        except Exception:
            self._destroy_scene_resources()
            raise

    def _create_gbuffer(self, scene: Scene) -> None:
        width, height = scene.internal_size
        formats = {
            "RGBA32F": (GL_RGBA32F, GL_RGBA, GL_FLOAT),
            "RGBA16F": (GL_RGBA16F, GL_RGBA, GL_FLOAT),
            "RGBA8": (GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE),
        }
        textures = []
        for attachment in GBUFFER_ATTACHMENTS:
            internal, pixel_format, pixel_type = formats[attachment.internal_format]
            textures.append(
                create_texture_2d(
                    width,
                    height,
                    internal,
                    pixel_format,
                    pixel_type,
                    min_filter=GL_NEAREST,
                    mag_filter=GL_NEAREST,
                )
            )
        fbo, depth_buffer = create_color_fbo(textures, width, height, depth_renderbuffer=True)
        self._gbuffer = GBuffer(width, height, textures, fbo, depth_buffer)

    def _create_targets(self, scene: Scene) -> None:
        width, height = scene.internal_size
        out_width, out_height = scene.rendering.width, scene.rendering.height
        for name in ("direct", "indirect", "composite"):
            texture = create_texture_2d(
                width, height, GL_RGBA32F, GL_RGBA, GL_FLOAT, min_filter=GL_NEAREST, mag_filter=GL_NEAREST
            )
            fbo, depth_buffer = create_color_fbo([texture], width, height, depth_renderbuffer=False)
            self._internal_targets[name] = ColorTarget(name, width, height, texture, fbo, depth_buffer)
            target_texture = create_texture_2d(
                out_width, out_height, GL_RGBA32F, GL_RGBA, GL_FLOAT, min_filter=GL_NEAREST, mag_filter=GL_NEAREST
            )
            target_fbo, target_depth = create_color_fbo(
                [target_texture], out_width, out_height, depth_renderbuffer=False
            )
            self._target_targets[name] = ColorTarget(
                f"target_{name}", out_width, out_height, target_texture, target_fbo, target_depth
            )
        final_texture = create_texture_2d(
            out_width, out_height, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, min_filter=GL_NEAREST, mag_filter=GL_NEAREST
        )
        final_fbo, final_depth = create_color_fbo(
            [final_texture], out_width, out_height, depth_renderbuffer=False
        )
        self._final_target = ColorTarget("final", out_width, out_height, final_texture, final_fbo, final_depth)

    def _create_scene_gpu(self, scene: Scene) -> None:
        for name, mesh in scene.meshes.items():
            self._mesh_gpu[name] = MeshGPU(mesh)
        for name, material in scene.materials.items():
            if material.normal_map is not None:
                texture = create_texture_2d(
                    material.normal_map.shape[1],
                    material.normal_map.shape[0],
                    GL_RGBA8,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    data=np.ascontiguousarray(material.normal_map),
                    min_filter=GL_LINEAR_MIPMAP_LINEAR,
                    mag_filter=GL_LINEAR,
                    wrap_s=GL_REPEAT,
                    wrap_t=GL_REPEAT,
                )
                glBindTexture(GL_TEXTURE_2D, texture)
                glGenerateMipmap(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, 0)
                self._normal_textures[name] = texture

    def _create_shadows(self, scene: Scene) -> None:
        if not scene.rendering.shadows.enabled:
            return
        for light in scene.lights:
            if not light.casts_shadow:
                continue
            if light.type == "directional":
                shadow = self._make_directional_shadow(scene, light)
                self._shadows_2d[light.name] = shadow
            elif light.type == "spot":
                shadow = self._make_spot_shadow(scene, light)
                self._shadows_2d[light.name] = shadow
            elif light.type == "point":
                shadow = self._make_point_shadow(scene, light)
                self._shadows_cube[light.name] = shadow

    def _make_directional_shadow(self, scene: Scene, light: Light) -> Shadow2D:
        size = scene.rendering.shadows.directional_resolution
        center = scene.bounds_center.astype(np.float64)
        radius = max(float(scene.bounds_radius), 1.0)
        to_light = normalize(-light.direction, fallback=np.asarray([0.0, 1.0, 0.0]))
        eye = center + to_light * (radius * 2.5 + 1.0)
        up = np.asarray([0.0, 1.0, 0.0]) if abs(to_light[1]) < 0.95 else np.asarray([1.0, 0.0, 0.0])
        view = np.asarray(glm.lookAt(glm.vec3(*eye), glm.vec3(*center), glm.vec3(*up)), dtype=np.float32)
        projection = np.asarray(
            glm.ortho(-radius * 1.15, radius * 1.15, -radius * 1.15, radius * 1.15, 0.01, radius * 5.0 + 2.0),
            dtype=np.float32,
        )
        texture = create_depth_texture_2d(size, size)
        fbo = create_depth_fbo(texture)
        return Shadow2D(light.name, size, texture, fbo, projection @ view)

    def _make_spot_shadow(self, scene: Scene, light: Light) -> Shadow2D:
        size = scene.rendering.shadows.spot_resolution
        position = light.position.astype(np.float64)
        forward = normalize(light.direction)
        up = np.asarray([0.0, 1.0, 0.0]) if abs(forward[1]) < 0.95 else np.asarray([1.0, 0.0, 0.0])
        target = position + forward
        view = np.asarray(glm.lookAt(glm.vec3(*position), glm.vec3(*target), glm.vec3(*up)), dtype=np.float32)
        projection = np.asarray(
            glm.perspective(
                glm.radians(float(light.outer_deg) * 2.0 + 1.0),
                1.0,
                0.05,
                float(light.range),
            ),
            dtype=np.float32,
        )
        texture = create_depth_texture_2d(size, size)
        fbo = create_depth_fbo(texture)
        return Shadow2D(light.name, size, texture, fbo, projection @ view)

    def _make_point_shadow(self, scene: Scene, light: Light) -> ShadowCube:
        size = scene.rendering.shadows.point_resolution
        texture = create_cube_depth_texture(size)
        fbo = create_depth_fbo(texture, GL_TEXTURE_CUBE_MAP_POSITIVE_X)
        position = light.position.astype(np.float64)
        directions = (
            (np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, -1.0, 0.0])),
            (np.asarray([-1.0, 0.0, 0.0]), np.asarray([0.0, -1.0, 0.0])),
            (np.asarray([0.0, 1.0, 0.0]), np.asarray([0.0, 0.0, 1.0])),
            (np.asarray([0.0, -1.0, 0.0]), np.asarray([0.0, 0.0, -1.0])),
            (np.asarray([0.0, 0.0, 1.0]), np.asarray([0.0, -1.0, 0.0])),
            (np.asarray([0.0, 0.0, -1.0]), np.asarray([0.0, -1.0, 0.0])),
        )
        projection = np.asarray(
            glm.perspective(glm.radians(90.0), 1.0, 0.05, float(light.range)), dtype=np.float32
        )
        matrices = []
        for direction, up in directions:
            view = np.asarray(
                glm.lookAt(glm.vec3(*position), glm.vec3(*(position + direction)), glm.vec3(*up)),
                dtype=np.float32,
            )
            matrices.append(projection @ view)
        return ShadowCube(light.name, size, texture, fbo, matrices)

    def _destroy_scene_resources(self) -> None:
        if self._gbuffer is not None:
            self._gbuffer.delete()
            self._gbuffer = None
        for target in list(self._internal_targets.values()) + list(self._target_targets.values()):
            target.delete()
        self._internal_targets.clear()
        self._target_targets.clear()
        if self._final_target is not None:
            self._final_target.delete()
            self._final_target = None
        for shadow in self._shadows_2d.values():
            shadow.delete()
        self._shadows_2d.clear()
        for shadow in self._shadows_cube.values():
            shadow.delete()
        self._shadows_cube.clear()
        for mesh in self._mesh_gpu.values():
            mesh.delete()
        self._mesh_gpu.clear()
        if self._normal_textures:
            glDeleteTextures(len(self._normal_textures), list(self._normal_textures.values()))
            self._normal_textures.clear()
        self._scene_signature = None
        check_gl("resource destruction")

    def render(self, scene: Scene) -> RenderResult:
        self._ensure_resources(scene)
        assert self._gbuffer is not None
        assert self._final_target is not None
        width, height = scene.internal_size
        view, projection, view_projection = scene.camera.matrices(width / height)
        del projection

        glDisable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDisable(GL_BLEND)
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE)

        self._render_shadows(scene)

        glBindFramebuffer(GL_FRAMEBUFFER, self._gbuffer.fbo)
        glViewport(0, 0, width, height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        program = self._programs["gbuffer"]
        glUseProgram(program)
        self._set_mat4(program, "u_view_proj", view_projection)
        self._set_vec3(program, "u_camera_position", scene.camera.position)
        self._set_int(program, "u_normal_map", 1)
        for node in scene.nodes:
            mesh = self._mesh_gpu[node.mesh_name]
            material = scene.materials[node.material_name]
            self._set_mat4(program, "u_model", node.world_matrix)
            rotation_scale = np.asarray(node.world_matrix, dtype=np.float64)[:3, :3]
            normal_matrix = np.linalg.inv(rotation_scale).T
            self._set_mat3(program, "u_normal_matrix", normal_matrix)
            self._set_vec3(program, "u_albedo", material.albedo)
            self._set_float(program, "u_metallic", material.metallic)
            self._set_float(program, "u_roughness", material.roughness)
            self._set_float(program, "u_normal_strength", material.normal_strength)
            texture = self._normal_textures.get(node.material_name, self._normal_map_dummy)
            self._set_int(program, "u_has_normal_map", 1 if node.material_name in self._normal_textures else 0)
            self._bind_texture(1, GL_TEXTURE_2D, texture)
            mesh.draw()

        self._render_direct(scene)
        self._render_indirect(scene)
        self._composite()
        self._downsample_targets(scene)
        self._tonemap(scene)
        check_gl("render passes")
        return self._read_result(scene)

    def _render_shadows(self, scene: Scene) -> None:
        if not scene.rendering.shadows.enabled:
            return
        program = self._programs["depth"]
        glUseProgram(program)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDisable(GL_BLEND)
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.5, 2.0)
        for light in scene.lights:
            if not light.casts_shadow:
                continue
            if light.name in self._shadows_2d:
                shadow = self._shadows_2d[light.name]
                glBindFramebuffer(GL_FRAMEBUFFER, shadow.fbo)
                glViewport(0, 0, shadow.size, shadow.size)
                glClear(GL_DEPTH_BUFFER_BIT)
                self._set_mat4(program, "u_light_view_proj", shadow.matrix)
                for node in scene.nodes:
                    self._set_mat4(program, "u_model", node.world_matrix)
                    self._mesh_gpu[node.mesh_name].draw()
            elif light.name in self._shadows_cube:
                shadow = self._shadows_cube[light.name]
                glBindFramebuffer(GL_FRAMEBUFFER, shadow.fbo)
                glViewport(0, 0, shadow.size, shadow.size)
                for face_index, matrix in enumerate(shadow.matrices):
                    glFramebufferTexture2D(
                        GL_FRAMEBUFFER,
                        GL_DEPTH_ATTACHMENT,
                        GL_TEXTURE_CUBE_MAP_POSITIVE_X + face_index,
                        shadow.texture,
                        0,
                    )
                    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
                    if status != GL_FRAMEBUFFER_COMPLETE:
                        raise RenderError(f"point-shadow FBO incomplete on face {face_index}: 0x{status:04x}")
                    glClear(GL_DEPTH_BUFFER_BIT)
                    self._set_mat4(program, "u_light_view_proj", matrix)
                    for node in scene.nodes:
                        self._set_mat4(program, "u_model", node.world_matrix)
                        self._mesh_gpu[node.mesh_name].draw()
        glDisable(GL_POLYGON_OFFSET_FILL)

    def _bind_gbuffer(self, program: int) -> None:
        assert self._gbuffer is not None
        for index in range(4):
            self._bind_texture(index, GL_TEXTURE_2D, self._gbuffer.textures[index])
        for name in ("u_g_position", "u_g_normal", "u_g_albedo", "u_g_material"):
            self._set_int(program, name, ("u_g_position", "u_g_normal", "u_g_albedo", "u_g_material").index(name))

    def _render_direct(self, scene: Scene) -> None:
        target = self._internal_targets["direct"]
        glBindFramebuffer(GL_FRAMEBUFFER, target.fbo)
        glViewport(0, 0, target.width, target.height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE)
        program = self._programs["direct"]
        glUseProgram(program)
        self._bind_gbuffer(program)
        self._bind_texture(4, GL_TEXTURE_2D, self._shadow_dummy_2d)
        self._bind_texture(5, GL_TEXTURE_CUBE_MAP, self._shadow_dummy_cube)
        self._set_int(program, "u_shadow_map", 4)
        self._set_int(program, "u_shadow_cube", 5)
        self._set_vec3(program, "u_camera_position", scene.camera.position)
        shadows = scene.rendering.shadows
        for light in scene.lights:
            self._set_int(program, "u_light_type", {"directional": 0, "point": 1, "spot": 2}[light.type])
            self._set_vec3(program, "u_light_direction", light.direction)
            self._set_vec3(program, "u_light_color", light.color)
            self._set_float(program, "u_light_intensity", light.intensity)
            self._set_float(program, "u_light_range", light.range)
            self._set_float(program, "u_cos_inner", np.cos(np.deg2rad(light.inner_deg)))
            self._set_float(program, "u_cos_outer", np.cos(np.deg2rad(light.outer_deg)))
            if light.position is not None:
                self._set_vec3(program, "u_light_position", light.position)
            else:
                self._set_vec3(program, "u_light_position", np.zeros(3))
            self._set_float(program, "u_shadow_bias", shadows.bias)
            self._set_float(program, "u_shadow_normal_bias", shadows.normal_bias)
            self._set_int(program, "u_filter_radius", shadows.filter_radius)
            self._set_int(program, "u_filter_taps", shadows.filter_taps)
            if shadows.enabled and light.casts_shadow and light.name in self._shadows_2d:
                shadow = self._shadows_2d[light.name]
                self._set_int(program, "u_shadow_kind", 1)
                self._bind_texture(4, GL_TEXTURE_2D, shadow.texture)
                self._set_mat4(program, "u_light_vp", shadow.matrix)
                self._set_vec2(program, "u_shadow_texel", np.asarray([1.0 / shadow.size, 1.0 / shadow.size]))
            elif shadows.enabled and light.casts_shadow and light.name in self._shadows_cube:
                shadow = self._shadows_cube[light.name]
                self._set_int(program, "u_shadow_kind", 2)
                self._bind_texture(5, GL_TEXTURE_CUBE_MAP, shadow.texture)
                angular_texel = 2.0 / float(shadow.size)
                self._set_vec2(program, "u_shadow_texel", np.asarray([angular_texel, angular_texel]))
            else:
                self._set_int(program, "u_shadow_kind", 0)
            self.fullscreen.draw()
        glDisable(GL_BLEND)

    def _render_indirect(self, scene: Scene) -> None:
        target = self._internal_targets["indirect"]
        glBindFramebuffer(GL_FRAMEBUFFER, target.fbo)
        glViewport(0, 0, target.width, target.height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        program = self._programs["indirect"]
        glUseProgram(program)
        self._bind_gbuffer(program)
        self._set_vec3(program, "u_camera_position", scene.camera.position)
        self._set_vec3(program, "u_sky_color", scene.indirect.sky_color)
        self._set_vec3(program, "u_ground_color", scene.indirect.ground_color)
        self._set_float(program, "u_probe_intensity", scene.indirect.intensity)
        self._set_int(program, "u_indirect_enabled", 1 if scene.indirect.enabled else 0)
        self.fullscreen.draw()

    def _composite(self) -> None:
        target = self._internal_targets["composite"]
        glBindFramebuffer(GL_FRAMEBUFFER, target.fbo)
        glViewport(0, 0, target.width, target.height)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        program = self._programs["composite"]
        glUseProgram(program)
        self._bind_texture(0, GL_TEXTURE_2D, self._internal_targets["direct"].texture)
        self._bind_texture(1, GL_TEXTURE_2D, self._internal_targets["indirect"].texture)
        self._set_int(program, "u_direct", 0)
        self._set_int(program, "u_indirect", 1)
        self.fullscreen.draw()

    def _downsample_targets(self, scene: Scene) -> None:
        program = self._programs["downsample"]
        glUseProgram(program)
        self._set_int(program, "u_source", 0)
        self._set_int(program, "u_ssaa", scene.rendering.ssaa)
        for name in ("direct", "indirect", "composite"):
            source = self._internal_targets[name]
            target = self._target_targets[name]
            glBindFramebuffer(GL_FRAMEBUFFER, target.fbo)
            glViewport(0, 0, target.width, target.height)
            glClear(GL_COLOR_BUFFER_BIT)
            self._bind_texture(0, GL_TEXTURE_2D, source.texture)
            self.fullscreen.draw()

    def _tonemap(self, scene: Scene) -> None:
        assert self._final_target is not None
        glBindFramebuffer(GL_FRAMEBUFFER, self._final_target.fbo)
        glViewport(0, 0, self._final_target.width, self._final_target.height)
        glClear(GL_COLOR_BUFFER_BIT)
        program = self._programs["tonemap"]
        glUseProgram(program)
        self._bind_texture(0, GL_TEXTURE_2D, self._internal_targets["composite"].texture)
        self._set_int(program, "u_source", 0)
        self._set_int(program, "u_ssaa", scene.rendering.ssaa)
        self._set_float(program, "u_exposure", scene.rendering.exposure)
        self._set_int(program, "u_operator", 0 if scene.rendering.tone_mapper == "aces" else 1)
        self.fullscreen.draw()

    def _read_result(self, scene: Scene) -> RenderResult:
        assert self._gbuffer is not None
        assert self._final_target is not None
        internal: dict[str, np.ndarray] = {}
        for attachment in GBUFFER_ATTACHMENTS:
            glBindFramebuffer(GL_READ_FRAMEBUFFER, self._gbuffer.fbo)
            glReadBuffer(GL_COLOR_ATTACHMENT0 + attachment.location)
            if attachment.pixel_type == "UNSIGNED_BYTE":
                internal[attachment.name] = read_rgba_u8(self._gbuffer.width, self._gbuffer.height)
            else:
                internal[attachment.name] = read_rgba_float(self._gbuffer.width, self._gbuffer.height)
        target_linear: dict[str, np.ndarray] = {}
        for name in ("direct", "indirect", "composite"):
            target = self._target_targets[name]
            glBindFramebuffer(GL_READ_FRAMEBUFFER, target.fbo)
            glReadBuffer(GL_COLOR_ATTACHMENT0)
            target_linear[name] = read_rgba_float(target.width, target.height)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._final_target.fbo)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        final_rgba = read_rgba_u8(self._final_target.width, self._final_target.height)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)

        ssaa = scene.rendering.ssaa
        gbuffer_target: dict[str, np.ndarray] = {}
        gbuffer_target["position"] = downsample_box(internal["position"], ssaa).astype(np.float32)
        normal = downsample_box(internal["normal"].astype(np.float32), ssaa).astype(np.float32)
        coverage = np.clip(normal[..., 3], 0.0, 1.0)
        xyz = normal[..., :3].astype(np.float64)
        lengths = np.linalg.norm(xyz, axis=2, keepdims=True)
        xyz = np.divide(xyz, np.maximum(lengths, 1e-8))
        normal[..., :3] = xyz.astype(np.float32)
        normal[..., 3] = coverage
        gbuffer_target["normal"] = normal
        albedo = downsample_box(internal["albedo"].astype(np.float32) / 255.0, ssaa).astype(np.float32)
        gbuffer_target["albedo"] = albedo
        gbuffer_target["material"] = downsample_box(internal["material"], ssaa).astype(np.float32)

        if self._gbuffer is None or self._final_target is None:
            raise RenderError("render targets disappeared before readback")
        shadow_info = {
            "2d": {name: {"size": shadow.size} for name, shadow in self._shadows_2d.items()},
            "cube": {name: {"size": shadow.size} for name, shadow in self._shadows_cube.items()},
        }
        return RenderResult(
            width=scene.rendering.width,
            height=scene.rendering.height,
            ssaa=scene.rendering.ssaa,
            internal_width=scene.internal_size[0],
            internal_height=scene.internal_size[1],
            gbuffer_target=gbuffer_target,
            gbuffer_internal=internal,
            direct=target_linear["direct"][..., :3].astype(np.float32),
            indirect=target_linear["indirect"][..., :3].astype(np.float32),
            composite=target_linear["composite"][..., :3].astype(np.float32),
            final_rgb=final_rgba[..., :3].astype(np.uint8),
            gl_info={
                "version": self.gl_info.version,
                "renderer": self.gl_info.renderer,
                "glsl_version": self.gl_info.glsl_version,
                "vendor": self.gl_info.vendor,
                "max_texture_size": self.gl_info.max_texture_size,
            },
            shadow_resources=shadow_info,
        )

    def close(self) -> None:
        if not self._compiled:
            return
        self._destroy_scene_resources()
        if self._normal_map_dummy:
            glDeleteTextures(1, [self._normal_map_dummy])
            self._normal_map_dummy = 0
        if self._shadow_dummy_2d:
            glDeleteTextures(1, [self._shadow_dummy_2d])
            self._shadow_dummy_2d = 0
        if self._shadow_dummy_cube:
            glDeleteTextures(1, [self._shadow_dummy_cube])
            self._shadow_dummy_cube = 0
        for program in self._programs.values():
            glDeleteProgram(program)
        self._programs.clear()
        self.fullscreen.delete()
        self._compiled = False
