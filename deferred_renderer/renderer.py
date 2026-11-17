from __future__ import annotations

import hashlib
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import moderngl
import numpy as np

from .culling import (
    CullingResult,
    build_tile_lights,
    coc_statistics,
)
from .errors import RenderError
from .math3d import matrix_bytes
from .reference import compare_reference, fullscreen_shading_reference
from .scene import (
    MAX_LIGHTS,
    MAX_MATERIALS,
    AssetBundle,
    Scene,
)
from .window_backends import create_window_backend


@dataclass
class GLWindow:
    backend: Any


@dataclass
class GLContextBundle:
    context: moderngl.Context
    mode: str
    window: GLWindow | None
    gl_info: dict[str, Any]


@dataclass
class RenderResult:
    image_rgba: np.ndarray
    stats: dict[str, Any]


@dataclass
class GpuObject:
    vao: Any
    vertex_buffer: Any
    index_buffer: Any
    material_textures: tuple[Any, Any, Any]


_SHADER_DIR = Path(__file__).with_name("shaders")


def _read_shader(name: str) -> str:
    try:
        return (_SHADER_DIR / name).read_text(encoding="utf-8")
    except OSError as exc:
        raise RenderError(
            "E_SHADER_LOAD",
            f"cannot load shader {name}",
            {"shader": name, "reason": str(exc),
             },
            exit_code=3,
        ) from exc


def _make_program(
    context: moderngl.Context,
    vertex: str,
    fragment: str,
) -> moderngl.Program:
    try:
        return context.program(
            vertex_shader=_read_shader(vertex),
            fragment_shader=_read_shader(fragment),
        )
    except Exception as exc:
        raise RenderError(
            "E_SHADER_COMPILE",
            "OpenGL shader compilation failed",
            {"vertex": vertex, "fragment": fragment, "reason": str(exc)},
            exit_code=3,
        ) from exc


def _create_headless_context() -> moderngl.Context:
    errors: list[str] = []
    if sys.platform.startswith("linux"):
        old_platform = os.environ.get("EGL_PLATFORM")
        try:
            for candidate in ("surfaceless", "device"):
                os.environ["EGL_PLATFORM"] = candidate
                try:
                    return moderngl.create_standalone_context(
                        backend="egl", require=330
                    )
                except Exception as exc:
                    errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
        finally:
            if old_platform is None:
                os.environ.pop("EGL_PLATFORM", None)
            else:
                os.environ["EGL_PLATFORM"] = old_platform
    try:
        return moderngl.create_standalone_context(require=330)
    except Exception as exc:
        errors.append(f"default: {type(exc).__name__}: {exc}")
    raise RenderError(
        "E_GL_CONTEXT",
        "failed to create an OpenGL 3.3 headless context",
        {"attempts": errors},
        exit_code=3,
    )


def _create_window_context(
    width: int,
    height: int,
    hidden: bool,
) -> tuple[moderngl.Context, GLWindow]:
    try:
        backend = create_window_backend(width, height, hidden)
    except Exception as exc:
        raise RenderError(
            "E_WINDOW_CREATE",
            "failed to create the OpenGL window",
            {
                "width": width,
                "height": height,
                "hidden": hidden,
                "reason": f"{type(exc).__name__}: {exc}",
            },
            exit_code=3,
        ) from exc

    try:
        context = moderngl.create_context(require=330)
        return context, GLWindow(backend=backend)
    except Exception as exc:
        backend.close()
        raise RenderError(
            "E_WINDOW_CREATE",
            "failed to attach ModernGL to the window context",
            {
                "width": width,
                "height": height,
                "hidden": hidden,
                "backend": backend.name,
                "reason": f"{type(exc).__name__}: {exc}",
            },
            exit_code=3,
        ) from exc


@contextmanager
def open_gl_context(
    mode: str,
    width: int,
    height: int,
    *,
    hidden: bool = False,
) -> Iterator[GLContextBundle]:
    if mode == "headless":
        context = _create_headless_context()
        info = dict(context.info)
        try:
            yield GLContextBundle(
                context=context,
                mode=mode,
                window=None,
                gl_info=info,
            )
        finally:
            context.release()
        return
    if mode != "window":
        raise RenderError(
            "E_BAD_MODE",
            "mode must be headless or window",
            {"mode": mode},
        )
    context, window = _create_window_context(width, height, hidden)
    info = dict(context.info)
    try:
        yield GLContextBundle(
            context=context,
            mode=mode,
            window=window,
            gl_info=info,
        )
    finally:
        context.release()
        window.backend.close()


def _texture_read_f2(
    framebuffer: moderngl.Framebuffer,
    attachment: int,
    width: int,
    height: int,
) -> np.ndarray:
    payload = framebuffer.read(components=4, dtype="f2", attachment=attachment)
    return np.frombuffer(payload, dtype="<f2").reshape(height, width, 4).astype(np.float32)


def _texture_read_f4(
    framebuffer: moderngl.Framebuffer,
    attachment: int,
    width: int,
    height: int,
) -> np.ndarray:
    payload = framebuffer.read(components=4, dtype="f4", attachment=attachment)
    return np.frombuffer(payload, dtype="<f4").reshape(height, width, 4).astype(np.float32)


def _texture_read_r32f(
    framebuffer: moderngl.Framebuffer,
    attachment: int,
    width: int,
    height: int,
) -> np.ndarray:
    payload = framebuffer.read(components=1, dtype="f4", attachment=attachment)
    return np.frombuffer(payload, dtype="<f4").reshape(height, width).astype(np.float32)


def _texture_read_r32ui(
    framebuffer: moderngl.Framebuffer,
    attachment: int,
    width: int,
    height: int,
) -> np.ndarray:
    payload = framebuffer.read(components=1, dtype="u4", attachment=attachment)
    return np.frombuffer(payload, dtype="<u4").reshape(height, width).astype(np.uint32)


def _texture_read_rgba8(
    framebuffer: moderngl.Framebuffer,
    width: int,
    height: int,
) -> np.ndarray:
    payload = framebuffer.read(components=4, dtype="f1")
    image = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 4)
    return np.ascontiguousarray(image[::-1, :, :])


def _pixel_statistics(array: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    values = np.asarray(array, dtype=np.float64)
    if mask is not None:
        values = values[mask]
    if values.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "finite": True,
        }
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "finite": bool(np.all(np.isfinite(values))),
    }


def _per_channel_statistics(array: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    names = ("x", "y", "z", "w")[: array.shape[-1]]
    for channel, name in enumerate(names):
        result[name] = _pixel_statistics(array[..., channel], mask)
    return result


class DeferredRenderer:
    def __init__(self, scene: Scene, assets: AssetBundle, context: moderngl.Context):
        self.scene = scene
        self.assets = assets
        self.context = context
        self.width = scene.render.width
        self.height = scene.render.height
        self._created_resources: list[Any] = []
        self._window_bundle: GLWindow | None = None
        self._build_programs()
        self._build_objects()
        self._build_framebuffers()

    def _build_programs(self) -> None:
        self.gbuffer_program = _make_program(
            self.context, "gbuffer.vert", "gbuffer.frag"
        )
        self.clear_program = _make_program(
            self.context, "fullscreen.vert", "clear_gbuffer.frag"
        )
        self.deferred_program = _make_program(
            self.context, "fullscreen.vert", "deferred.frag"
        )
        self.dof_program = _make_program(
            self.context, "fullscreen.vert", "dof.frag"
        )
        self.tonemap_program = _make_program(
            self.context, "fullscreen.vert", "tonemap.frag"
        )
        self.window_program = _make_program(
            self.context, "fullscreen.vert", "window_blit.frag"
        )
        self.clear_vao = self.context.vertex_array(self.clear_program, [])
        self.deferred_vao = self.context.vertex_array(self.deferred_program, [])
        self.dof_vao = self.context.vertex_array(self.dof_program, [])
        self.tonemap_vao = self.context.vertex_array(self.tonemap_program, [])
        self.window_vao = self.context.vertex_array(self.window_program, [])
        self._created_resources.extend(
            [
                self.gbuffer_program,
                self.clear_program,
                self.deferred_program,
                self.dof_program,
                self.tonemap_program,
                self.window_program,
                self.clear_vao,
                self.deferred_vao,
                self.dof_vao,
                self.tonemap_vao,
                self.window_vao,
            ]
        )

    def _build_objects(self) -> None:
        self.gpu_objects: list[GpuObject] = []
        self._material_textures: dict[int, tuple[Any, Any, Any]] = {}
        for material in self.scene.materials:
            texture = self.assets.textures[material.id]
            albedo = self.context.texture(
                (texture.size, texture.size),
                4,
                data=texture.albedo,
                dtype="f1",
            )
            roughness = self.context.texture(
                (texture.size, texture.size),
                3,
                data=texture.roughness,
                dtype="f1",
            )
            normal = self.context.texture(
                (texture.size, texture.size),
                3,
                data=texture.normal,
                dtype="f1",
            )
            for item in (albedo, roughness, normal):
                item.filter = (moderngl.LINEAR, moderngl.LINEAR)
                item.repeat_x = True
                item.repeat_y = True
            self._material_textures[material.id] = (albedo, roughness, normal)
            self._created_resources.extend([albedo, roughness, normal])

        for obj, mesh in zip(self.scene.objects, self.assets.meshes, strict=True):
            vertex_buffer = self.context.buffer(mesh.vertex_bytes)
            index_buffer = self.context.buffer(mesh.index_bytes)
            vao = self.context.vertex_array(
                self.gbuffer_program,
                [
                    (
                        vertex_buffer,
                        "3f 3f 4f 2f",
                        "in_position",
                        "in_normal",
                        "in_tangent",
                        "in_uv",
                    )
                ],
                index_buffer=index_buffer,
                index_element_size=4,
            )
            self.gpu_objects.append(
                GpuObject(
                    vao=vao,
                    vertex_buffer=vertex_buffer,
                    index_buffer=index_buffer,
                    material_textures=self._material_textures[obj.material_id],
                )
            )
            self._created_resources.extend([vertex_buffer, index_buffer, vao])

    def _build_framebuffers(self) -> None:
        width, height = self.width, self.height
        self.albedo_roughness = self.context.texture(
            (width, height), 4, dtype="f2"
        )
        self.normal_side = self.context.texture((width, height), 4, dtype="f2")
        self.linear_depth = self.context.texture((width, height), 1, dtype="f4")
        self.material_ids = self.context.texture((width, height), 1, dtype="u4")
        for texture in (
            self.albedo_roughness,
            self.normal_side,
            self.linear_depth,
            self.material_ids,
        ):
            texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            texture.repeat_x = False
            texture.repeat_y = False
        self.depth_renderbuffer = self.context.depth_renderbuffer((width, height))
        self.gbuffer_fbo = self.context.framebuffer(
            color_attachments=[
                self.albedo_roughness,
                self.normal_side,
                self.linear_depth,
                self.material_ids,
            ],
            depth_attachment=self.depth_renderbuffer,
        )

        self.hdr_texture = self.context.texture((width, height), 4, dtype="f4")
        self.dof_texture = self.context.texture((width, height), 4, dtype="f4")
        for texture in (self.hdr_texture, self.dof_texture):
            texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            texture.repeat_x = False
            texture.repeat_y = False
        self.hdr_fbo = self.context.framebuffer(color_attachments=[self.hdr_texture])
        self.dof_fbo = self.context.framebuffer(color_attachments=[self.dof_texture])

        self.final_texture = self.context.texture((width, height), 4, dtype="f1")
        self.final_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.final_texture.repeat_x = False
        self.final_texture.repeat_y = False
        self.final_fbo = self.context.framebuffer(color_attachments=[self.final_texture])

        words_per_tile = math.ceil((len(self.scene.lights) + 1) / 4)
        tiles_x = math.ceil(width / self.scene.render.tile_size)
        tiles_y = math.ceil(height / self.scene.render.tile_size)
        self.tile_lights_texture = self.context.texture(
            (words_per_tile, tiles_x * tiles_y),
            4,
            data=np.zeros(
                (words_per_tile * (tiles_x * tiles_y), 4),
                dtype=np.uint32,
            ).tobytes(),
            dtype="u4",
        )
        self.tile_lights_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tile_lights_texture.repeat_x = False
        self.tile_lights_texture.repeat_y = False
        self._created_resources.extend(
            [
                self.albedo_roughness,
                self.normal_side,
                self.linear_depth,
                self.material_ids,
                self.depth_renderbuffer,
                self.gbuffer_fbo,
                self.hdr_texture,
                self.dof_texture,
                self.hdr_fbo,
                self.dof_fbo,
                self.final_texture,
                self.final_fbo,
                self.tile_lights_texture,
            ]
        )

    def _clear_gbuffer(self) -> None:
        self.gbuffer_fbo.use()
        self.context.viewport = (0, 0, self.width, self.height)
        self.context.scissor = None
        self.context.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self.context.disable(moderngl.DEPTH_TEST)
        self.clear_vao.render(vertices=3)
        self.context.enable(moderngl.DEPTH_TEST)
        self.context.depth_func = "<="
        self.context.disable(moderngl.BLEND)
        self.context.disable(moderngl.CULL_FACE)

    def _draw_gbuffer_tiles(self) -> None:
        render = self.scene.render
        tile_size = render.tile_size
        program = self.gbuffer_program
        program["u_view"].write(matrix_bytes(self.scene.camera.view))
        program["u_projection"].write(matrix_bytes(self.scene.projection))
        program["u_near"].value = self.scene.camera.near
        program["u_far"].value = self.scene.camera.far
        program["u_albedo_texture"].value = 0
        program["u_roughness_texture"].value = 1
        program["u_normal_texture"].value = 2
        self.context.disable(moderngl.BLEND)
        self.context.disable(moderngl.CULL_FACE)
        self.context.enable(moderngl.DEPTH_TEST)
        self.context.depth_func = "<="

        for tile_y in range(0, self.height, tile_size):
            tile_height = min(tile_size, self.height - tile_y)
            for tile_x in range(0, self.width, tile_size):
                tile_width = min(tile_size, self.width - tile_x)
                self.context.scissor = (tile_x, tile_y, tile_width, tile_height)
                for obj, gpu_object in zip(
                    self.scene.objects, self.gpu_objects, strict=True
                ):
                    material = self.scene.materials[obj.material_id - 1]
                    self._bind_object_uniforms(obj, material, gpu_object, program)
                    gpu_object.vao.render()
        self.context.scissor = None

    def _bind_object_uniforms(
        self,
        obj: Any,
        material: Any,
        gpu_object: GpuObject,
        program: moderngl.Program,
    ) -> None:
        model = obj.model
        program["u_model"].write(matrix_bytes(model))
        program["u_normal_matrix"].write(
            matrix_bytes(np.linalg.inv(model[:3, :3]).T)
        )
        program["u_uv_scale"].value = tuple(material.uv_scale)
        program["u_uv_offset"].value = (0.0, 0.0)
        program["u_normal_strength"].value = material.normal_strength
        program["u_material_id"].value = material.id
        albedo, roughness, normal = gpu_object.material_textures
        albedo.use(location=0)
        roughness.use(location=1)
        normal.use(location=2)

    def _configure_deferred_program(self, culling: CullingResult) -> None:
        program = self.deferred_program
        program["u_resolution"].value = (float(self.width), float(self.height))
        program["u_inv_projection"].write(matrix_bytes(self.scene.inv_projection))
        program["u_inv_view"].write(matrix_bytes(self.scene.camera.inv_view))
        program["u_camera_position"].value = self.scene.camera.position
        program["u_background"].value = self.scene.render.background
        program["u_ambient_strength"].value = self.scene.render.ambient_strength
        program["u_tile_size"].value = self.scene.render.tile_size
        program["u_tiles_x"].value = culling.tiles_x
        program["u_light_count"].value = len(self.scene.lights)
        light_positions = np.zeros((MAX_LIGHTS, 4), dtype=np.float32)
        light_colors = np.zeros((MAX_LIGHTS, 4), dtype=np.float32)
        back_scales = np.ones(MAX_MATERIALS + 1, dtype=np.float32)
        back_biases = np.zeros(MAX_MATERIALS + 1, dtype=np.float32)
        for index, light in enumerate(self.scene.lights):
            light_positions[index, :3] = light.position
            light_positions[index, 3] = light.radius
            light_colors[index, :3] = light.color
            light_colors[index, 3] = light.intensity
        for material in self.scene.materials:
            back_scales[material.id] = material.back_roughness_scale
            back_biases[material.id] = material.back_roughness_bias
        program["u_light_position_radius"].write(light_positions.tobytes(order="C"))
        program["u_light_color_intensity"].write(light_colors.tobytes(order="C"))
        program["u_material_back_roughness_scale"].write(
            back_scales.tobytes(order="C")
        )
        program["u_material_back_roughness_bias"].write(
            back_biases.tobytes(order="C")
        )

    def _draw_deferred_tiles(self, culling: CullingResult) -> None:
        program = self.deferred_program
        self.hdr_fbo.use()
        self.context.viewport = (0, 0, self.width, self.height)
        self.context.scissor = None
        self.context.clear(0.0, 0.0, 0.0, 1.0)
        self.context.disable(moderngl.DEPTH_TEST)
        self.context.disable(moderngl.BLEND)
        self.context.disable(moderngl.CULL_FACE)

        self.albedo_roughness.use(location=0)
        self.normal_side.use(location=1)
        self.linear_depth.use(location=2)
        self.material_ids.use(location=3)
        self.tile_lights_texture.use(location=4)
        program["u_gbuffer_albedo_roughness"].value = 0
        program["u_gbuffer_normal_side"].value = 1
        program["u_gbuffer_linear_depth"].value = 2
        program["u_gbuffer_material_id"].value = 3
        program["u_tile_lights"].value = 4

        tile_size = self.scene.render.tile_size
        for tile_y in range(0, self.height, tile_size):
            tile_height = min(tile_size, self.height - tile_y)
            for tile_x in range(0, self.width, tile_size):
                tile_width = min(tile_size, self.width - tile_x)
                self.context.scissor = (tile_x, tile_y, tile_width, tile_height)
                self.deferred_vao.render(vertices=3)
        self.context.scissor = None

    def _draw_dof(self) -> Any:
        render = self.scene.render
        if not render.dof_enabled or render.max_blur_radius <= 0.0:
            return self.hdr_texture
        program = self.dof_program
        self.dof_fbo.use()
        self.context.viewport = (0, 0, self.width, self.height)
        self.context.scissor = None
        self.context.clear(0.0, 0.0, 0.0, 1.0)
        self.context.disable(moderngl.DEPTH_TEST)
        self.context.disable(moderngl.BLEND)
        self.context.disable(moderngl.CULL_FACE)
        self.hdr_texture.use(location=0)
        self.linear_depth.use(location=1)
        program["u_hdr_texture"].value = 0
        program["u_linear_depth"].value = 1
        program["u_focus_distance"].value = render.focus_distance
        program["u_aperture"].value = render.aperture
        program["u_focal_length_m"].value = render.focal_length_mm * 0.001
        program["u_sensor_width_m"].value = render.sensor_width_mm * 0.001
        program["u_max_blur_radius"].value = render.max_blur_radius
        program["u_resolution_width"].value = float(self.width)
        self.dof_vao.render(vertices=3)
        return self.dof_texture

    def _draw_tonemap(self, source_texture: Any) -> None:
        program = self.tonemap_program
        curves = {"linear": 0, "reinhard": 1, "aces": 2, "filmic": 3}
        self.final_fbo.use()
        self.context.viewport = (0, 0, self.width, self.height)
        self.context.scissor = None
        self.context.disable(moderngl.DEPTH_TEST)
        self.context.disable(moderngl.BLEND)
        self.context.disable(moderngl.CULL_FACE)
        source_texture.use(location=0)
        program["u_hdr_texture"].value = 0
        program["u_exposure"].value = self.scene.render.exposure
        program["u_curve"].value = curves[self.scene.render.tone_curve]
        self.tonemap_vao.render(vertices=3)

    def _read_gbuffer(self) -> dict[str, np.ndarray]:
        return {
            "albedo_roughness": _texture_read_f2(
                self.gbuffer_fbo, 0, self.width, self.height
            ),
            "normal_side": _texture_read_f2(
                self.gbuffer_fbo, 1, self.width, self.height
            ),
            "linear_depth": _texture_read_r32f(
                self.gbuffer_fbo, 2, self.width, self.height
            ),
            "material_id": _texture_read_r32ui(
                self.gbuffer_fbo, 3, self.width, self.height
            ),
        }

    def _gbuffer_stats(self, gbuffer: dict[str, np.ndarray]) -> dict[str, Any]:
        albedo_roughness = gbuffer["albedo_roughness"]
        normal_side = gbuffer["normal_side"]
        depth = gbuffer["linear_depth"]
        material_id = gbuffer["material_id"]
        valid = depth > 0.0
        normal_valid = valid & (np.linalg.norm(normal_side[..., :3], axis=-1) > 1e-6)
        normal_length = np.linalg.norm(normal_side[..., :3], axis=-1)
        unique_ids = sorted(int(value) for value in np.unique(material_id) if value != 0)
        return {
            "attachments": [
                {
                    "name": "albedo_roughness",
                    "format": "RGBA16F",
                    "bits_per_channel": 16,
                    "range": "albedo RGB [0,1]; roughness A [0,1]",
                    "encoding": "linear RGB albedo, unfiltered base roughness",
                },
                {
                    "name": "normal_side",
                    "format": "RGBA16F",
                    "bits_per_channel": 16,
                    "range": "normal XYZ signed and normalized [-1,1]; side W {-1,+1}",
                    "encoding": "world-space signed normal; +1 front, -1 back",
                },
                {
                    "name": "linear_depth",
                    "format": "R32F",
                    "bits_per_channel": 32,
                    "range": f"visible geometry [{self.scene.camera.near},{self.scene.camera.far}]; 0 means background",
                    "encoding": "positive view-space depth, derived from clip depth and near/far",
                },
                {
                    "name": "material_id",
                    "format": "R32UI",
                    "bits_per_channel": 32,
                    "range": f"0..{len(self.scene.materials)}",
                    "encoding": "unsigned integer material id; 0 means background",
                },
            ],
            "scene_attachment_order": list(self.scene.render.attachments),
            "measured": {
                "valid_geometry_pixels": int(valid.sum()),
                "albedo_rgb": {
                    channel: _pixel_statistics(albedo_roughness[..., index], valid)
                    for index, channel in enumerate(("r", "g", "b"))
                },
                "roughness": _pixel_statistics(albedo_roughness[..., 3], valid),
                "normal_xyz": _per_channel_statistics(normal_side[..., :3], normal_valid),
                "normal_length": _pixel_statistics(normal_length, normal_valid),
                "front_side_pixels": int(np.count_nonzero(valid & (normal_side[..., 3] > 0.0))),
                "back_side_pixels": int(np.count_nonzero(valid & (normal_side[..., 3] < 0.0))),
                "linear_depth_geometry": _pixel_statistics(depth, valid),
                "linear_depth_min_including_background": float(np.min(depth)),
                "linear_depth_max_including_background": float(np.max(depth)),
                "background_pixels": int(np.count_nonzero(~valid)),
                "material_ids_present": unique_ids,
            },
        }

    def _hdr_stats(self, hdr: np.ndarray) -> dict[str, Any]:
        rgb = hdr[..., :3]
        return {
            "format": "RGBA32F",
            "min": float(np.min(rgb)),
            "max": float(np.max(rgb)),
            "mean": float(np.mean(rgb)),
            "finite": bool(np.all(np.isfinite(rgb))),
            "nan_count": int(np.count_nonzero(~np.isfinite(rgb))),
        }

    def _memory_estimate(self, culling: CullingResult) -> dict[str, Any]:
        width, height = self.width, self.height
        gbuffer_bytes = width * height * (8 + 8 + 4 + 4)
        depth_bytes = width * height * 4
        hdr_bytes = width * height * 4 * 4
        dof_bytes = width * height * 4 * 4
        final_bytes = width * height * 4
        tile_bytes = culling.packed_lights.nbytes
        geometry_bytes = sum(
            len(mesh.vertex_bytes) + len(mesh.index_bytes)
            for mesh in self.assets.meshes
        )
        texture_bytes = sum(
            texture.size * texture.size * (4 + 3 + 3)
            for texture in self.assets.textures.values()
        )
        total = (
            gbuffer_bytes
            + depth_bytes
            + hdr_bytes
            + dof_bytes
            + final_bytes
            + tile_bytes
            + geometry_bytes
            + texture_bytes
        )
        return {
            "estimated_total_bytes": int(total),
            "estimated_total_mb": float(total / (1024.0 * 1024.0)),
            "breakdown_bytes": {
                "gbuffer_color_attachments": int(gbuffer_bytes),
                "gbuffer_depth_renderbuffer": int(depth_bytes),
                "hdr_color": int(hdr_bytes),
                "dof_color": int(dof_bytes),
                "final_rgba8": int(final_bytes),
                "tile_light_index_texture": int(tile_bytes),
                "geometry_buffers": int(geometry_bytes),
                "procedural_textures": int(texture_bytes),
            },
            "note": "CPU-side estimate of persistent GPU allocations; driver overhead excluded",
        }

    def render_frame(
        self,
        *,
        mode: str,
        gl_info: dict[str, Any],
    ) -> RenderResult:
        started = time.perf_counter()
        timings: dict[str, float] = {}

        tick = time.perf_counter()
        self._clear_gbuffer()
        self._draw_gbuffer_tiles()
        timings["gbuffer_ms"] = (time.perf_counter() - tick) * 1000.0

        tick = time.perf_counter()
        gbuffer = self._read_gbuffer()
        timings["gbuffer_readback_ms"] = (time.perf_counter() - tick) * 1000.0

        tick = time.perf_counter()
        culling = build_tile_lights(self.scene, gbuffer["linear_depth"])
        self.tile_lights_texture.write(culling.packed_bytes)
        timings["tile_culling_ms"] = (time.perf_counter() - tick) * 1000.0

        tick = time.perf_counter()
        self._configure_deferred_program(culling)
        self._draw_deferred_tiles(culling)
        timings["deferred_lighting_ms"] = (time.perf_counter() - tick) * 1000.0

        tick = time.perf_counter()
        hdr = _texture_read_f4(self.hdr_fbo, 0, self.width, self.height)
        timings["hdr_readback_ms"] = (time.perf_counter() - tick) * 1000.0

        reference_stats: dict[str, Any] | None = None
        if self.scene.render.reference_check:
            tick = time.perf_counter()
            reference = fullscreen_shading_reference(
                self.scene,
                gbuffer["albedo_roughness"],
                gbuffer["normal_side"],
                gbuffer["linear_depth"],
                gbuffer["material_id"],
            )
            reference_stats = compare_reference(
                hdr,
                reference,
                gbuffer["linear_depth"],
                self.scene.render.reference_threshold,
            )
            timings["cpu_reference_ms"] = (time.perf_counter() - tick) * 1000.0
            if not reference_stats["passed"]:
                raise RenderError(
                    "E_REFERENCE_MISMATCH",
                    "deferred GPU result exceeds the CPU reference error threshold",
                    reference_stats,
                    exit_code=4,
                )

        tick = time.perf_counter()
        final_hdr_texture = self._draw_dof()
        timings["dof_ms"] = (time.perf_counter() - tick) * 1000.0

        tick = time.perf_counter()
        self._draw_tonemap(final_hdr_texture)
        timings["tonemap_ms"] = (time.perf_counter() - tick) * 1000.0

        tick = time.perf_counter()
        image = _texture_read_rgba8(self.final_fbo, self.width, self.height)
        pixel_hash = hashlib.sha256(image.tobytes(order="C")).hexdigest()
        timings["output_readback_ms"] = (time.perf_counter() - tick) * 1000.0
        timings["total_ms"] = (time.perf_counter() - started) * 1000.0

        coc_stats = coc_statistics(gbuffer["linear_depth"], self.scene.render)
        stats: dict[str, Any] = {
            "ok": True,
            "renderer": {
                "name": "deterministic-tiled-deferred",
                "version": "1.0.0",
            },
            "mode": mode,
            "resolution": {"width": self.width, "height": self.height},
            "render_execution": {
                "tile_order": "row_major_bottom_left",
                "gbuffer_scissor_tiles": True,
                "deferred_scissor_tiles": True,
                "fbo_and_texture_reuse": True,
                "tile_size": self.scene.render.tile_size,
            },
            "scene": {
                "sha256": self.scene.scene_sha256,
                "geometry_sha256": self.assets.geometry_sha256,
                "texture_sha256": self.assets.texture_sha256,
                "combined_asset_sha256": self.assets.combined_sha256,
                "object_count": len(self.scene.objects),
                "light_count": len(self.scene.lights),
                "material_count": len(self.scene.materials),
                "camera": {
                    "position": list(self.scene.camera.position),
                    "target": list(self.scene.camera.target),
                    "fov_y_deg": self.scene.camera.fov_y_deg,
                    "near": self.scene.camera.near,
                    "far": self.scene.camera.far,
                },
            },
            "g_buffer": self._gbuffer_stats(gbuffer),
            "tile_lighting": {
                **culling.stats,
                "lists": culling.tile_records,
            },
            "hdr": self._hdr_stats(hdr),
            "dof": {
                "enabled": bool(
                    self.scene.render.dof_enabled
                    and self.scene.render.max_blur_radius > 0.0
                ),
                "composition_order": ["background", "foreground"],
                "foreground_rule": "depth < focus_distance",
                "background_rule": "depth >= focus_distance or background",
                **coc_stats,
            },
            "tone_mapping": {
                "curve": self.scene.render.tone_curve,
                "exposure": self.scene.render.exposure,
                "srgb_encoding": True,
                "output_format": "RGBA8 PNG",
            },
            "timings_ms": timings,
            "memory_estimate": self._memory_estimate(culling),
            "gl": {
                "vendor": gl_info.get("GL_VENDOR"),
                "renderer": gl_info.get("GL_RENDERER"),
                "version": gl_info.get("GL_VERSION"),
            },
            "output": {
                "width": self.width,
                "height": self.height,
                "channels": "RGBA",
                "pixel_sha256": pixel_hash,
            },
        }
        if reference_stats is not None:
            stats["reference_check"] = reference_stats
        return RenderResult(image_rgba=image, stats=stats)

    def present(self) -> None:
        # The CLI owns window polling; this method only blits the already
        # rendered and saved RGBA8 frame to the default framebuffer.
        if self._window_bundle is None:
            return
        backend = self._window_bundle.backend
        width, height = backend.framebuffer_size()
        self.context.screen.use()
        self.context.viewport = (0, 0, width, height)
        self.context.scissor = None
        self.context.clear(0.0, 0.0, 0.0, 1.0)
        self.context.disable(moderngl.DEPTH_TEST)
        self.final_texture.use(location=0)
        self.window_program["u_frame_texture"].value = 0
        self.window_program["u_resolution"].value = (float(width), float(height))
        self.window_vao.render(vertices=3)
        backend.swap_buffers()

    def attach_window(self, bundle: GLContextBundle) -> None:
        self._window_bundle = bundle.window

    def run_window_loop(self, seconds: float) -> None:
        if self._window_bundle is None:
            return
        backend = self._window_bundle.backend
        self.present()
        deadline = time.monotonic() + max(0.0, seconds)
        while not backend.should_close():
            backend.poll_events()
            if backend.escape_pressed():
                break
            if time.monotonic() >= deadline:
                break
            self.present()
            time.sleep(1.0 / 60.0)

