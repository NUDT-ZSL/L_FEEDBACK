from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from OpenGL.GL import (
    GL_BACK,
    GL_BLEND,
    GL_COLOR_ATTACHMENT0,
    GL_COLOR_BUFFER_BIT,
    GL_CULL_FACE,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DRAW_FRAMEBUFFER,
    GL_FUNC_ADD,
    GL_FRAMEBUFFER,
    GL_INT,
    GL_LESS,
    GL_LINEAR,
    GL_NEAREST,
    GL_ONE,
    GL_R32I,
    GL_READ_FRAMEBUFFER,
    GL_RED_INTEGER,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_CUBE_MAP,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TRIANGLES,
    GL_UNSIGNED_INT,
    glActiveTexture,
    glBindFramebuffer,
    glBindTexture,
    glBlendEquation,
    glBlendFunc,
    glBlitFramebuffer,
    glClear,
    glClearColor,
    glDepthFunc,
    glDisable,
    glDrawBuffer,
    glEnable,
    glFinish,
    glGetIntegerv,
    glGenTextures,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
    glViewport,
)

from .clustering import (
    VirtualPointLight,
    cluster_lights,
    rank_vpl_importance,
    truncation_warning,
)
from .config import DiagnosticLog, Light, SceneConfig
from .cubemap_cache import VPLCubeScheduler
from .geometry import MeshData
from .gl_resources import (
    ColorTarget,
    DepthCube,
    FinalTarget,
    FullscreenTriangle,
    GBuffer,
    MeshGPU,
    ShaderProgram,
)
from .mathutil import light_view_projections, look_at, normalize, perspective, transform_points

LOGGER = logging.getLogger("proc_gi.renderer")
MAX_DELTA_VPLS = 128
SHADOW_BATCH_SIZE = 8


@dataclass
class CubeResource:
    cube: DepthCube


@dataclass
class RenderStats:
    frame: int = 0
    cluster_count: int = 0
    active_vpl_count: int = 0
    cube_builds: int = 0
    cube_reused: int = 0
    cube_released: int = 0
    delta_entries: int = 0
    subtract_entries: int = 0
    full_rebuild: bool = False
    truncated_vpl_count: int = 0
    material_draw_calls: int = 0
    accumulation_batches: int = 0

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class _Entry:
    sign: float
    vpl: VirtualPointLight
    resource: CubeResource


class Renderer:
    """OpenGL 3.3 renderer. A current core-profile context must already exist."""

    def __init__(
        self,
        config: SceneConfig,
        mesh: MeshData,
        diagnostics: DiagnosticLog | None = None,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.config = config
        self.mesh = mesh
        self.diagnostics = diagnostics or DiagnosticLog()
        self.width = int(width or config.render.width)
        self.height = int(height or config.render.height)
        if self.width < 64 or self.height < 64:
            raise ValueError("renderer resolution must be at least 64x64")
        self._mesh_gpu = MeshGPU(mesh)
        self._gbuffer = GBuffer(self.width, self.height)
        self._hdr = ColorTarget(self.width, self.height)
        self._final = FinalTarget(self.width, self.height)
        self._fullscreen = FullscreenTriangle()
        self._max_material_slot = max((material.id for material in config.materials), default=0)
        material_array_size = max(8, self._max_material_slot + 1)
        self._gbuffer_shader = ShaderProgram("g_buffer.vert", "g_buffer.frag")
        self._cube_shader = ShaderProgram("depth_cube.vert", "depth_cube.frag")
        self._vpl_shader = ShaderProgram(
            "post.vert",
            "vpl_accum.frag",
            defines={
                "MAX_DELTA": MAX_DELTA_VPLS,
                "MAX_MATERIALS": material_array_size,
                "MAX_SHADOWS": SHADOW_BATCH_SIZE,
            },
        )
        self._post_shader = ShaderProgram("post.vert", "post.frag")
        self._tile_size = int(config.clustering.tile_size)
        self._tiles_x = (self.width + self._tile_size - 1) // self._tile_size
        self._tiles_y = (self.height + self._tile_size - 1) // self._tile_size
        self._tile_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, int(self._tile_texture))
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_R32I,
            MAX_DELTA_VPLS,
            self._tiles_x * self._tiles_y,
            0,
            GL_RED_INTEGER,
            GL_INT,
            None,
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(GL_TEXTURE_2D, 0)
        texture_units = int(glGetIntegerv(0x8872))
        self._shadow_batch_size = min(
            SHADOW_BATCH_SIZE, max(1, texture_units - 4)
        )
        self._scheduler: VPLCubeScheduler[CubeResource] = VPLCubeScheduler()
        self._resources: dict[str, CubeResource] = {}
        self._vpls: list[VirtualPointLight] = []
        self._all_clustered_vpls: list[VirtualPointLight] = []
        self._pending_cache_sync = True
        self._pending_vpl_recluster = True
        self._changed_light_ids: set[str] = set()
        self._light_signatures = {
            light.id: self._light_signature(light) for light in config.lights
        }
        self._frame = 0
        self._force_full_rebuild = True
        self._last_entries: list[_Entry] = []
        self._camera_position = np.asarray(config.camera.get("position", (0, 1.6, 3.5)), dtype=np.float64)
        self._camera_target = np.asarray(config.camera.get("target", (0, 1.0, 0.0)), dtype=np.float64)
        self._camera_up = np.asarray(config.camera.get("up", (0, 1, 0)), dtype=np.float64)
        self._fov_y = float(config.camera.get("fov_y_deg", 58.0))
        self._near = float(config.camera.get("near", 0.05))
        self._far = float(config.camera.get("far", 25.0))
        self._stats = RenderStats()
        self._upload_material_uniforms(material_array_size)

    def _light_signature(self, light: Light) -> str:
        payload = light.to_dict()
        return repr(sorted(payload.items()))

    @property
    def view(self) -> np.ndarray:
        return look_at(self._camera_position, self._camera_target, self._camera_up)

    @property
    def projection(self) -> np.ndarray:
        return perspective(self._fov_y, self.width / self.height, self._near, self._far)

    @property
    def view_projection(self) -> np.ndarray:
        return self.projection @ self.view

    def _upload_material_uniforms(self, count: int) -> None:
        types = [0] * count
        albedo = np.zeros((count, 3), dtype=np.float32)
        roughness = [0.8] * count
        specular = [0.0] * count
        shininess = [32.0] * count
        for material in self.config.materials:
            index = material.id
            if index < 0 or index >= count:
                continue
            types[index] = material.shader_type
            albedo[index] = material.albedo
            roughness[index] = material.roughness
            specular[index] = material.specular
            shininess[index] = material.shininess
        self._vpl_shader.use()
        self._vpl_shader.set_int("uMaterialCount", count)
        self._vpl_shader.set_int_array("uMaterialType", types)
        self._vpl_shader.set_vec3_array("uMaterialAlbedo", albedo)
        self._vpl_shader.set_float_array("uMaterialRoughness", roughness)
        self._vpl_shader.set_float_array("uMaterialSpecular", specular)
        self._vpl_shader.set_float_array("uMaterialShininess", shininess)

    def set_lights(self, lights: Iterable[Light]) -> None:
        new_lights = sorted(list(lights), key=lambda light: light.id)
        if len(new_lights) > self.config.clustering.max_lights:
            raise ValueError("too many lights")
        new_signatures = {light.id: self._light_signature(light) for light in new_lights}
        changed = {
            light_id
            for light_id in set(new_signatures) | set(self._light_signatures)
            if new_signatures.get(light_id) != self._light_signatures.get(light_id)
        }
        self.config.lights = new_lights
        self._light_signatures = new_signatures
        if changed:
            self._changed_light_ids.update(changed)
            self._pending_vpl_recluster = True

    def update_light(self, light_id: str, **changes: object) -> None:
        lights: list[Light] = []
        found = False
        for light in self.config.lights:
            if light.id == light_id:
                payload = light.to_dict()
                payload.update(changes)
                lights.append(Light.from_dict(payload, 0))
                found = True
            else:
                lights.append(light)
        if not found:
            raise KeyError(f"unknown light {light_id}")
        self.set_lights(lights)

    def set_clustering(
        self,
        *,
        max_vpls: int | None = None,
        vpl_capacity: int | None = None,
        orientation_weight: float | None = None,
    ) -> None:
        clustering = self.config.clustering
        payload = clustering.to_dict()
        if max_vpls is not None:
            payload["max_vpls"] = int(max_vpls)
        if vpl_capacity is not None:
            payload["vpl_capacity"] = int(vpl_capacity)
        if orientation_weight is not None:
            payload["orientation_weight"] = float(orientation_weight)
        from .config import ClusteringConfig

        self.config.clustering = ClusteringConfig.from_dict(payload)
        self._pending_vpl_recluster = True
        self._pending_cache_sync = True
        self._force_full_rebuild = True
        self._all_clustered_vpls = []
        self._changed_light_ids = set()

    def set_camera(
        self,
        position: Sequence[float] | None = None,
        target: Sequence[float] | None = None,
    ) -> None:
        if position is not None:
            self._camera_position = np.asarray(position, dtype=np.float64)
        if target is not None:
            self._camera_target = np.asarray(target, dtype=np.float64)
        self._force_full_rebuild = True

    def _rebuild_vpls(self) -> None:
        previous = self._all_clustered_vpls if self._all_clustered_vpls else None
        forced = previous is None
        if forced:
            clustered = cluster_lights(self.config.lights, self.config.clustering)
        else:
            clustered = cluster_lights(
                self.config.lights,
                self.config.clustering,
                previous=previous,
                changed_ids=self._changed_light_ids,
            )
        self._all_clustered_vpls = clustered
        importance = rank_vpl_importance(
            clustered,
            capacity=self.config.clustering.vpl_capacity,
            view_proj=self.view_projection,
            camera_position=self._camera_position,
            scene_triangles=self.mesh.triangles(),
        )
        self._vpls = importance.accepted
        if importance.rejected:
            self.diagnostics.extend(
                [
                    truncation_warning(
                        len(clustered),
                        importance,
                        self.config.clustering.vpl_capacity,
                    )
                ]
            )
        self._pending_vpl_recluster = False
        self._pending_cache_sync = True
        self._changed_light_ids.clear()
        self.diagnostics.write(self.config.render.warning_path)

    def _create_cube(self, vpl: VirtualPointLight) -> CubeResource:
        cube = DepthCube(self.config.clustering.cube_size)
        matrices = light_view_projections(
            np.asarray(vpl.position),
            vpl.range,
            self.config.clustering.cube_size,
        )
        self._cube_shader.use()
        self._cube_shader.set_vec3("uLightPosition", vpl.position)
        self._cube_shader.set_float("uLightRange", vpl.range)
        glDisable(GL_BLEND)
        glDisable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        for face, matrix in enumerate(matrices):
            cube.bind_face(face)
            glClearColor(1.0, 1.0, 1.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self._cube_shader.set_mat4("uLightViewProjection", matrix)
            self._mesh_gpu.draw_groups()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return CubeResource(cube=cube)

    def _release_cube(self, resource: CubeResource) -> None:
        resource.cube.close()

    def _sync_resources(self) -> tuple[dict[str, CubeResource], list[CubeResource]]:
        old_resources = {resource_id: entry.resource for resource_id, entry in self._scheduler.entries.items()}
        result = self._scheduler.sync(self._vpls, self._create_cube)
        self._resources = {
            resource_id: entry.resource for resource_id, entry in result.entries.items()
        }
        self._stats.cube_builds = len(result.added) + len(result.changed)
        self._stats.cube_reused = len(result.reused)
        self._stats.cube_released = len(result.stale_resources)
        return old_resources, result.stale_resources

    def _render_gbuffer(self) -> int:
        self._gbuffer.bind()
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDisable(GL_CULL_FACE)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._gbuffer_shader.use()
        self._gbuffer_shader.set_mat4("uModel", np.eye(4, dtype=np.float32))
        self._gbuffer_shader.set_mat4("uViewProjection", self.view_projection)
        draw_calls = self._mesh_gpu.draw_groups()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        return draw_calls

    def _screen_tile_bounds(
        self,
        vpl: VirtualPointLight,
    ) -> tuple[int, int, int, int] | None:
        view = self.view
        projection = self.projection
        view_position = (view @ np.asarray((*vpl.position, 1.0), dtype=np.float64))
        view_z = -float(view_position[2])
        if view_z <= 1.0e-4:
            if np.linalg.norm(self._camera_position - np.asarray(vpl.position)) <= vpl.range:
                return 0, self._tiles_x - 1, 0, self._tiles_y - 1
            return None
        clip = transform_points(self.view_projection, np.asarray(vpl.position)[None, :])[0]
        if clip[3] <= 1.0e-5:
            return None
        ndc = clip[:3] / clip[3]
        radius_pixels = vpl.range * projection[1, 1] * self.height * 0.5 / view_z
        center_x = (ndc[0] * 0.5 + 0.5) * self.width
        center_y = (ndc[1] * 0.5 + 0.5) * self.height
        if (
            center_x + radius_pixels < 0.0
            or center_x - radius_pixels > self.width
            or center_y + radius_pixels < 0.0
            or center_y - radius_pixels > self.height
        ):
            return None
        margin = 8.0
        min_x = max(0, int(np.floor((center_x - radius_pixels - margin) / self._tile_size)))
        max_x = min(
            self._tiles_x - 1,
            int(np.floor((center_x + radius_pixels + margin) / self._tile_size)),
        )
        min_y = max(0, int(np.floor((center_y - radius_pixels - margin) / self._tile_size)))
        max_y = min(
            self._tiles_y - 1,
            int(np.floor((center_y + radius_pixels + margin) / self._tile_size)),
        )
        if min_x > max_x or min_y > max_y:
            return None
        return min_x, max_x, min_y, max_y

    def _build_tile_table(self, entries: Sequence[_Entry]) -> np.ndarray:
        width = MAX_DELTA_VPLS
        table = np.full((self._tiles_x * self._tiles_y, width), -1, dtype=np.int32)
        for entry_index, entry in enumerate(entries):
            bounds = self._screen_tile_bounds(entry.vpl)
            if bounds is None:
                continue
            min_x, max_x, min_y, max_y = bounds
            for tile_y in range(min_y, max_y + 1):
                row = tile_y * self._tiles_x
                for tile_x in range(min_x, max_x + 1):
                    row_values = table[row + tile_x]
                    free = int(np.argmax(row_values < 0))
                    if row_values[free] < 0:
                        row_values[free] = entry_index
        return table

    def _upload_entries(self, entries: Sequence[_Entry], table: np.ndarray) -> None:
        count = len(entries)
        positions = np.zeros((MAX_DELTA_VPLS, 3), dtype=np.float32)
        colors = np.zeros((MAX_DELTA_VPLS, 3), dtype=np.float32)
        directions = np.zeros((MAX_DELTA_VPLS, 3), dtype=np.float32)
        params = np.zeros((MAX_DELTA_VPLS, 4), dtype=np.float32)
        shadow_indices = [-1] * MAX_DELTA_VPLS
        for index, entry in enumerate(entries):
            positions[index] = entry.vpl.position
            colors[index] = np.asarray(entry.vpl.color) * entry.vpl.intensity
            directions[index] = entry.vpl.direction
            shadow_flag = 1.0
            if entry.resource is None:
                shadow_flag = 0.0
            params[index] = (
                entry.vpl.range,
                entry.vpl.cone_cos_outer,
                entry.sign,
                shadow_flag,
            )
            shadow_indices[index] = index
        self._vpl_shader.set_vec3_array("uEntryVplPosition", positions)
        self._vpl_shader.set_vec3_array("uEntryVplColorPower", colors)
        self._vpl_shader.set_vec3_array("uEntryVplDirection", directions)
        self._vpl_shader.set_vec4_array("uEntryVplParams", params)
        self._vpl_shader.set_int_array("uEntryShadowMapIndex", shadow_indices)
        self._vpl_shader.set_int("uEntryCount", count)
        self._vpl_shader.set_texture("uTileEntries", 2, int(self._tile_texture))
        glBindTexture(GL_TEXTURE_2D, int(self._tile_texture))
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            MAX_DELTA_VPLS,
            table.shape[0],
            GL_RED_INTEGER,
            GL_INT,
            np.ascontiguousarray(table),
        )
        for unit_offset, entry in enumerate(entries):
            unit = 3 + unit_offset
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_CUBE_MAP, entry.resource.cube.texture)
        # Dynamic sampler-array elements are initialized by index, not by an
        # array uniform call. Keep a single lookup per entry.
        from OpenGL.GL import glUniform1i

        for index in range(count):
            glUniform1i(self._vpl_shader.location(f"uShadowMaps[{index}]"), 3 + index)

    def _accumulate(self, entries: Sequence[_Entry], *, clear: bool) -> int:
        if not entries:
            if clear:
                self._hdr.bind()
                glClearColor(0.0, 0.0, 0.0, 1.0)
                glClear(GL_COLOR_BUFFER_BIT)
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
            return 0
        self._hdr.bind()
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendEquation(GL_FUNC_ADD)
        glBlendFunc(GL_ONE, GL_ONE)
        if clear:
            glClearColor(0.0, 0.0, 0.0, 1.0)
            glClear(GL_COLOR_BUFFER_BIT)
        self._vpl_shader.use()
        self._vpl_shader.set_texture("uPositionTexture", 0, self._gbuffer.position)
        self._vpl_shader.set_texture("uNormalMaterialTexture", 1, self._gbuffer.normal_material)
        self._vpl_shader.set_vec2("uScreenSize", (self.width, self.height))
        self._vpl_shader.set_int("uTileSize", self._tile_size)
        self._vpl_shader.set_int("uTilesX", self._tiles_x)
        self._vpl_shader.set_mat4("uView", self.view)
        self._vpl_shader.set_vec3("uCameraPosition", self._camera_position)
        self._upload_material_uniforms(max(8, self._max_material_slot + 1))
        batches = 0
        for offset in range(0, len(entries), self._shadow_batch_size):
            batch = list(entries[offset : offset + self._shadow_batch_size])
            table = self._build_tile_table(batch)
            self._upload_entries(batch, table)
            self._fullscreen.draw()
            batches += 1
        glDisable(GL_BLEND)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self._stats.accumulation_batches = batches
        return batches

    def _entries_for_full(self) -> list[_Entry]:
        return [
            _Entry(1.0, vpl, self._resources[vpl.id])
            for vpl in self._vpls
            if vpl.id in self._resources
        ]

    def _entries_for_delta(
        self,
        old_vpls: Sequence[VirtualPointLight],
        old_resources: dict[str, CubeResource],
    ) -> list[_Entry]:
        old_by_id = {vpl.id: vpl for vpl in old_vpls}
        new_by_id = {vpl.id: vpl for vpl in self._vpls}
        entries: list[_Entry] = []
        for vpl in old_vpls:
            new = new_by_id.get(vpl.id)
            if new is not None and new.signature == vpl.signature:
                continue
            resource = old_resources.get(vpl.id)
            if resource is not None:
                entries.append(_Entry(-1.0, vpl, resource))
        for vpl in self._vpls:
            old = old_by_id.get(vpl.id)
            if old is not None and old.signature == vpl.signature:
                continue
            resource = self._resources.get(vpl.id)
            if resource is not None:
                entries.append(_Entry(1.0, vpl, resource))
        return entries

    def render(self, *, rebuild: bool = False) -> RenderStats:
        self._stats = RenderStats(frame=self._frame)
        old_vpls = list(self._vpls)
        old_resources: dict[str, CubeResource] = dict(self._resources)
        if self._pending_vpl_recluster or not self._vpls:
            self._rebuild_vpls()
        stale_resources: list[CubeResource] = []
        if self._pending_cache_sync:
            old_resources, stale_resources = self._sync_resources()
            self._pending_cache_sync = False
        self._render_gbuffer()
        full_rebuild = self._force_full_rebuild or rebuild or self._frame == 0
        if full_rebuild:
            entries = self._entries_for_full()
        else:
            entries = self._entries_for_delta(old_vpls, old_resources)
        self._accumulate(entries, clear=full_rebuild)
        released: set[int] = set()
        for resource in stale_resources:
            if id(resource) in released:
                continue
            if all(resource is not active for active in self._resources.values()):
                self._release_cube(resource)
                released.add(id(resource))
        self._stats.frame = self._frame
        self._stats.cluster_count = len(self._all_clustered_vpls)
        self._stats.active_vpl_count = len(self._vpls)
        self._stats.delta_entries = len(entries)
        self._stats.subtract_entries = sum(1 for entry in entries if entry.sign < 0.0)
        self._stats.full_rebuild = full_rebuild
        self._stats.material_draw_calls = len(self.mesh.groups)
        self._stats.truncated_vpl_count = max(
            0, len(self._all_clustered_vpls) - len(self._vpls)
        )
        self._last_entries = entries
        self._force_full_rebuild = False
        self._frame += 1
        return self._stats

    def present(self, *, tonemap: bool = True) -> None:
        self._final.target.bind()
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        self._post_shader.use()
        self._post_shader.set_texture("uHdrTexture", 0, self._hdr.texture)
        self._post_shader.set_float("uExposure", self.config.render.exposure)
        self._post_shader.set_int("uTonemap", 1 if tonemap else 0)
        self._fullscreen.draw()
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def blit_to_default_framebuffer(self) -> None:
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self._final.target.fbo)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(
            0,
            0,
            self.width,
            self.height,
            0,
            0,
            self.width,
            self.height,
            0x4000,
            0x2600,
        )
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)

    def save_screenshot(self, path: str | Path | None = None, *, tonemap: bool = True) -> Path:
        self.present(tonemap=tonemap)
        target = Path(path or (
            self.config.render.screenshot_path if tonemap else self.config.render.debug_screenshot_path
        ))
        self._final.save_png(target)
        return target

    def read_hdr(self) -> np.ndarray:
        return self._hdr.read_rgb()

    def close(self) -> None:
        try:
            self._scheduler.clear(self._release_cube)
        finally:
            self._mesh_gpu.close()
            self._gbuffer.close()
            self._hdr.close()
            self._final.close()
            self._fullscreen.close()
            self._gbuffer_shader.close()
            self._cube_shader.close()
            self._vpl_shader.close()
            self._post_shader.close()
            from OpenGL.GL import glDeleteTextures

            glDeleteTextures(1, [int(self._tile_texture)])

