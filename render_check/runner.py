"""Validation run orchestration and multi-channel rendering."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import moderngl
import numpy as np

from . import artifacts
from .batch import (
    CAMERA_FAR,
    CAMERA_FOV_Y,
    CAMERA_NEAR,
    CAMERA_TARGET,
    CAMERA_UP,
    InstanceSpec,
    Scene,
    build_scene,
)
from .context import create_headless_context
from .errors import ValidationError, diff_error, gl_context_error, texture_error
from .geometry import MeshData
from .materials import Material, TextureData, load_materials
from .perf import PerformanceMonitor
from .pipeline import compile_material_programs
from .state import (
    apply_expected_uniforms,
    make_snapshot,
    public_snapshot,
    verify_snapshot,
)
from .textures import load_texture


DETERMINISM_TOLERANCE = 1e-6
CHANNELS = ("color", "normal", "depth", "object_id")
GL_DITHER = 0x0BD0


@dataclass(frozen=True)
class RunConfig:
    materials_path: Path
    output_dir: Path
    instances: int
    seed: int
    width: int
    height: int
    backend: str = "auto"
    fault_inject: str = "none"
    memory_limit_mb: float = 512.0
    time_limit_sec: float = 30.0


@dataclass
class RenderRow:
    instance_index: int
    material_name: str
    color_file: str
    normal_file: str
    depth_file: str
    object_id_file: str
    render_time_ms: float
    state_hash: str

    def as_csv_row(self) -> dict[str, Any]:
        return {
            "instance_index": self.instance_index,
            "material_name": self.material_name,
            "color_file": self.color_file,
            "normal_file": self.normal_file,
            "depth_file": self.depth_file,
            "object_id_file": self.object_id_file,
            "render_time_ms": f"{self.render_time_ms:.6f}",
            "state_hash": self.state_hash,
        }


@dataclass
class RunState:
    config: RunConfig
    tracker: PerformanceMonitor
    materials: list[Material] = field(default_factory=list)
    textures: dict[str, TextureData] = field(default_factory=dict)
    context_info: dict[str, Any] = field(default_factory=dict)
    scene: Scene | None = None
    rows: list[RenderRow] = field(default_factory=list)
    state_snapshots: list[dict[str, Any]] = field(default_factory=list)
    successful_instances: int = 0
    max_abs_diff: float = 0.0
    determinism_checks: int = 0
    error: ValidationError | None = None


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "material"


def _read_channel(
    framebuffer: moderngl.Framebuffer,
    attachment: int,
    width: int,
    height: int,
) -> np.ndarray:
    raw = framebuffer.read(attachment=attachment, components=4, dtype="f4")
    array = np.frombuffer(raw, dtype="<f4").reshape((height, width, 4)).copy()
    return np.ascontiguousarray(array[::-1, :, :])


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
    if not np.isfinite(difference).all():
        return float("inf")
    return float(difference.max(initial=0.0))


class RenderValidator:
    def __init__(self, config: RunConfig, tracker: PerformanceMonitor) -> None:
        self.state = RunState(config=config, tracker=tracker)

    def _load_inputs(self) -> None:
        self.state.materials = load_materials(self.state.config.materials_path)
        self.state.tracker.check()
        for material in self.state.materials:
            self.state.textures[material.name] = load_texture(material)
        self.state.tracker.check()

    def _create_texture(
        self,
        ctx: moderngl.Context,
        material: Material,
        data: TextureData,
    ) -> moderngl.Texture:
        dtype = "u1" if data.actual_format == "RGBA8" else "f4"
        try:
            texture = ctx.texture(
                (data.width, data.height),
                4,
                data.array.tobytes(),
                dtype=dtype,
            )
        except Exception as exc:
            raise texture_error(
                f"material '{material.name}': cannot upload texture: {exc}",
                material=material.name,
                expected=data.declared_format,
                actual=data.actual_description,
            ) from exc
        if texture.dtype != dtype or texture.components != 4:
            actual = f"dtype={texture.dtype}, components={texture.components}"
            texture.release()
            raise texture_error(
                f"material '{material.name}': GL texture creation produced "
                f"an unexpected format ({actual})",
                material=material.name,
                expected=data.declared_format,
                actual=actual,
            )
        texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        texture.repeat_x = True
        texture.repeat_y = True
        texture.use(material.texture.unit)
        return texture

    def _create_vaos(
        self,
        ctx: moderngl.Context,
        programs: dict[str, moderngl.Program],
        meshes: dict[str, MeshData],
    ) -> tuple[
        dict[tuple[str, str], moderngl.VertexArray],
        list[moderngl.Buffer],
    ]:
        buffers: list[moderngl.Buffer] = []
        vaos: dict[tuple[str, str], moderngl.VertexArray] = {}
        mesh_buffers: dict[str, tuple[moderngl.Buffer, moderngl.Buffer]] = {}
        for name, mesh in meshes.items():
            vertex_buffer = ctx.buffer(mesh.vertices.tobytes())
            index_buffer = ctx.buffer(mesh.indices.tobytes())
            mesh_buffers[name] = (vertex_buffer, index_buffer)
            buffers.extend((vertex_buffer, index_buffer))
        for material in self.state.materials:
            program = programs[material.name]
            for shape, (vertex_buffer, index_buffer) in mesh_buffers.items():
                vao = ctx.vertex_array(
                    program,
                    [(vertex_buffer, "3f 3f 2f", "in_position", "in_normal", "in_uv")],
                    index_buffer,
                )
                vaos[(material.name, shape)] = vao
        return vaos, buffers

    def _make_framebuffer(
        self,
        ctx: moderngl.Context,
        width: int,
        height: int,
    ) -> tuple[moderngl.Framebuffer, list[moderngl.Texture]]:
        channel_textures: list[moderngl.Texture] = []
        try:
            for _ in CHANNELS:
                texture = ctx.texture((width, height), 4, dtype="f4")
                if (
                    texture.dtype != "f4"
                    or texture.components != 4
                    or tuple(texture.size) != (width, height)
                ):
                    raise gl_context_error(
                        "driver did not create the required RGBA32F render attachment",
                        width=width,
                        height=height,
                        actual_dtype=str(texture.dtype),
                        actual_components=int(texture.components),
                    )
                texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
                channel_textures.append(texture)
        except Exception:
            for texture in channel_textures:
                texture.release()
            raise
        depth_texture: moderngl.Texture | None = None
        framebuffer: moderngl.Framebuffer | None = None
        try:
            depth_texture = ctx.depth_texture((width, height))
            channel_textures.append(depth_texture)
            framebuffer = ctx.framebuffer(
                color_attachments=channel_textures[:4],
                depth_attachment=depth_texture,
            )
            if len(framebuffer.color_attachments) != len(CHANNELS):
                raise gl_context_error(
                    "framebuffer does not expose four color attachments",
                    attachments=len(framebuffer.color_attachments),
                )
            return framebuffer, channel_textures
        except Exception:
            if framebuffer is not None:
                framebuffer.release()
            for texture in channel_textures:
                texture.release()
            raise

    def _render_pass(
        self,
        ctx: moderngl.Context,
        framebuffer: moderngl.Framebuffer,
        program: moderngl.Program,
        vao: moderngl.VertexArray,
        texture: moderngl.Texture,
        material: Material,
        instance: InstanceSpec,
        scene: Scene,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        config = self.state.config
        expected = apply_expected_uniforms(program, material, instance, scene)

        residue_target = 1 if config.instances > 1 else 0
        if config.fault_inject == "state_residue" and instance.index == residue_target:
            program["u_metallic"].value = float(program["u_metallic"].value) + 0.25

        texture.use(material.texture.unit)
        framebuffer.use()
        ctx.viewport = (0, 0, config.width, config.height)
        ctx.enable_only(moderngl.DEPTH_TEST)
        ctx.disable_direct(GL_DITHER)
        ctx.depth_func = "<"
        framebuffer.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        vao.render(moderngl.TRIANGLES)
        channels = {
            name: _read_channel(framebuffer, index, config.width, config.height)
            for index, name in enumerate(CHANNELS)
        }
        snapshot = make_snapshot(program, expected, instance, material)
        return channels, snapshot

    def _render_instance(
        self,
        ctx: moderngl.Context,
        instance: InstanceSpec,
        material: Material,
        program: moderngl.Program,
        vao: moderngl.VertexArray,
        texture: moderngl.Texture,
        scene: Scene,
    ) -> None:
        config = self.state.config
        framebuffer, attachments = self._make_framebuffer(ctx, config.width, config.height)
        try:
            started = time.perf_counter()
            first, snapshot = self._render_pass(
                ctx,
                framebuffer,
                program,
                vao,
                texture,
                material,
                instance,
                scene,
            )
            render_time_ms = (time.perf_counter() - started) * 1000.0
            self.state.tracker.check()

            self.state.state_snapshots.append(public_snapshot(snapshot))
            verify_snapshot(snapshot)
            self.state.tracker.check()

            second, second_snapshot = self._render_pass(
                ctx,
                framebuffer,
                program,
                vao,
                texture,
                material,
                instance,
                scene,
            )
            verify_snapshot(second_snapshot)
            self.state.tracker.check()

            if config.fault_inject == "diff" and instance.index == 0:
                second["color"] = second["color"].copy()
                second["color"][4, 4, 0] += 0.01

            max_diff = 0.0
            for channel in CHANNELS:
                max_diff = max(max_diff, _max_abs_difference(first[channel], second[channel]))
            self.state.max_abs_diff = max(self.state.max_abs_diff, max_diff)
            self.state.determinism_checks += 1
            if max_diff > DETERMINISM_TOLERANCE:
                raise diff_error(
                    f"instance {instance.index} material '{material.name}': determinism "
                    f"difference {max_diff:.9g} exceeds {DETERMINISM_TOLERANCE:g}",
                    instance_index=instance.index,
                    material=material.name,
                    max_abs_diff=max_diff,
                    tolerance=DETERMINISM_TOLERANCE,
                )

            slug = _slug(material.name)
            filenames = {
                channel: artifacts.channel_filename(instance.index, slug, channel)
                for channel in CHANNELS
            }
            for channel in CHANNELS:
                artifacts.write_npy(
                    config.output_dir / filenames[channel],
                    first[channel],
                )
            self.state.rows.append(
                RenderRow(
                    instance_index=instance.index,
                    material_name=material.name,
                    color_file=filenames["color"],
                    normal_file=filenames["normal"],
                    depth_file=filenames["depth"],
                    object_id_file=filenames["object_id"],
                    render_time_ms=render_time_ms,
                    state_hash=snapshot["state_hash"],
                )
            )
            self.state.successful_instances += 1
            self.state.tracker.check()
        finally:
            framebuffer.release()
            for attachment in attachments:
                attachment.release()

    def run(self) -> RunState:
        config = self.state.config
        artifacts.prepare_output_dir(config.output_dir)
        self.state.tracker.check()
        self._load_inputs()

        ctx: moderngl.Context | None = None
        programs: dict[str, moderngl.Program] = {}
        texture_objects: dict[str, moderngl.Texture] = {}
        vaos: dict[tuple[str, str], moderngl.VertexArray] = {}
        buffers: list[moderngl.Buffer] = []
        try:
            ctx, context_info = create_headless_context(config.backend)
            self.state.context_info = context_info
            programs = compile_material_programs(ctx, self.state.materials)
            self.state.tracker.check()

            scene = build_scene(
                self.state.materials,
                config.instances,
                config.seed,
                config.width,
                config.height,
            )
            self.state.scene = scene
            for material in self.state.materials:
                texture_objects[material.name] = self._create_texture(
                    ctx, material, self.state.textures[material.name]
                )
            vaos, buffers = self._create_vaos(ctx, programs, scene.meshes)
            self.state.tracker.check()

            materials_by_name = {material.name: material for material in self.state.materials}
            for instance in scene.instances:
                self.state.tracker.check()
                material = materials_by_name[instance.material_name]
                self._render_instance(
                    ctx,
                    instance,
                    material,
                    programs[material.name],
                    vaos[(material.name, instance.shape)],
                    texture_objects[material.name],
                    scene,
                )
            self.state.tracker.check()
        finally:
            for vao in vaos.values():
                vao.release()
            for buffer in buffers:
                buffer.release()
            for texture in texture_objects.values():
                texture.release()
            for program in programs.values():
                program.release()
            if ctx is not None:
                ctx.release()
        return self.state


def _material_summary(state: RunState) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for material in state.materials:
        texture = state.textures.get(material.name)
        result.append(
            {
                "name": material.name,
                "description": material.description,
                "vertex_shader": str(material.vertex_shader),
                "fragment_shader": str(material.fragment_shader),
                "texture": {
                    "path": str(material.texture.path),
                    "declared_format": material.texture.format,
                    "actual_format": texture.actual_format if texture else None,
                    "actual_description": texture.actual_description if texture else None,
                    "width": texture.width if texture else None,
                    "height": texture.height if texture else None,
                },
            }
        )
    return result


def channel_map(state: RunState) -> list[dict[str, Any]]:
    mapping: list[dict[str, Any]] = []
    for row in state.rows:
        mapping.append(
            {
                "instance_index": row.instance_index,
                "material": row.material_name,
                "object_id": row.instance_index + 1,
                "channels": {
                    "color": row.color_file,
                    "normal": row.normal_file,
                    "depth": row.depth_file,
                    "object_id": row.object_id_file,
                },
                "state_hash": row.state_hash,
            }
        )
    return mapping


def finalize(state: RunState, error: ValidationError | None) -> dict[str, Any]:
    config = state.config
    output_dir = config.output_dir
    if error is not None:
        state.error = error
    if state.error is None:
        state.tracker.check()

    try:
        artifacts.write_csv(
            output_dir / "render_log.csv",
            (row.as_csv_row() for row in state.rows),
        )
        map_payload = channel_map(state)
        artifacts.write_json(output_dir / "channel_map.json", map_payload)
        artifacts.write_json_lines(
            output_dir / "state_snapshots.jsonl",
            state.state_snapshots,
        )

        failed_instances = (
            config.instances - state.successful_instances if state.error else 0
        )
        summary = {
            "status": "ok" if state.error is None else "error",
            "requested_instances": config.instances,
            "successful_instances": state.successful_instances,
            "failed_instances": failed_instances,
            "peak_memory_bytes": state.tracker.peak_memory_bytes,
            "peak_memory_mb": round(
                state.tracker.peak_memory_bytes / (1024.0 * 1024.0), 3
            ),
            "elapsed_seconds": round(state.tracker.elapsed_seconds(), 6),
            "memory_limit_bytes": state.tracker.memory_limit_bytes,
            "time_limit_seconds": state.tracker.time_limit_sec,
            "seed": config.seed,
            "resolution": [config.width, config.height],
            "channel_format": "GL_RGBA32F",
            "channel_shape": [config.height, config.width, 4],
            "channels": list(CHANNELS),
            "backend_requested": config.backend,
            "context": state.context_info,
            "scene": {
                "camera_eye": (
                    list(state.scene.camera_position) if state.scene is not None else None
                ),
                "camera_target": list(CAMERA_TARGET),
                "camera_up": list(CAMERA_UP),
                "camera_fov_y_degrees": CAMERA_FOV_Y,
                "camera_near": CAMERA_NEAR,
                "camera_far": CAMERA_FAR,
                "light_position": (
                    list(state.scene.light_position) if state.scene is not None else None
                ),
                "light_color": (
                    list(state.scene.light_color) if state.scene is not None else None
                ),
                "material_cycle": [material.name for material in state.materials],
                "geometry_cycle": "cube if (instance_index + seed) % 2 == 0 else sphere",
            },
            "materials_path": str(config.materials_path),
            "output_dir": str(output_dir),
            "render_log_csv": "render_log.csv",
            "channel_map_json": "channel_map.json",
            "state_snapshots_jsonl": "state_snapshots.jsonl",
            "channel_map": map_payload,
            "materials": _material_summary(state),
            "determinism": {
                "passes_per_instance": 2,
                "tolerance": DETERMINISM_TOLERANCE,
                "checked_instances": state.determinism_checks,
                "max_abs_diff": state.max_abs_diff,
            },
            "fault_inject": config.fault_inject,
            "error": state.error.as_dict() if state.error else None,
        }
        artifacts.write_json(output_dir / "summary.json", summary)
    except ValidationError as exc:
        if state.error is None:
            state.error = exc
        summary = {
            "status": "error",
            "requested_instances": config.instances,
            "successful_instances": state.successful_instances,
            "failed_instances": config.instances - state.successful_instances,
            "peak_memory_bytes": state.tracker.peak_memory_bytes,
            "elapsed_seconds": round(state.tracker.elapsed_seconds(), 6),
            "error": state.error.as_dict(),
        }
        try:
            artifacts.write_json(output_dir / "summary.json", summary)
        except ValidationError:
            pass
        raise
    return summary

