from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import moderngl

from .compare import analyze_gbuffer, compare_images
from .diagnostics import Diagnostic
from .gl_context import ContextInfo, create_headless_context
from .manifest import (
    BLEND_EQUATIONS,
    BLEND_FACTORS,
    DEPTH_FUNCS,
    MaterialSpec,
    ObjectSpec,
    RenderSettings,
    SceneSpec,
    UniformSpec,
)
from .samples import (
    COVERAGE_FRAGMENT,
    DEFERRED_FRAGMENT,
    FULLSCREEN_VERTEX,
)


COVERAGE_VERTEX = """\
#version 330 core
in vec3 in_position;
uniform mat4 u_model;
uniform mat4 u_view_proj;
void main() {
    gl_Position = u_view_proj * u_model * vec4(in_position, 1.0);
}
"""


@dataclass
class RenderSceneResult:
    scene: str
    width: int
    height: int
    buffers: dict[str, np.ndarray] = field(default_factory=dict)
    comparison: dict[str, Any] | None = None
    gbuffer: dict[str, Any] | None = None
    depth_prepass_mismatch_pixels: int = 0
    diagnostics: list[Diagnostic] = field(default_factory=list)
    context: ContextInfo | None = None

    @property
    def status(self) -> str:
        return "error" if any(item.severity == "error" for item in self.diagnostics) else "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "status": self.status,
            "width": self.width,
            "height": self.height,
            "context": self.context.to_dict() if self.context else None,
            "gbuffer": self.gbuffer,
            "depth_prepass_mismatch_pixels": self.depth_prepass_mismatch_pixels,
            "comparison": self.comparison,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@dataclass
class MaterialRuntime:
    spec: MaterialSpec
    program: Any


def _normalized(values: Iterable[float]) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length < 1.0e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return vector / length


def _look_at(eye: list[float], target: list[float], up: list[float]) -> np.ndarray:
    eye_array = np.asarray(eye, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    up_array = np.asarray(up, dtype=np.float64)
    forward = _normalized(target_array - eye_array)
    right = _normalized(np.cross(forward, up_array))
    corrected_up = np.cross(right, forward)
    matrix = np.identity(4, dtype=np.float64)
    matrix[0, :3] = right
    matrix[1, :3] = corrected_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -float(np.dot(right, eye_array))
    matrix[1, 3] = -float(np.dot(corrected_up, eye_array))
    matrix[2, 3] = float(np.dot(forward, eye_array))
    return matrix


def _perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    focal = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = focal / aspect
    matrix[1, 1] = focal
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def _rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, cosine, -sine, 0.0], [0.0, sine, cosine, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [[cosine, 0.0, sine, 0.0], [0.0, 1.0, 0.0, 0.0], [-sine, 0.0, cosine, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rotation_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0, 0.0], [sine, cosine, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _model_matrix(obj: ObjectSpec) -> np.ndarray:
    translation = np.identity(4, dtype=np.float64)
    translation[:3, 3] = obj.position
    scale = np.diag([obj.scale[0], obj.scale[1], obj.scale[2], 1.0])
    rotation = _rotation_z(obj.rotation[2]) @ _rotation_y(obj.rotation[1]) @ _rotation_x(obj.rotation[0])
    return translation @ rotation @ scale


def _mesh_data(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "cube":
        faces = (
            ((0.0, 0.0, 1.0), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)),
            ((0.0, 0.0, -1.0), (0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5)),
            ((1.0, 0.0, 0.0), (0.5, -0.5, 0.5), (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)),
            ((-1.0, 0.0, 0.0), (-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5)),
            ((0.0, 1.0, 0.0), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5), (-0.5, 0.5, -0.5)),
            ((0.0, -1.0, 0.0), (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5)),
        )
        vertices: list[float] = []
        indices: list[int] = []
        for normal, a, b, c, d in faces:
            base = len(vertices) // 6
            for position in (a, b, c, d):
                vertices.extend((*position, *normal))
            indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
        return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.int32)
    normal = (0.0, 0.0, 1.0) if name == "plane" else (0.0, 0.0, 0.0)
    vertices = np.asarray(
        [
            -0.5, -0.5, 0.0, *normal,
            0.5, -0.5, 0.0, *normal,
            0.5, 0.5, 0.0, *normal,
            -0.5, 0.5, 0.0, *normal,
        ],
        dtype=np.float32,
    )
    indices = np.asarray((0, 1, 2, 0, 2, 3), dtype=np.int32)
    return vertices, indices


def _read_texture(texture: Any, width: int, height: int, components: int) -> np.ndarray:
    data = np.frombuffer(texture.read(), dtype=np.float32)
    return data.reshape(height, width, components).copy()


def _member(program: Any, name: str) -> Any | None:
    try:
        return program[name]
    except KeyError:
        return None


def _write_matrix(program: Any, name: str, matrix: Any) -> None:
    member = _member(program, name)
    if member is None:
        return
    array = np.asarray(matrix, dtype=np.float32)
    member.write(np.ascontiguousarray(array.T).tobytes())


def _bind_manifest_uniform(program: Any, spec: UniformSpec) -> None:
    member = _member(program, spec.name)
    if member is None:
        return
    if spec.type.startswith("mat"):
        _write_matrix(program, spec.name, spec.value)
        return
    if spec.type == "bool":
        member.value = int(bool(spec.value))
    elif spec.type.startswith("bvec"):
        member.value = tuple(int(bool(item)) for item in spec.value)
    elif spec.type in {"float", "int", "uint"}:
        member.value = spec.value
    else:
        member.value = tuple(spec.value)


def _set_array_uniform(
    program: Any,
    name: str,
    values: list[Any],
    scalar: bool,
) -> None:
    member = _member(program, name)
    if member is None:
        return
    count = max(1, int(getattr(member, "array_length", 1)))
    prepared = values[:count]
    if count == 1:
        member.value = prepared[0]
    else:
        member.value = tuple(prepared)


def _scene_light_uniforms(scene: SceneSpec) -> dict[str, Any]:
    positions: list[tuple[float, float, float]] = []
    colors: list[tuple[float, float, float]] = []
    intensities: list[float] = []
    radii: list[float] = []
    enabled = [light for light in scene.lights if light.enabled][:4]
    for item in enabled:
        if item.type == "directional":
            direction = _normalized(item.direction)
            position = tuple((-direction * 1000.0).tolist())
            radius = 1.0e6
        else:
            position = tuple(float(value) for value in item.position)
            radius = float(item.radius)
        positions.append(position)
        colors.append(tuple(float(value) for value in item.color))
        intensities.append(float(item.intensity))
        radii.append(radius)
    while len(positions) < 4:
        positions.append((0.0, 0.0, 0.0))
        colors.append((0.0, 0.0, 0.0))
        intensities.append(0.0)
        radii.append(1.0)
    return {
        "u_light_count": len(enabled),
        "u_light_position": positions,
        "u_light_color": colors,
        "u_light_intensity": intensities,
        "u_light_radius": radii,
    }


def _bind_global_uniforms(
    program: Any,
    scene: SceneSpec,
    camera_position: list[float],
    normal_fallback: list[float],
    background: list[float],
) -> None:
    member = _member(program, "u_camera_position")
    if member is not None:
        member.value = tuple(camera_position)
    member = _member(program, "u_ambient_color")
    if member is not None:
        member.value = (0.032, 0.038, 0.05)
    member = _member(program, "u_normal_fallback")
    if member is not None:
        member.value = tuple(normal_fallback)
    member = _member(program, "u_background_color")
    if member is not None:
        member.value = tuple(background)
    lights = _scene_light_uniforms(scene)
    member = _member(program, "u_light_count")
    if member is not None:
        member.value = lights["u_light_count"]
    for name, scalar in (
        ("u_light_intensity", True),
        ("u_light_radius", True),
        ("u_light_position", False),
        ("u_light_color", False),
    ):
        _set_array_uniform(program, name, lights[name], scalar)


def _bind_object_uniforms(program: Any, view_proj: np.ndarray, model: np.ndarray) -> None:
    _write_matrix(program, "u_view_proj", view_proj)
    _write_matrix(program, "u_model", model)
    normal_matrix = np.linalg.inv(model[:3, :3]).T
    _write_matrix(program, "u_normal_matrix", normal_matrix)


def _apply_depth(context: Any, material: MaterialSpec) -> None:
    import moderngl

    if material.depth.test:
        context.enable(moderngl.DEPTH_TEST)
    else:
        context.disable(moderngl.DEPTH_TEST)
    context.depth_func = DEPTH_FUNCS[material.depth.func]
    context.depth_mask = material.depth.write


def _apply_blend(context: Any, material: MaterialSpec, enabled: bool) -> None:
    import moderngl

    if not enabled or not material.blend.enabled:
        context.disable(moderngl.BLEND)
        return
    context.enable(moderngl.BLEND)
    context.blend_func = (
        BLEND_FACTORS[material.blend.src_factor],
        BLEND_FACTORS[material.blend.dst_factor],
    )
    context.blend_equation = BLEND_EQUATIONS[material.blend.equation]


def _prepare_runtime_source(source: str) -> str:
    if not source.lstrip().startswith("#version"):
        return "#version 330 core\n" + source
    return source


class SceneRenderer:
    def __init__(
        self,
        context: Any,
        materials: dict[str, MaterialSpec],
        scene: SceneSpec,
        width: int,
        height: int,
        normal_fallback: list[float],
        normal_tolerance: float,
        background: list[float],
    ) -> None:
        self.context = context
        self.materials = materials
        self.scene = scene
        self.width = width
        self.height = height
        self.normal_fallback = normal_fallback
        self.normal_tolerance = normal_tolerance
        self.background = background
        self.view = _look_at(scene.camera.position, scene.camera.target, scene.camera.up)
        self.projection = _perspective(
            scene.camera.fov_y_deg,
            width / height,
            scene.camera.near,
            scene.camera.far,
        )
        self.view_proj = self.projection @ self.view
        self.runtime: dict[str, MaterialRuntime] = {}
        self.meshes: dict[str, tuple[Any, Any]] = {}
        self.coverage_buffers: dict[str, Any] = {}
        self.vaos: dict[tuple[str, str], Any] = {}
        self.buffers: list[Any] = []
        self.textures: list[Any] = []
        self.framebuffers: list[Any] = []
        self.objects = [item for item in scene.objects if item.material in materials]
        self.context.disable(0x0BD0)
        self.context.disable(moderngl.CULL_FACE)
        self.context.disable(moderngl.BLEND)
        self.context.disable(0x809D)
        self._build_material_programs()
        self._build_meshes()
        self._build_internal_programs()
        self._validate_builtin_uniforms()

    def _build_material_programs(self) -> None:
        for name, material in self.materials.items():
            vertex = _prepare_runtime_source(material.vertex_path.read_text(encoding="utf-8"))
            fragment = _prepare_runtime_source(material.fragment_path.read_text(encoding="utf-8"))
            program = self.context.program(vertex_shader=vertex, fragment_shader=fragment)
            self.runtime[name] = MaterialRuntime(spec=material, program=program)

    def _build_meshes(self) -> None:
        for mesh_name in {item.mesh for item in self.objects}:
            vertices, indices = _mesh_data(mesh_name)
            vbo = self.context.buffer(vertices.tobytes())
            ibo = self.context.buffer(indices.tobytes())
            positions = self.context.buffer(np.ascontiguousarray(vertices.reshape(-1, 6)[:, :3]).tobytes())
            self.buffers.extend((vbo, ibo, positions))
            self.meshes[mesh_name] = (vbo, ibo)
            self.coverage_buffers[mesh_name] = positions

    def _build_internal_programs(self) -> None:
        self.deferred_program = self.context.program(
            vertex_shader=FULLSCREEN_VERTEX,
            fragment_shader=DEFERRED_FRAGMENT,
        )
        self.coverage_program = self.context.program(
            vertex_shader=COVERAGE_VERTEX,
            fragment_shader=COVERAGE_FRAGMENT,
        )
        self.fullscreen_vao = self.context.vertex_array(self.deferred_program, [])

    def _validate_builtin_uniforms(self) -> None:
        required = (
            "u_gbuffer_position",
            "u_gbuffer_normal",
            "u_gbuffer_albedo",
            "u_gbuffer_metallic_roughness",
            "u_background_color",
            "u_camera_position",
            "u_ambient_color",
            "u_normal_fallback",
            "u_light_count",
            "u_light_position",
            "u_light_color",
            "u_light_intensity",
            "u_light_radius",
        )
        for name in required:
            if _member(self.deferred_program, name) is None:
                raise RuntimeError(f"Internal deferred shader is missing uniform {name}")

    def _vao(self, material_name: str, mesh_name: str) -> Any:
        key = (material_name, mesh_name)
        if key in self.vaos:
            return self.vaos[key]
        program = self.runtime[material_name].program
        vbo, ibo = self.meshes[mesh_name]
        vao = self.context.vertex_array(
            program,
            [(vbo, "3f 3f", "in_position", "in_normal")],
            ibo,
            index_element_size=4,
        )
        self.vaos[key] = vao
        return vao

    def _coverage_vao(self, mesh_name: str) -> Any:
        key = ("__coverage__", mesh_name)
        if key in self.vaos:
            return self.vaos[key]
        positions = self.coverage_buffers[mesh_name]
        _, ibo = self.meshes[mesh_name]
        vao = self.context.vertex_array(
            self.coverage_program,
            [(positions, "3f", "in_position")],
            ibo,
            index_element_size=4,
        )
        self.vaos[key] = vao
        return vao

    def _alloc_color(self, components: int = 4) -> Any:
        texture = self.context.texture((self.width, self.height), components, dtype="f4")
        texture.filter = (0x2600, 0x2600)
        texture.repeat_x = False
        texture.repeat_y = False
        self.textures.append(texture)
        return texture

    def _alloc_depth(self) -> Any:
        texture = self.context.depth_texture((self.width, self.height))
        self.textures.append(texture)
        return texture

    def _alloc_framebuffer(self, colors: list[Any], depth: Any | None = None) -> Any:
        framebuffer = self.context.framebuffer(color_attachments=colors, depth_attachment=depth)
        self.framebuffers.append(framebuffer)
        return framebuffer

    def _bind_object(self, runtime: MaterialRuntime, obj: ObjectSpec) -> Any:
        program = runtime.program
        _bind_global_uniforms(
            program,
            self.scene,
            self.scene.camera.position,
            self.normal_fallback,
            self.background,
        )
        for spec in runtime.spec.uniforms.values():
            _bind_manifest_uniform(program, spec)
        model = _model_matrix(obj)
        _bind_object_uniforms(program, self.view_proj, model)
        return self._vao(runtime.spec.name, obj.mesh)

    def _draw_objects(self, framebuffer: Any, blend_enabled: bool) -> None:
        framebuffer.use()
        for obj in self.objects:
            runtime = self.runtime[obj.material]
            _apply_depth(self.context, runtime.spec)
            _apply_blend(self.context, runtime.spec, blend_enabled)
            vao = self._bind_object(runtime, obj)
            vao.render()

    def _depth_prepass(self) -> np.ndarray:
        depth = self._alloc_depth()
        framebuffer = self._alloc_framebuffer([], depth)
        framebuffer.use()
        framebuffer.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self._draw_objects(framebuffer, False)
        return _read_texture(depth, self.width, self.height, 1)

    def _blend_coverage(self) -> np.ndarray:
        texture = self._alloc_color(1)
        framebuffer = self._alloc_framebuffer([texture])
        framebuffer.use()
        framebuffer.clear(0.0, 0.0, 0.0, 0.0)
        self.context.disable(moderngl.DEPTH_TEST)
        self.context.disable(moderngl.BLEND)
        for obj in self.objects:
            material = self.runtime[obj.material].spec
            if not material.blend.enabled:
                continue
            program = self.coverage_program
            _bind_object_uniforms(program, self.view_proj, _model_matrix(obj))
            self._coverage_vao(obj.mesh).render()
        return _read_texture(texture, self.width, self.height, 1)

    def _deferred(self, gbuffer: dict[str, np.ndarray]) -> np.ndarray:
        textures = (
            gbuffer["world_position_texture"],
            gbuffer["world_normal_texture"],
            gbuffer["albedo_texture"],
            gbuffer["metallic_roughness_texture"],
        )
        for unit, texture in enumerate(textures):
            texture.use(location=unit)
        self.deferred_program["u_gbuffer_position"].value = 0
        self.deferred_program["u_gbuffer_normal"].value = 1
        self.deferred_program["u_gbuffer_albedo"].value = 2
        self.deferred_program["u_gbuffer_metallic_roughness"].value = 3
        _bind_global_uniforms(
            self.deferred_program,
            self.scene,
            self.scene.camera.position,
            self.normal_fallback,
            self.background,
        )
        texture = self._alloc_color()
        framebuffer = self._alloc_framebuffer([texture])
        framebuffer.use()
        framebuffer.clear(*self.background)
        self.context.disable(moderngl.DEPTH_TEST)
        self.context.disable(moderngl.BLEND)
        self.fullscreen_vao.render(vertices=3)
        return _read_texture(texture, self.width, self.height, 4)

    def _forward(self) -> np.ndarray:
        attachments = [self._alloc_color() for _ in range(5)]
        depth = self._alloc_depth()
        framebuffer = self._alloc_framebuffer(attachments, depth)
        framebuffer.use()
        framebuffer.clear(*self.background, depth=1.0)
        self._draw_objects(framebuffer, True)
        return _read_texture(attachments[4], self.width, self.height, 4)

    def _gbuffer_textures(self) -> dict[str, Any]:
        textures = [self._alloc_color() for _ in range(4)]
        depth = self._alloc_depth()
        self._gbuffer_texture_frame = self._alloc_framebuffer(textures, depth)
        return {
            "world_position_texture": textures[0],
            "world_normal_texture": textures[1],
            "albedo_texture": textures[2],
            "metallic_roughness_texture": textures[3],
            "depth_texture": depth,
        }

    def render(self, comparison_settings: Any) -> RenderSceneResult:
        diagnostics: list[Diagnostic] = []
        depth_prepass = self._depth_prepass()
        textures = self._gbuffer_textures()
        self._gbuffer_texture_frame.use()
        self._gbuffer_texture_frame.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self._draw_objects(self._gbuffer_texture_frame, False)
        gbuffer = {
            "world_position": _read_texture(textures["world_position_texture"], self.width, self.height, 4),
            "world_normal": _read_texture(textures["world_normal_texture"], self.width, self.height, 4),
            "albedo": _read_texture(textures["albedo_texture"], self.width, self.height, 4),
            "metallic_roughness": _read_texture(textures["metallic_roughness_texture"], self.width, self.height, 4),
            "depth": _read_texture(textures["depth_texture"], self.width, self.height, 1),
        }
        blend_mask = self._blend_coverage()
        gbuffer_for_lighting = dict(gbuffer)
        gbuffer_for_lighting.update(textures)
        deferred = self._deferred(gbuffer_for_lighting)
        forward = self._forward()
        render_settings = RenderSettings(
            self.width,
            self.height,
            self.background,
            self.normal_fallback,
            self.normal_tolerance,
        )
        stats = analyze_gbuffer(
            gbuffer["world_position"],
            gbuffer["world_normal"],
            gbuffer["albedo"],
            gbuffer["metallic_roughness"],
            render_settings,
        )
        comparison = compare_images(
            deferred,
            forward,
            gbuffer["world_normal"],
            gbuffer["metallic_roughness"],
            blend_mask,
            comparison_settings,
            render_settings,
        )
        depth_difference = np.abs(depth_prepass[:, :, 0] - gbuffer["depth"][:, :, 0])
        depth_mismatch = int(np.count_nonzero(depth_difference > 1.0e-6))
        if stats["normal_fallback_pixels"] > 0:
            diagnostics.append(
                Diagnostic(
                    "warning",
                    "normal_fallback_applied",
                    f"{stats['normal_fallback_pixels']} covered pixels received the configured fallback normal",
                    material=None,
                    details={"pixels": stats["normal_fallback_pixels"], "fallback": list(self.normal_fallback)},
                )
            )
        if stats["normal_non_unit_pixels"] > 0:
            diagnostics.append(
                Diagnostic(
                    "error",
                    "gbuffer_normal_non_unit",
                    f"{stats['normal_non_unit_pixels']} G-Buffer normals are not unit length",
                    details={"pixels": stats["normal_non_unit_pixels"], "tolerance": self.normal_tolerance},
                )
            )
        if stats["normal_fallback_mismatch_pixels"] > 0:
            diagnostics.append(
                Diagnostic(
                    "error",
                    "normal_fallback_mismatch",
                    f"{stats['normal_fallback_mismatch_pixels']} fallback normals differ from the manifest value",
                    details={"pixels": stats["normal_fallback_mismatch_pixels"]},
                )
            )
        if stats["metallic_out_of_range_pixels"] > 0:
            diagnostics.append(
                Diagnostic(
                    "error",
                    "rendered_metallic_out_of_range",
                    f"{stats['metallic_out_of_range_pixels']} covered pixels reported out-of-range metallic input",
                    details={"pixels": stats["metallic_out_of_range_pixels"]},
                )
            )
        if stats["roughness_out_of_range_pixels"] > 0:
            diagnostics.append(
                Diagnostic(
                    "error",
                    "rendered_roughness_out_of_range",
                    f"{stats['roughness_out_of_range_pixels']} covered pixels reported out-of-range roughness input",
                    details={"pixels": stats["roughness_out_of_range_pixels"]},
                )
            )
        if stats["non_finite_pixels"] > 0:
            diagnostics.append(
                Diagnostic(
                    "error",
                    "gbuffer_non_finite",
                    f"{stats['non_finite_pixels']} covered pixels contain non-finite G-Buffer data",
                    details={"pixels": stats["non_finite_pixels"]},
                )
            )
        if depth_mismatch > 0:
            diagnostics.append(
                Diagnostic(
                    "error",
                    "depth_prepass_mismatch",
                    f"Depth prepass and G-Buffer depth differ at {depth_mismatch} pixels",
                    details={"pixels": depth_mismatch},
                )
            )
        buffers = {
            "world_position": gbuffer["world_position"],
            "world_normal": gbuffer["world_normal"],
            "albedo": gbuffer["albedo"],
            "metallic_roughness": gbuffer["metallic_roughness"],
            "depth_prepass": depth_prepass,
            "depth_gbuffer": gbuffer["depth"],
            "deferred": deferred,
            "forward": forward,
            "blend_coverage": blend_mask,
        }
        return RenderSceneResult(
            scene=self.scene.name,
            width=self.width,
            height=self.height,
            buffers=buffers,
            comparison=comparison,
            gbuffer={
                "slots": {
                    "world_position": 0,
                    "world_normal": 1,
                    "albedo": 2,
                    "metallic_roughness": 3,
                },
                **stats,
            },
            depth_prepass_mismatch_pixels=depth_mismatch,
            diagnostics=diagnostics,
        )

    def release(self) -> None:
        for vao in self.vaos.values():
            vao.release()
        self.fullscreen_vao.release()
        for program in self.runtime.values():
            program.program.release()
        self.deferred_program.release()
        self.coverage_program.release()
        for framebuffer in self.framebuffers:
            framebuffer.release()
        for texture in self.textures:
            texture.release()
        for buffer in self.buffers:
            buffer.release()


def _render_scene_once(
    scene: SceneSpec,
    materials: dict[str, MaterialSpec],
    width: int,
    height: int,
    normal_fallback: list[float],
    normal_tolerance: float,
    background: list[float],
    comparison: Any,
) -> RenderSceneResult:
    try:
        context, context_info = create_headless_context()
    except Exception as exc:
        return RenderSceneResult(
            scene=scene.name,
            width=width,
            height=height,
            diagnostics=[
                Diagnostic(
                    "error",
                    "render_context_error",
                    f"Cannot create render OpenGL context for scene {scene.name}: {type(exc).__name__}: {exc}",
                )
            ],
        )
    renderer: SceneRenderer | None = None
    try:
        renderer = SceneRenderer(
            context,
            materials,
            scene,
            width,
            height,
            normal_fallback,
            normal_tolerance,
            background,
        )
        result = renderer.render(comparison)
        result.context = context_info
        return result
    except Exception as exc:
        return RenderSceneResult(
            scene=scene.name,
            width=width,
            height=height,
            context=context_info,
            diagnostics=[
                Diagnostic(
                    "error",
                    "scene_render_error",
                    f"Scene {scene.name} failed to render: {type(exc).__name__}: {exc}",
                )
            ],
        )
    finally:
        if renderer is not None:
            try:
                renderer.release()
            except Exception:
                pass
        context.release()


def render_scenes(
    scenes: Iterable[SceneSpec],
    materials: dict[str, MaterialSpec],
    width: int,
    height: int,
    normal_fallback: list[float],
    normal_tolerance: float,
    background: list[float],
    comparison: Any,
    workers: int = 1,
) -> list[RenderSceneResult]:
    scene_list = list(scenes)
    arguments = [
        (
            scene,
            materials,
            width,
            height,
            normal_fallback,
            normal_tolerance,
            background,
            comparison,
        )
        for scene in scene_list
    ]
    if workers <= 1:
        return [_render_scene_once(*item) for item in arguments]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scene-render") as executor:
        return list(executor.map(lambda item: _render_scene_once(*item), arguments))
