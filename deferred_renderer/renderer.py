"""Two-pass deferred renderer: MRT geometry pass and Cook-Torrance lighting."""

from __future__ import annotations

import hashlib
import os
from ctypes import c_void_p
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import OpenGL.GL as gl
from PIL import Image

from .gbuffer import ATTACHMENT_SPECS, GBuffer, default_requested_formats
from .gl_helpers import ShaderProgram, binding_state_snapshot, drain_gl_errors
from .scene import (
    Mesh,
    look_at,
    make_ground_plane,
    make_unit_sphere,
    perspective,
    sphere_model,
)


def _load_shader(filename: str) -> str:
    path = Path(__file__).parent / "shaders" / filename
    return path.read_text(encoding="utf-8")


def upload_mesh(vertex_data: np.ndarray, indices: np.ndarray) -> Mesh:
    vao = int(gl.glGenVertexArrays(1))
    gl.glBindVertexArray(vao)

    vbo = int(gl.glGenBuffers(1))
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(
        gl.GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, gl.GL_STATIC_DRAW
    )
    stride = 6 * 4
    gl.glEnableVertexAttribArray(0)
    gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
    gl.glEnableVertexAttribArray(1)
    gl.glVertexAttribPointer(
        1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, c_void_p(3 * 4)
    )

    ebo = int(gl.glGenBuffers(1))
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(
        gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW
    )
    gl.glBindVertexArray(0)
    return Mesh(vao=vao, vbo=vbo, ebo=ebo, index_count=int(indices.size))


def pixels_sha256(pixels: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class FrameResult:
    frame_index: int
    png_path: str
    png_sha256: str
    gbuffer_write_hash: str
    gbuffer_verify_hash: str
    state_mismatch: bool


class RenderError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class DeferredRenderer:
    def __init__(
        self,
        scene: dict[str, Any],
        width: int,
        height: int,
        fast_approx: bool,
        output_dir: str,
        force_formats: dict[str, str] | None = None,
    ):
        self.scene = scene
        self.width = width
        self.height = height
        self.fast_approx = fast_approx
        self.output_dir = output_dir
        self.force_formats = force_formats or {}
        self.warnings: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []

        requested = default_requested_formats()
        requested.update(self.force_formats)
        self.requested_formats = requested

        defines = {"FAST_APPROX": "1"} if fast_approx else {}
        self.geometry_program = ShaderProgram(
            _load_shader("geometry.vert"), _load_shader("geometry.frag")
        )
        self.lighting_program = ShaderProgram(
            _load_shader("lighting.vert"),
            _load_shader("lighting.frag"),
            defines=defines,
        )

        sphere_verts, sphere_idx = make_unit_sphere()
        ground_verts, ground_idx = make_ground_plane(scene["ground"]["size"])
        self.sphere_mesh = upload_mesh(sphere_verts, sphere_idx)
        self.ground_mesh = upload_mesh(ground_verts, ground_idx)

        cam = scene["camera"]
        self.view = look_at(cam["position"], cam["target"], cam["up"])
        self.proj = perspective(
            cam["fov_y_deg"], width / height, cam["near"], cam["far"]
        )
        self.cam_pos = np.asarray(cam["position"], dtype=np.float32)

        self.gbuffer = GBuffer(width, height, requested)
        build_warnings = self.gbuffer.build()
        self.warnings.extend(build_warnings)

        self._empty_vao = int(gl.glGenVertexArrays(1))

    # -- uniform helpers ----------------------------------------------------

    def _set_material_arrays(self, program: ShaderProgram) -> None:
        materials = self.scene["materials"]
        count = len(materials)
        program.set_vec3_array("uAlbedo", [m["albedo"] for m in materials])
        program.set_float_array("uRoughness", [m["roughness"] for m in materials])
        program.set_float_array("uMetallic", [m["metallic"] for m in materials])
        program.set_vec3_array("uEmissive", [m["emissive"] for m in materials])
        program.set_int("uMaterialCount", count)

    def _set_light_uniforms(self) -> None:
        prog = self.lighting_program
        point_lights = self.scene["point_lights"][:16]
        for index, light in enumerate(point_lights):
            base = f"uPointLights[{index}]"
            prog.set_vec3(f"{base}.position", light["position"])
            prog.set_float(f"{base}.range", light["range"])
            prog.set_vec3(f"{base}.color", light["color"])
            prog.set_int("uPointLightCount", len(point_lights))
            prog.set_float(f"{base}.intensity", light["intensity"])
        if not point_lights:
            prog.set_int("uPointLightCount", 0)

        directional = self.scene["directional_light"]
        prog.set_vec3("uDirDirection", directional["direction"])
        prog.set_vec3("uDirColor", directional["color"])
        prog.set_float("uDirIntensity", directional["intensity"])

    # -- passes -------------------------------------------------------------

    def _draw_object(
        self, mesh: Mesh, model: np.ndarray, material_id: int
    ) -> None:
        prog = self.geometry_program
        normal_mat = np.linalg.inv(model[:3, :3]).T
        prog.set_mat4("uModel", model)
        prog.set_mat3("uNormalMat", normal_mat)
        prog.set_int("uMaterialId", material_id)
        gl.glBindVertexArray(mesh.vao)
        gl.glDrawElements(
            gl.GL_TRIANGLES, mesh.index_count, gl.GL_UNSIGNED_INT, None
        )

    def geometry_pass(self, frame_index: int) -> str:
        self.gbuffer.bind()
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glCullFace(gl.GL_BACK)
        gl.glFrontFace(gl.GL_CCW)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(
            gl.GL_COLOR_BUFFER_BIT
            | gl.GL_DEPTH_BUFFER_BIT
            | gl.GL_STENCIL_BUFFER_BIT
        )

        self.geometry_program.use()
        self.geometry_program.set_mat4("uView", self.view)
        self.geometry_program.set_mat4("uProj", self.proj)
        self.geometry_program.set_vec3("uCamPos", self.cam_pos)
        self._set_material_arrays(self.geometry_program)

        ground_model = np.eye(4, dtype=np.float32)
        self._draw_object(
            self.ground_mesh, ground_model, self.scene["ground"]["material_id"]
        )
        for sphere in self.scene["spheres"]:
            model = sphere_model(sphere, frame_index)
            self._draw_object(self.sphere_mesh, model, sphere["material_id"])

        gl.glBindVertexArray(0)
        # Real GPU readback -> authoritative G-Buffer write hash.
        pixels = self.gbuffer.read_position_pixels(10)
        return pixels_sha256(pixels)

    def verify_gbuffer(self, expected_hash: str) -> tuple[str, np.ndarray]:
        # Independent readback immediately before the lighting pass.
        pixels = self.gbuffer.read_position_pixels(10)
        actual = pixels_sha256(pixels)
        return actual, pixels

    def lighting_pass(self) -> np.ndarray:
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glDrawBuffer(gl.GL_BACK)
        gl.glViewport(0, 0, self.width, self.height)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)

        self.gbuffer.bind_color_textures()
        self.lighting_program.use()
        for spec in ATTACHMENT_SPECS:
            self.lighting_program.set_int(f"g{self._sampler_suffix(spec['name'])}",
                                          spec["slot"])
        self.lighting_program.set_vec3("uCamPos", self.cam_pos)
        self._set_material_arrays(self.lighting_program)
        self._set_light_uniforms()

        gl.glBindVertexArray(self._empty_vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)
        gl.glBindVertexArray(0)

        gl.glReadBuffer(gl.GL_BACK)
        pixels = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        gl.glReadPixels(
            0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, pixels
        )
        return pixels

    @staticmethod
    def _sampler_suffix(attachment_name: str) -> str:
        return {
            "world_position": "Position",
            "world_normal": "Normal",
            "albedo_material": "AlbedoMat",
        }[attachment_name]

    # -- frame lifecycle ----------------------------------------------------

    def render_frame(self, frame_index: int) -> FrameResult:
        write_hash = self.geometry_pass(frame_index)
        verify_hash, _ = self.verify_gbuffer(write_hash)
        if verify_hash != write_hash:
            raise RenderError(
                "RENDER_STATE_MISMATCH",
                "G-Buffer position readback changed between geometry and lighting",
                {
                    "frame": frame_index,
                    "expected_sha256": write_hash,
                    "actual_sha256": verify_hash,
                },
            )

        pixels = self.lighting_pass()
        png_name = f"frame_{frame_index:04d}.png"
        png_path = os.path.join(self.output_dir, png_name)
        image = Image.fromarray(pixels[::-1], mode="RGBA")
        image.save(png_path, format="PNG")
        # Hash the actual PNG bytes written to disk so the manifest matches
        # what a verifier recomputes from the file.
        png_hash = file_sha256(png_path)

        return FrameResult(
            frame_index=frame_index,
            png_path=png_name,
            png_sha256=png_hash,
            gbuffer_write_hash=write_hash,
            gbuffer_verify_hash=verify_hash,
            state_mismatch=False,
        )

    def rebuild_gbuffer(self, resume_frame: int) -> dict[str, Any]:
        """Destroy and recreate an identical G-Buffer/FBO (render-target rebuild)."""
        before = binding_state_snapshot()
        old_fbo = self.gbuffer.fbo
        self.gbuffer.destroy()

        rebuilt = GBuffer(self.width, self.height, self.requested_formats)
        rebuild_warnings = rebuilt.build()
        self.warnings.extend(rebuild_warnings)
        rebuilt.bind()
        self.gbuffer = rebuilt
        after = binding_state_snapshot()

        event = {
            "event": "gbuffer_rebuild",
            "resume_frame": resume_frame,
            "destroyed_framebuffer": old_fbo,
            "recreated_framebuffer": self.gbuffer.fbo,
            "active_formats": dict(self.gbuffer.active_formats),
            "binding_state_before": before,
            "binding_state_after": after,
        }
        self.events.append(event)
        return event

    def close(self) -> None:
        self.gbuffer.destroy()
        for mesh in (self.sphere_mesh, self.ground_mesh):
            gl.glDeleteBuffers(1, [mesh.ebo])
            gl.glDeleteBuffers(1, [mesh.vbo])
            gl.glDeleteVertexArrays(1, [mesh.vao])
        gl.glDeleteVertexArrays(1, [self._empty_vao])
        self.geometry_program.close()
        self.lighting_program.close()
        drain_gl_errors()
