from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._gl import GL
from .config import HARD_MAX_POINT_LIGHTS, RenderConfig
from .errors import RenderError
from .gl_utils import (
    ShaderProgram,
    check_framebuffer,
    check_gl_error,
    clear_gl_errors,
)
from .math3d import (
    cube_face_view_matrices,
    gl_matrix,
    gl_normal_matrix,
    look_at,
    orthographic,
    perspective,
    rotation_y,
    to_column_major,
)
from .model import PointLight, SceneModel, select_point_lights

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uViewProjection;
uniform mat3 uNormalMatrix;

out vec3 vWorldPosition;
out vec3 vNormal;

void main() {
    vec4 world = uModel * vec4(aPosition, 1.0);
    vWorldPosition = world.xyz;
    vNormal = normalize(uNormalMatrix * aNormal);
    gl_Position = uViewProjection * world;
}
"""

DEPTH_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPosition;
uniform mat4 uModel;
uniform mat4 uLightViewProjection;
void main() {
    gl_Position = uLightViewProjection * uModel * vec4(aPosition, 1.0);
}
"""

DEPTH_FRAGMENT_SHADER = """
#version 330 core
void main() {
}
"""

FRAGMENT_SHADER = """
#version 330 core
#define MAX_POINT_LIGHTS 8

const float PI = 3.14159265359;
const float POINT_NEAR = 0.05;

in vec3 vWorldPosition;
in vec3 vNormal;
out vec4 FragColor;

uniform vec3 uCameraPosition;
uniform vec3 uAmbient;
uniform vec3 uMaterialDiffuse;
uniform vec3 uMaterialSpecular;
uniform float uRoughness;
uniform float uMetallic;

uniform vec3 uDirectionalDirection;
uniform vec3 uDirectionalColor;
uniform bool uDirectionalCastShadow;
uniform mat4 uDirectionalViewProjection;
uniform sampler2DShadow uDirectionalShadow;
uniform float uShadowBias;
uniform float uShadowNormalBias;

uniform int uPointCount;
uniform vec3 uPointPositions[MAX_POINT_LIGHTS];
uniform vec3 uPointColors[MAX_POINT_LIGHTS];
uniform float uPointRanges[MAX_POINT_LIGHTS];
uniform int uPointCastShadow[MAX_POINT_LIGHTS];
uniform samplerCubeShadow uPointShadows[MAX_POINT_LIGHTS];
uniform float uPointShadowBias;
uniform float uPointShadowNormalBias;

uniform float uExposure;

float distribution_ggx(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float n_dot_h = max(dot(N, H), 0.0);
    float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * d * d, 1e-7);
}

float geometry_schlick_ggx(float n_dot_v, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
    float n_dot_v = max(dot(N, V), 0.0);
    float n_dot_l = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

vec3 fresnel_schlick(float cosine, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosine, 0.0, 1.0), 5.0);
}

float directional_shadow(vec3 N, vec3 L) {
    if (!uDirectionalCastShadow) {
        return 1.0;
    }
    vec3 shadow_position = vWorldPosition + N * uShadowNormalBias;
    vec4 light_clip = uDirectionalViewProjection * vec4(shadow_position, 1.0);
    vec3 proj = light_clip.xyz / light_clip.w * 0.5 + 0.5;
    if (proj.x <= 0.0 || proj.x >= 1.0 || proj.y <= 0.0 || proj.y >= 1.0 ||
        proj.z <= 0.0 || proj.z >= 1.0) {
        return 1.0;
    }
    float n_dot_l = max(dot(N, L), 0.0);
    float slope = sqrt(max(1.0 - n_dot_l * n_dot_l, 0.0)) / max(n_dot_l, 0.15);
    float compare = proj.z - uShadowBias * (1.0 + slope);
    vec2 texel = 1.0 / vec2(textureSize(uDirectionalShadow, 0));
    float visibility = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            visibility += texture(
                uDirectionalShadow,
                vec3(proj.xy + vec2(x, y) * texel, compare)
            );
        }
    }
    return visibility / 9.0;
}

float point_shadow(int index, vec3 N) {
    if (uPointCastShadow[index] == 0) {
        return 1.0;
    }
    vec3 shadow_position = vWorldPosition + N * uPointShadowNormalBias;
    vec3 delta = shadow_position - uPointPositions[index];
    float distance_to_light = max(length(delta), POINT_NEAR);
    float far_plane = uPointRanges[index];
    float ndc_depth =
        ((far_plane + POINT_NEAR) - 2.0 * far_plane * POINT_NEAR / distance_to_light) /
        (far_plane - POINT_NEAR);
    float depth_derivative =
        2.0 * far_plane * POINT_NEAR /
        max(distance_to_light * distance_to_light * (far_plane - POINT_NEAR), 1e-7);
    float compare = ndc_depth * 0.5 + 0.5 - uPointShadowBias * depth_derivative;
    vec4 sample_coord = vec4(delta, compare);
    if (index == 0) return texture(uPointShadows[0], sample_coord);
    if (index == 1) return texture(uPointShadows[1], sample_coord);
    if (index == 2) return texture(uPointShadows[2], sample_coord);
    if (index == 3) return texture(uPointShadows[3], sample_coord);
    if (index == 4) return texture(uPointShadows[4], sample_coord);
    if (index == 5) return texture(uPointShadows[5], sample_coord);
    if (index == 6) return texture(uPointShadows[6], sample_coord);
    return texture(uPointShadows[7], sample_coord);
}

vec3 direct_brdf(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 base_color, vec3 f0) {
    vec3 H = normalize(V + L);
    float n_dot_l = max(dot(N, L), 0.0);
    float n_dot_v = max(dot(N, V), 0.0);
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3(0.0);
    }
    vec3 F = fresnel_schlick(max(dot(H, V), 0.0), f0);
    float D = distribution_ggx(N, H, uRoughness);
    float G = geometry_smith(N, V, L, uRoughness);
    vec3 specular = (D * G * F) / max(4.0 * n_dot_v * n_dot_l, 1e-6);
    vec3 kd = (vec3(1.0) - F) * (1.0 - uMetallic);
    vec3 diffuse = kd * base_color / PI;
    return (diffuse + specular) * radiance * n_dot_l;
}

vec3 tone_map(vec3 color) {
    color *= uExposure;
    color = color / (color + vec3(1.0));
    return pow(color, vec3(1.0 / 2.2));
}

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCameraPosition - vWorldPosition);
    vec3 albedo = uMaterialDiffuse;
    vec3 diffuse_color = albedo * (1.0 - uMetallic);
    vec3 f0 = mix(vec3(0.04) * uMaterialSpecular, uMaterialDiffuse * uMaterialSpecular, uMetallic);

    vec3 color = vec3(0.0);
    vec3 directional_l = normalize(-uDirectionalDirection);
    float directional_visibility = directional_shadow(N, directional_l);
    color += direct_brdf(
        N, V, directional_l, uDirectionalColor, albedo, f0
    ) * directional_visibility;

    for (int i = 0; i < uPointCount; ++i) {
        vec3 to_light = uPointPositions[i] - vWorldPosition;
        float distance_sq = max(dot(to_light, to_light), 1e-6);
        float distance_to_light = sqrt(distance_sq);
        vec3 L = to_light / distance_to_light;
        float range = max(uPointRanges[i], 1e-3);
        float window = clamp(1.0 - pow(distance_to_light / range, 4.0), 0.0, 1.0);
        float attenuation = (window * window) / (distance_sq + 1.0);
        float visibility = point_shadow(i, N);
        color += direct_brdf(
            N, V, L, uPointColors[i] * attenuation, albedo, f0
        ) * visibility;
    }

    vec3 hemi_ambient = mix(uAmbient * 0.45, uAmbient, N.y * 0.5 + 0.5);
    vec3 ambient_fresnel = f0 + (max(vec3(1.0 - uRoughness), f0) - f0) *
        pow(1.0 - max(dot(N, V), 0.0), 5.0);
    color += diffuse_color * hemi_ambient;
    color += ambient_fresnel * hemi_ambient * (0.12 + 0.55 * (1.0 - uRoughness));

    FragColor = vec4(tone_map(color), 1.0);
}
"""


@dataclass
class _MeshGPU:
    name: str
    vao: int
    index_count: int
    transform: np.ndarray
    normal_matrix: np.ndarray
    material_name: str


class _Resources:
    def __init__(self) -> None:
        self.textures: list[int] = []
        self.buffers: list[int] = []
        self.vertex_arrays: list[int] = []
        self.framebuffers: list[int] = []

    def texture(self) -> int:
        value = int(GL.glGenTextures(1))
        self.textures.append(value)
        return value

    def buffer(self) -> int:
        value = int(GL.glGenBuffers(1))
        self.buffers.append(value)
        return value

    def vertex_array(self) -> int:
        value = int(GL.glGenVertexArrays(1))
        self.vertex_arrays.append(value)
        return value

    def framebuffer(self) -> int:
        value = int(GL.glGenFramebuffers(1))
        self.framebuffers.append(value)
        return value

    def close(self) -> None:
        for framebuffer in reversed(self.framebuffers):
            GL.glDeleteFramebuffers(1, [framebuffer])
        for vao in reversed(self.vertex_arrays):
            GL.glDeleteVertexArrays(1, [vao])
        for buffer in reversed(self.buffers):
            GL.glDeleteBuffers(1, [buffer])
        for texture in reversed(self.textures):
            GL.glDeleteTextures([texture])
        self.framebuffers.clear()
        self.vertex_arrays.clear()
        self.buffers.clear()
        self.textures.clear()

    def counts(self) -> dict[str, int]:
        return {
            "textures": len(self.textures),
            "buffers": len(self.buffers),
            "vertex_arrays": len(self.vertex_arrays),
            "framebuffers": len(self.framebuffers),
        }


class _RenderStateSnapshot:
    """Capture and restore state touched by the render passes."""

    def __init__(self) -> None:
        self.viewport = [int(x) for x in GL.glGetIntegerv(GL.GL_VIEWPORT)]
        self.depth_test = bool(GL.glIsEnabled(GL.GL_DEPTH_TEST))
        self.depth_func = int(GL.glGetIntegerv(GL.GL_DEPTH_FUNC))
        self.depth_write = bool(GL.glGetIntegerv(GL.GL_DEPTH_WRITEMASK))
        self.cull_face = bool(GL.glIsEnabled(GL.GL_CULL_FACE))
        self.cull_mode = int(GL.glGetIntegerv(GL.GL_CULL_FACE_MODE))
        self.front_face = int(GL.glGetIntegerv(GL.GL_FRONT_FACE))
        self.blend = bool(GL.glIsEnabled(GL.GL_BLEND))
        self.scissor_test = bool(GL.glIsEnabled(GL.GL_SCISSOR_TEST))
        self.stencil_test = bool(GL.glIsEnabled(GL.GL_STENCIL_TEST))
        self.framebuffer_srgb = bool(GL.glIsEnabled(GL.GL_FRAMEBUFFER_SRGB))
        self.rasterizer_discard = bool(GL.glIsEnabled(GL.GL_RASTERIZER_DISCARD))
        self.color_mask = [bool(value) for value in GL.glGetBooleanv(GL.GL_COLOR_WRITEMASK)]
        self.polygon_offset = bool(GL.glIsEnabled(GL.GL_POLYGON_OFFSET_FILL))
        self.polygon_offset_factor = float(
            GL.glGetFloatv(GL.GL_POLYGON_OFFSET_FACTOR)
        )
        self.polygon_offset_units = float(GL.glGetFloatv(GL.GL_POLYGON_OFFSET_UNITS))
        self.pack_alignment = int(GL.glGetIntegerv(GL.GL_PACK_ALIGNMENT))
        self.depth_clear = float(GL.glGetFloatv(GL.GL_DEPTH_CLEAR_VALUE))
        self.depth_range = [float(value) for value in GL.glGetFloatv(GL.GL_DEPTH_RANGE)]
        self.draw_framebuffer = int(GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING))
        self.read_framebuffer = int(GL.glGetIntegerv(GL.GL_READ_FRAMEBUFFER_BINDING))
        self.program = int(GL.glGetIntegerv(GL.GL_CURRENT_PROGRAM))
        self.vertex_array = int(GL.glGetIntegerv(GL.GL_VERTEX_ARRAY_BINDING))
        self.active_texture = int(GL.glGetIntegerv(GL.GL_ACTIVE_TEXTURE))
        self.color_clear = [float(x) for x in GL.glGetFloatv(GL.GL_COLOR_CLEAR_VALUE)]
        self.texture_2d: list[int] = []
        self.texture_cube: list[int] = []
        for unit in range(HARD_MAX_POINT_LIGHTS + 1):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            self.texture_2d.append(int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D)))
            self.texture_cube.append(int(GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_CUBE_MAP)))
        GL.glActiveTexture(self.active_texture)

    def restore(self) -> None:
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.draw_framebuffer)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.read_framebuffer)
        GL.glViewport(*self.viewport)
        GL.glUseProgram(self.program)
        GL.glBindVertexArray(self.vertex_array)
        if self.depth_test:
            GL.glEnable(GL.GL_DEPTH_TEST)
        else:
            GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(self.depth_func)
        GL.glDepthMask(self.depth_write)
        if self.cull_face:
            GL.glEnable(GL.GL_CULL_FACE)
        else:
            GL.glDisable(GL.GL_CULL_FACE)
        GL.glCullFace(self.cull_mode)
        GL.glFrontFace(self.front_face)
        if self.blend:
            GL.glEnable(GL.GL_BLEND)
        else:
            GL.glDisable(GL.GL_BLEND)
        for capability, enabled in (
            (GL.GL_SCISSOR_TEST, self.scissor_test),
            (GL.GL_STENCIL_TEST, self.stencil_test),
            (GL.GL_FRAMEBUFFER_SRGB, self.framebuffer_srgb),
            (GL.GL_RASTERIZER_DISCARD, self.rasterizer_discard),
        ):
            if enabled:
                GL.glEnable(capability)
            else:
                GL.glDisable(capability)
        GL.glColorMask(*[GL.GL_TRUE if value else GL.GL_FALSE for value in self.color_mask])
        GL.glClearDepth(self.depth_clear)
        GL.glDepthRange(*self.depth_range)
        if self.polygon_offset:
            GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        else:
            GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(self.polygon_offset_factor, self.polygon_offset_units)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, self.pack_alignment)
        GL.glClearColor(*self.color_clear)
        for unit in range(HARD_MAX_POINT_LIGHTS + 1):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_2d[unit])
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.texture_cube[unit])
        GL.glActiveTexture(self.active_texture)


class CharacterRenderer:
    """Owns all OpenGL objects needed by one renderer instance."""

    def __init__(self, model: SceneModel, config: RenderConfig) -> None:
        self.model = model
        self.config = config
        self._resources = _Resources()
        self._programs: list[ShaderProgram] = []
        self._main_program: ShaderProgram | None = None
        self._depth_program: ShaderProgram | None = None
        self._meshes: list[_MeshGPU] = []
        self._main_fbo = 0
        self._main_color = 0
        self._main_depth = 0
        self._directional_fbo = 0
        self._directional_shadow = 0
        self._cube_fbo = 0
        self._point_shadows: list[int] = []
        self._initialized = False
        self._directional_view_projection = np.eye(4)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def resource_counts(self) -> dict[str, int]:
        counts = self._resources.counts()
        counts["programs"] = sum(1 for program in self._programs if program.program)
        return counts

    def initialize(self) -> None:
        if self._initialized:
            return
        clear_gl_errors()
        snapshot = _RenderStateSnapshot()
        try:
            self._create_programs()
            self._create_meshes()
            self._create_directional_shadow()
            self._create_point_shadows()
            self._create_main_target()
            self._prepare_directional_projection()
            check_gl_error("renderer initialization")
            self._initialized = True
        except Exception:
            self.close()
            raise
        finally:
            snapshot.restore()

    def close(self) -> None:
        GL.glUseProgram(0)
        GL.glBindVertexArray(0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        for program in reversed(self._programs):
            program.destroy()
        self._programs.clear()
        self._main_program = None
        self._depth_program = None
        self._resources.close()
        self._meshes.clear()
        self._point_shadows.clear()
        self._main_fbo = 0
        self._main_color = 0
        self._main_depth = 0
        self._directional_fbo = 0
        self._directional_shadow = 0
        self._cube_fbo = 0
        self._initialized = False

    def _create_programs(self) -> None:
        main = ShaderProgram(VERTEX_SHADER, FRAGMENT_SHADER, "pbr-main")
        self._programs.append(main)
        self._main_program = main
        depth = ShaderProgram(
            DEPTH_VERTEX_SHADER, DEPTH_FRAGMENT_SHADER, "shadow-depth"
        )
        self._programs.append(depth)
        self._depth_program = depth
        check_gl_error("create shader programs")

    def _create_meshes(self) -> None:
        for node in self.model.nodes:
            vao = self._resources.vertex_array()
            vbo = self._resources.buffer()
            ebo = self._resources.buffer()
            interleaved = np.concatenate(
                [
                    np.asarray(node.positions, dtype=np.float32),
                    np.asarray(node.normals, dtype=np.float32),
                ],
                axis=1,
            )
            indices = np.asarray(node.indices, dtype=np.uint32)
            GL.glBindVertexArray(vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL.GL_STATIC_DRAW
            )
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(
                0, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, ctypes.c_void_p(0)
            )
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(
                1, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, ctypes.c_void_p(12)
            )
            self._meshes.append(
                _MeshGPU(
                    name=node.name,
                    vao=vao,
                    index_count=int(indices.size),
                    transform=np.asarray(node.transform, dtype=np.float64),
                    normal_matrix=gl_normal_matrix(node.transform),
                    material_name=node.material.name,
                )
            )
        GL.glBindVertexArray(0)
        check_gl_error("upload mesh geometry")

    def _create_directional_shadow(self) -> None:
        size = self.config.shadow_map_size
        self._directional_shadow = self._resources.texture()
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._directional_shadow)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH_COMPONENT24,
            size,
            size,
            0,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
        GL.glTexParameterfv(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BORDER_COLOR, np.array([1, 1, 1, 1], dtype=np.float32)
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_COMPARE_MODE, GL.GL_COMPARE_REF_TO_TEXTURE
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_COMPARE_FUNC, GL.GL_LEQUAL)
        self._directional_fbo = self._resources.framebuffer()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._directional_fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_TEXTURE_2D,
            self._directional_shadow,
            0,
        )
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)
        check_framebuffer("directional shadow")
        check_gl_error("create directional shadow target")

    def _create_point_shadows(self) -> None:
        self._cube_fbo = self._resources.framebuffer()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._cube_fbo)
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)
        for light in self.model.point_lights:
            self._point_shadows.append(self._create_cube_shadow())
        if not self._point_shadows:
            # The shader declares a fixed-size sampler array. Keep a valid cube
            # texture available even for scenes without point lights.
            self._point_shadows.append(self._create_cube_shadow())
        check_gl_error("create point shadow cube maps")

    def _create_cube_shadow(self) -> int:
        size = self.config.point_shadow_size
        texture = self._resources.texture()
        GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, texture)
        for face in range(6):
            GL.glTexImage2D(
                GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                0,
                GL.GL_DEPTH_COMPONENT24,
                size,
                size,
                0,
                GL.GL_DEPTH_COMPONENT,
                GL.GL_FLOAT,
                None,
            )
        GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP,
            GL.GL_TEXTURE_COMPARE_MODE,
            GL.GL_COMPARE_REF_TO_TEXTURE,
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_CUBE_MAP, GL.GL_TEXTURE_COMPARE_FUNC, GL.GL_LEQUAL
        )
        return texture

    def _create_main_target(self) -> None:
        width, height = self.config.width, self.config.height
        self._main_color = self._resources.texture()
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._main_color)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA8,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            None,
        )
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

        self._main_depth = self._resources.texture()
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._main_depth)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_DEPTH_COMPONENT24,
            width,
            height,
            0,
            GL.GL_DEPTH_COMPONENT,
            GL.GL_FLOAT,
            None,
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )
        self._main_fbo = self._resources.framebuffer()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._main_fbo)
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_COLOR_ATTACHMENT0,
            GL.GL_TEXTURE_2D,
            self._main_color,
            0,
        )
        GL.glFramebufferTexture2D(
            GL.GL_FRAMEBUFFER,
            GL.GL_DEPTH_ATTACHMENT,
            GL.GL_TEXTURE_2D,
            self._main_depth,
            0,
        )
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        check_framebuffer("main color/depth")
        check_gl_error("create main framebuffer")

    def _prepare_directional_projection(self) -> None:
        direction = self.model.directional.direction
        lower, upper = self.model.bounds
        center = (lower + upper) * 0.5
        radius = max(float(np.linalg.norm(upper - lower)) * 0.5, 0.5)
        distance = radius * 2.5 + 0.5
        eye = center - direction * distance
        up = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(direction, up))) > 0.98:
            up = np.array([0.0, 0.0, 1.0])
        view = look_at(eye, center, up)
        near = max(0.05, distance - radius * 1.6)
        far = distance + radius * 1.6
        projection = orthographic(-radius, radius, -radius, radius, near, far)
        self._directional_view_projection = projection @ view

    def _draw_geometry(
        self, program: ShaderProgram, light_view_projection: np.ndarray | None
    ) -> None:
        for mesh in self._meshes:
            program.set_mat4("uModel", gl_matrix(mesh.transform))
            if light_view_projection is not None:
                program.set_mat4(
                    "uLightViewProjection", gl_matrix(light_view_projection)
                )
            GL.glBindVertexArray(mesh.vao)
            GL.glDrawElements(
                GL.GL_TRIANGLES,
                mesh.index_count,
                GL.GL_UNSIGNED_INT,
                ctypes.c_void_p(0),
            )

    def _render_directional_shadow(self) -> None:
        assert self._depth_program is not None
        self._prepare_pass_state()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._directional_fbo)
        GL.glViewport(0, 0, self.config.shadow_map_size, self.config.shadow_map_size)
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glPolygonOffset(1.5, 3.0)
        GL.glClearDepth(1.0)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        self._depth_program.use()
        self._draw_geometry(self._depth_program, self._directional_view_projection)
        check_gl_error("directional shadow pass")

    def _render_point_shadows(self, lights: tuple[PointLight, ...]) -> None:
        assert self._depth_program is not None
        self._prepare_pass_state()
        self._depth_program.use()
        near = 0.05
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._cube_fbo)
        GL.glViewport(0, 0, self.config.point_shadow_size, self.config.point_shadow_size)
        GL.glDrawBuffer(GL.GL_NONE)
        GL.glReadBuffer(GL.GL_NONE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glPolygonOffset(1.2, 2.0)
        for light in lights:
            if not light.cast_shadow:
                continue
            texture = self._point_shadows[light.source_index]
            far = light.range
            projection = perspective(90.0, 1.0, near, far)
            views = cube_face_view_matrices(light.position)
            for face in range(6):
                chance = GL.glGetError()
                if chance != GL.GL_NO_ERROR:
                    raise RenderError("E_GL", "OpenGL error before point shadow face", hex(chance))
                GL.glFramebufferTexture2D(
                    GL.GL_FRAMEBUFFER,
                    GL.GL_DEPTH_ATTACHMENT,
                    GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
                    texture,
                    0,
                )
                check_framebuffer(f"point shadow cube face {face}")
                GL.glClearDepth(1.0)
                GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
                self._draw_geometry(self._depth_program, projection @ views[face])
        check_gl_error("point shadow passes")

    def _bind_shadow_textures(self) -> None:
        assert self._main_program is not None
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._directional_shadow)
        self._main_program.set_int("uDirectionalShadow", 0)
        # Every sampler in the fixed-size array must resolve to a cube texture
        # unit. Otherwise drivers can report INVALID_OPERATION because the
        # unused samplers default to unit 0, which holds a 2D shadow map.
        for index in range(HARD_MAX_POINT_LIGHTS):
            texture = self._point_shadows[index % len(self._point_shadows)]
            GL.glActiveTexture(GL.GL_TEXTURE1 + index)
            GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, texture)
        location = self._main_program.uniform_location("uPointShadows[0]")
        if location >= 0:
            units = np.arange(
                1, HARD_MAX_POINT_LIGHTS + 1, dtype=np.int32
            )
            GL.glUniform1iv(location, HARD_MAX_POINT_LIGHTS, units)

    def _set_main_uniforms(
        self,
        camera_position: np.ndarray,
        view_projection: np.ndarray,
        lights: tuple[PointLight, ...],
    ) -> None:
        assert self._main_program is not None
        program = self._main_program
        program.set_vec3("uCameraPosition", camera_position)
        program.set_mat4("uViewProjection", gl_matrix(view_projection))
        program.set_vec3("uAmbient", self.model.ambient)
        program.set_vec3(
            "uDirectionalDirection", self.model.directional.direction
        )
        directional_scale = self.config.directional_light_scale
        program.set_vec3(
            "uDirectionalColor",
            self.model.directional.color * self.model.directional.intensity * directional_scale,
        )
        program.set_int(
            "uDirectionalCastShadow", int(self.model.directional.cast_shadow)
        )
        program.set_mat4(
            "uDirectionalViewProjection", gl_matrix(self._directional_view_projection)
        )
        program.set_float("uShadowBias", self.config.shadow_bias)
        program.set_float("uShadowNormalBias", self.config.shadow_normal_bias)
        program.set_float("uPointShadowBias", self.config.point_shadow_bias)
        program.set_float("uPointShadowNormalBias", self.config.point_shadow_normal_bias)
        program.set_float("uExposure", self.config.exposure)

        count = len(lights)
        program.set_int("uPointCount", count)
        if count:
            positions = np.asarray([light.position for light in lights], dtype=np.float32)
            colors = np.asarray(
                [
                    light.color * light.intensity * self.config.point_light_scale
                    for light in lights
                ],
                dtype=np.float32,
            )
            ranges = np.asarray([light.range for light in lights], dtype=np.float32)
            cast = np.asarray([int(light.cast_shadow) for light in lights], dtype=np.int32)
            self._set_vec3_array("uPointPositions", positions)
            self._set_vec3_array("uPointColors", colors)
            self._set_float_array("uPointRanges", ranges)
            self._set_int_array("uPointCastShadow", cast)

    def _set_vec3_array(self, name: str, values: np.ndarray) -> None:
        assert self._main_program is not None
        location = self._main_program.uniform_location(f"{name}[0]")
        if location >= 0:
            GL.glUniform3fv(location, len(values), values.reshape(-1))

    def _set_float_array(self, name: str, values: np.ndarray) -> None:
        assert self._main_program is not None
        location = self._main_program.uniform_location(f"{name}[0]")
        if location >= 0:
            GL.glUniform1fv(location, len(values), values)

    def _set_int_array(self, name: str, values: np.ndarray) -> None:
        assert self._main_program is not None
        location = self._main_program.uniform_location(f"{name}[0]")
        if location >= 0:
            GL.glUniform1iv(location, len(values), values)

    def _render_main(
        self,
        camera_position: np.ndarray,
        view_projection: np.ndarray,
        lights: tuple[PointLight, ...],
    ) -> None:
        assert self._main_program is not None
        self._prepare_pass_state()
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._main_fbo)
        GL.glViewport(0, 0, self.config.width, self.config.height)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glClearColor(0.012, 0.016, 0.026, 1.0)
        GL.glClearDepth(1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
        self._main_program.use()
        check_gl_error("main shader activation")
        self._bind_shadow_textures()
        check_gl_error("bind shadow textures")
        self._set_main_uniforms(camera_position, view_projection, lights)
        check_gl_error("upload main shader uniforms")
        for mesh in self._meshes:
            self._main_program.set_mat4("uModel", gl_matrix(mesh.transform))
            self._main_program.set_mat3("uNormalMatrix", mesh.normal_matrix)
            material = self.model.materials[mesh.material_name]
            self._main_program.set_vec3("uMaterialDiffuse", material.diffuse)
            self._main_program.set_vec3("uMaterialSpecular", material.specular)
            self._main_program.set_float("uRoughness", material.roughness)
            self._main_program.set_float("uMetallic", material.metallic)
            GL.glBindVertexArray(mesh.vao)
            check_gl_error(f"prepare mesh {mesh.name}")
            GL.glDrawElements(
                GL.GL_TRIANGLES,
                mesh.index_count,
                GL.GL_UNSIGNED_INT,
                ctypes.c_void_p(0),
            )
        check_gl_error("main render pass")

    @staticmethod
    def _prepare_pass_state() -> None:
        GL.glDisable(GL.GL_BLEND)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glDisable(GL.GL_STENCIL_TEST)
        GL.glDisable(GL.GL_FRAMEBUFFER_SRGB)
        GL.glDisable(GL.GL_RASTERIZER_DISCARD)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        GL.glFrontFace(GL.GL_CCW)
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glDepthRange(0.0, 1.0)

    def _animated_camera(self, time_seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        camera = self.model.camera
        offset = camera.position - camera.target
        angle = self.config.camera_orbit_speed_deg * time_seconds
        rotated = (rotation_y(angle) @ np.array([*offset, 1.0]))[:3]
        position = camera.target + rotated
        up = camera.up
        return position, camera.target, up

    def render_frame(
        self, frame_index: int, time_seconds: float, image_name: str
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not self._initialized:
            raise RenderError("E_RENDER_INIT", "renderer is not initialized")
        if frame_index < 0:
            raise RenderError("E_RENDER_ARGUMENT", "frame index must be non-negative")
        selected = select_point_lights(
            self.model.point_lights, self.config.max_point_lights, self.model.camera.target
        )
        camera_position, target, up = self._animated_camera(time_seconds)
        aspect = self.config.width / self.config.height
        projection = perspective(
            self.model.camera.fov_deg, aspect, 0.05, 100.0
        )
        view = look_at(camera_position, target, up)
        view_projection = projection @ view
        snapshot = _RenderStateSnapshot()
        try:
            self._render_directional_shadow()
            self._render_point_shadows(selected)
            self._render_main(camera_position, view_projection, selected)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._main_fbo)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
            data = GL.glReadPixels(
                0,
                0,
                self.config.width,
                self.config.height,
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
            )
            if not isinstance(data, (bytes, bytearray)):
                data = bytes(data)
            image = np.frombuffer(data, dtype=np.uint8).reshape(
                self.config.height, self.config.width, 4
            )[::-1].copy()
            check_gl_error("read framebuffer")
        finally:
            snapshot.restore()

        metadata = self.frame_metadata(
            frame_index,
            time_seconds,
            image_name,
            selected,
            camera_position,
            target,
            up,
            projection,
            view,
            view_projection,
        )
        return image, metadata

    def frame_metadata(
        self,
        frame_index: int,
        time_seconds: float,
        image_name: str,
        selected: tuple[PointLight, ...],
        camera_position: np.ndarray,
        target: np.ndarray,
        up: np.ndarray,
        projection: np.ndarray,
        view: np.ndarray,
        view_projection: np.ndarray,
    ) -> dict[str, Any]:
        selected_names = {light.name for light in selected}
        selected_rows: list[dict[str, Any]] = []
        for order, light in enumerate(selected):
            delta = light.position - target
            distance_sq = max(float(np.dot(delta, delta)), 1e-6)
            selected_rows.append(
                {
                    "order": order,
                    "name": light.name,
                    "position": [float(x) for x in light.position],
                    "color": [float(x) for x in light.color],
                    "intensity": float(light.intensity),
                    "range": float(light.range),
                    "cast_shadow": bool(light.cast_shadow),
                    "selection_score": float(light.intensity / distance_sq),
                }
            )
        return {
            "schema_version": 1,
            "frame_index": frame_index,
            "time_seconds": float(time_seconds),
            "image": image_name,
            "resolution": [self.config.width, self.config.height],
            "camera": {
                "position": [float(x) for x in camera_position],
                "target": [float(x) for x in target],
                "up": [float(x) for x in up],
                "fov_deg": float(self.model.camera.fov_deg),
                "near": 0.05,
                "far": 100.0,
                "projection_matrix": to_column_major(projection),
                "view_matrix": to_column_major(view),
                "view_projection_matrix": to_column_major(view_projection),
            },
            "lights": {
                "ambient": [float(x) for x in self.model.ambient],
                "directional": {
                    "name": self.model.directional.name,
                    "direction": [float(x) for x in self.model.directional.direction],
                    "color": [float(x) for x in self.model.directional.color],
                    "intensity": float(
                        self.model.directional.intensity
                        * self.config.directional_light_scale
                    ),
                    "cast_shadow": bool(self.model.directional.cast_shadow),
                    "shadow_map_size": self.config.shadow_map_size,
                },
                "point_lights": selected_rows,
                "unselected_point_lights": sorted(
                    light.name
                    for light in self.model.point_lights
                    if light.name not in selected_names
                ),
                "selection_policy": "descending intensity / distance(target)^2; name/index tie-break",
                "max_active_point_lights": self.config.max_point_lights,
                "point_shadow_size": self.config.point_shadow_size,
            },
            "shadows": {
                "shadow_bias": self.config.shadow_bias,
                "shadow_normal_bias": self.config.shadow_normal_bias,
                "point_shadow_bias": self.config.point_shadow_bias,
                "point_shadow_normal_bias": self.config.point_shadow_normal_bias,
                "directional_pcf": "3x3",
                "point_cube_faces": 6,
            },
            "color_pipeline": {
                "lighting_space": "linear",
                "output_transform": "exposure, Reinhard tone map, gamma 2.2",
            },
        }
