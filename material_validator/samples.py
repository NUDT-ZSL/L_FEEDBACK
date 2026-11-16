from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LIGHTING_GLSL = """\
uniform vec3 u_camera_position;
uniform vec3 u_ambient_color;
uniform int u_light_count;
uniform vec3 u_light_position[4];
uniform vec3 u_light_color[4];
uniform float u_light_intensity[4];
uniform float u_light_radius[4];

vec3 shade_surface(vec3 P, vec3 N, vec3 albedo, float metallic, float roughness) {
    vec3 V = normalize(u_camera_position - P);
    float metal = clamp(metallic, 0.0, 1.0);
    float rough = clamp(roughness, 0.0, 1.0);
    vec3 color = u_ambient_color * albedo * (1.0 - 0.35 * metal);
    vec3 F0 = mix(vec3(0.04), albedo, metal);
    vec3 diffuse = albedo * (1.0 - metal);
    float shininess = mix(64.0, 4.0, rough);
    for (int i = 0; i < 4; ++i) {
        if (i >= u_light_count) {
            continue;
        }
        vec3 to_light = u_light_position[i] - P;
        float distance_to_light = length(to_light);
        vec3 L = to_light / max(distance_to_light, 1.0e-5);
        float radius = max(u_light_radius[i], 1.0e-4);
        float attenuation = 1.0 / (1.0 + (distance_to_light * distance_to_light) / (radius * radius));
        float ndl = max(dot(N, L), 0.0);
        vec3 H = normalize(L + V);
        float specular = pow(max(dot(N, H), 0.0), shininess) * (1.0 - rough);
        vec3 radiance = u_light_color[i] * u_light_intensity[i] * attenuation;
        color += (diffuse * ndl + F0 * specular) * radiance;
    }
    return color;
}
"""


STANDARD_VERTEX = """\
#version 330 core
in vec3 in_position;
in vec3 in_normal;

uniform mat4 u_model;
uniform mat4 u_view_proj;
uniform mat3 u_normal_matrix;
uniform float u_vertex_offset;

out vec3 v_world_position;
out vec3 v_world_normal;

void main() {
    vec3 local_position = in_position + in_normal * u_vertex_offset;
    vec4 world_position = u_model * vec4(local_position, 1.0);
    v_world_position = world_position.xyz;
    v_world_normal = u_normal_matrix * in_normal;
    gl_Position = u_view_proj * world_position;
}
"""


STANDARD_FRAGMENT = f"""\
#version 330 core
in vec3 v_world_position;
in vec3 v_world_normal;

uniform vec3 u_base_color;
uniform float u_metallic;
uniform float u_roughness;
uniform mat3 u_tint_matrix;
uniform bool u_enable_tint;
uniform float u_opacity;
uniform vec3 u_normal_fallback;
{LIGHTING_GLSL}
layout(location = 0) out vec4 frag_world_position;
layout(location = 1) out vec4 frag_world_normal;
layout(location = 2) out vec4 frag_albedo;
layout(location = 3) out vec4 frag_metallic_roughness;
layout(location = 4) out vec4 frag_color;

void main() {{
    float normal_length_squared = dot(v_world_normal, v_world_normal);
    bool normal_valid = normal_length_squared > 1.0e-10;
    vec3 normal = normal_valid
        ? v_world_normal * inversesqrt(normal_length_squared)
        : normalize(u_normal_fallback);
    vec3 tinted = u_enable_tint ? u_tint_matrix * u_base_color : u_base_color;
    vec3 albedo = clamp(tinted, 0.0, 1.0);
    float metallic = clamp(u_metallic, 0.0, 1.0);
    float roughness = clamp(u_roughness, 0.0, 1.0);
    float metallic_invalid = (u_metallic < 0.0 || u_metallic > 1.0) ? 1.0 : 0.0;
    float roughness_invalid = (u_roughness < 0.0 || u_roughness > 1.0) ? 1.0 : 0.0;
    vec3 color = shade_surface(v_world_position, normal, albedo, metallic, roughness);

    frag_world_position = vec4(v_world_position, 1.0);
    frag_world_normal = vec4(normal, normal_valid ? 1.0 : 0.0);
    frag_albedo = vec4(albedo, 1.0);
    frag_metallic_roughness = vec4(metallic, roughness, metallic_invalid, roughness_invalid);
    frag_color = vec4(color, clamp(u_opacity, 0.0, 1.0));
}}
"""


DEFERRED_FRAGMENT = f"""\
#version 330 core
uniform sampler2D u_gbuffer_position;
uniform sampler2D u_gbuffer_normal;
uniform sampler2D u_gbuffer_albedo;
uniform sampler2D u_gbuffer_metallic_roughness;
uniform vec3 u_normal_fallback;
uniform vec4 u_background_color;
{LIGHTING_GLSL}
layout(location = 0) out vec4 frag_color;

void main() {{
    vec4 albedo_data = texture(u_gbuffer_albedo, gl_FragCoord.xy / vec2(textureSize(u_gbuffer_albedo, 0)));
    if (albedo_data.a < 0.5) {{
        frag_color = u_background_color;
        return;
    }}
    vec3 position = texture(u_gbuffer_position, gl_FragCoord.xy / vec2(textureSize(u_gbuffer_position, 0))).xyz;
    vec4 normal_data = texture(u_gbuffer_normal, gl_FragCoord.xy / vec2(textureSize(u_gbuffer_normal, 0)));
    float normal_length_squared = dot(normal_data.xyz, normal_data.xyz);
    vec3 normal = normal_length_squared > 1.0e-10
        ? normal_data.xyz * inversesqrt(normal_length_squared)
        : normalize(u_normal_fallback);
    vec2 metallic_roughness = texture(
        u_gbuffer_metallic_roughness,
        gl_FragCoord.xy / vec2(textureSize(u_gbuffer_metallic_roughness, 0))
    ).xy;
    vec3 color = shade_surface(position, normal, albedo_data.rgb, metallic_roughness.x, metallic_roughness.y);
    frag_color = vec4(color, u_background_color.a);
}}
"""


FULLSCREEN_VERTEX = """\
#version 330 core
const vec2 positions[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
void main() {
    gl_Position = vec4(positions[gl_VertexID], 0.0, 1.0);
}
"""


COVERAGE_FRAGMENT = """\
#version 330 core
layout(location = 0) out vec4 frag_mask;
void main() {
    frag_mask = vec4(1.0, 1.0, 1.0, 1.0);
}
"""


BROKEN_VERTEX = """\
#version 330 core
in vec3 in_position;
uniform mat4 u_model;
void main() {
    vec4 world = u_model * vec4(in_position, 1.0)
    gl_Position = world
}
"""


BROKEN_FRAGMENT = """\
#version 330 core
layout(location = 0) out vec4 frag_world_position;
void main() {
    vec3 bad_value = vec3(1.0, 2.0, );
    frag_world_position = vec4(bad_value, 1.0)
}
"""


MISSING_FRAGMENT = STANDARD_FRAGMENT.replace(
    "uniform vec3 u_normal_fallback;",
    "uniform vec3 u_normal_fallback;\nuniform float u_missing_tint;",
).replace(
    "vec3 albedo = clamp(tinted, 0.0, 1.0);",
    "vec3 albedo = clamp(tinted + vec3(u_missing_tint), 0.0, 1.0);",
)


def _uniform(type_name: str, value: Any) -> dict[str, Any]:
    return {"type": type_name, "value": value}


def _material(
    vertex: str,
    fragment: str,
    name: str,
    base_color: list[float] | None = None,
    metallic: float = 0.0,
    roughness: float = 0.5,
    opacity: float = 1.0,
    blend: dict[str, Any] | None = None,
    depth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    color = base_color or [0.7, 0.7, 0.7]
    return {
        "name": name,
        "vertex_shader": vertex,
        "fragment_shader": fragment,
        "uniforms": {
            "u_base_color": _uniform("vec3", color),
            "u_metallic": _uniform("float", metallic),
            "u_roughness": _uniform("float", roughness),
            "u_tint_matrix": _uniform(
                "mat3",
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            ),
            "u_enable_tint": _uniform("bool", True),
            "u_vertex_offset": _uniform("float", 0.0),
            "u_opacity": _uniform("float", opacity),
        },
        "blend": blend
        or {"enabled": False, "src_factor": "one", "dst_factor": "zero", "equation": "add"},
        "depth": depth or {"test": True, "write": True, "func": "lequal"},
    }


def _valid_manifest() -> dict[str, Any]:
    return {
        "schema_version": "1.1",
        "render": {
            "width": 160,
            "height": 120,
            "background": [0.012, 0.016, 0.024, 1.0],
            "normal_fallback": [0.0, 0.0, 1.0],
            "normal_tolerance": 1.0e-4,
        },
        "comparison": {
            "absolute": 2.0e-4,
            "relative": 2.0e-3,
            "channels": ["R", "G", "B"],
        },
        "uniform_policy": {"unused_declared": "warn"},
        "materials": [
            _material(
                "shaders/standard.vert",
                "shaders/standard.frag",
                "stone",
                [0.52, 0.61, 0.68],
                metallic=0.05,
                roughness=0.72,
            ),
            _material(
                "shaders/standard.vert",
                "shaders/standard.frag",
                "metal",
                [0.73, 0.43, 0.18],
                metallic=0.92,
                roughness=0.22,
            ),
        ],
        "channels": [
            {"name": "normal", "kind": "normal", "materials": ["stone", "metal"]},
            {"name": "depth", "kind": "depth_prepass", "materials": ["stone", "metal"]},
            {"name": "deferred", "kind": "deferred", "materials": ["stone", "metal"]},
        ],
        "scenes": [
            {
                "name": "deterministic_studio",
                "camera": {
                    "position": [1.65, 1.25, 4.25],
                    "target": [0.0, 0.0, 0.0],
                    "up": [0.0, 1.0, 0.0],
                    "fov_y_deg": 42.0,
                    "near": 0.1,
                    "far": 50.0,
                },
                "lights": [
                    {
                        "name": "key",
                        "type": "directional",
                        "direction": [0.35, -0.8, -0.5],
                        "color": [1.0, 0.95, 0.88],
                        "intensity": 1.15,
                        "radius": 10000.0,
                    },
                    {
                        "name": "fill",
                        "type": "point",
                        "position": [-3.0, 2.2, 3.5],
                        "color": [0.25, 0.42, 0.95],
                        "intensity": 2.8,
                        "radius": 8.0,
                    },
                ],
                "objects": [
                    {
                        "name": "stone_cube",
                        "mesh": "cube",
                        "material": "stone",
                        "position": [-0.62, -0.02, 0.0],
                        "rotation": [12.0, 24.0, 0.0],
                        "scale": [0.48, 0.48, 0.48],
                    },
                    {
                        "name": "metal_cube",
                        "mesh": "cube",
                        "material": "metal",
                        "position": [0.58, -0.08, 0.08],
                        "rotation": [-16.0, 36.0, 9.0],
                        "scale": [0.43, 0.57, 0.43],
                    },
                    {
                        "name": "zero_normal_probe",
                        "mesh": "zero_normal_plane",
                        "material": "stone",
                        "position": [0.0, 0.62, 0.12],
                        "rotation": [0.0, 0.0, 0.0],
                        "scale": [0.30, 0.18, 0.30],
                    },
                ],
            }
        ],
    }


def _invalid_manifest() -> dict[str, Any]:
    manifest = _valid_manifest()
    manifest["studio_note"] = "unknown fields must be retained"
    manifest["materials"] = [
        _material(
            "shaders/standard.vert",
            "shaders/standard.frag",
            "good",
            [0.45, 0.72, 0.55],
            metallic=0.1,
            roughness=0.6,
        ),
        _material(
            "shaders/standard.vert",
            "shaders/standard.frag",
            "out_of_range",
            [0.8, 0.25, 0.15],
            metallic=1.4,
            roughness=-0.2,
        ),
        _material("shaders/broken.vert", "shaders/broken.frag", "broken"),
        _material("shaders/standard.vert", "shaders/missing_uniform.frag", "missing_uniform"),
        _material(
            "shaders/standard.vert",
            "shaders/standard.frag",
            "invalid_transparent",
            opacity=0.4,
            blend={
                "enabled": True,
                "src_factor": "src_alpha",
                "dst_factor": "one_minus_src_alpha",
                "equation": "add",
            },
            depth={"test": True, "write": True, "func": "lequal"},
        ),
        _material("shaders/standard.vert", "shaders/standard.frag", "conflicted", [0.3, 0.5, 0.7]),
    ]
    manifest["channels"] = [
        {
            "name": "depth",
            "kind": "depth_prepass",
            "materials": [
                "good",
                "out_of_range",
                {
                    "name": "conflicted",
                    "uniforms": {"u_base_color": _uniform("vec3", [0.2, 0.4, 0.8])},
                },
            ],
        },
        {
            "name": "normal",
            "kind": "normal",
            "materials": [
                "good",
                "out_of_range",
                {
                    "name": "conflicted",
                    "uniforms": {"u_base_color": _uniform("vec3", [0.9, 0.2, 0.1])},
                },
            ],
        },
        {
            "name": "deferred",
            "kind": "deferred",
            "materials": [
                "good",
                "out_of_range",
                {
                    "name": "conflicted",
                    "uniforms": {"u_base_color": _uniform("vec3", [0.1, 0.9, 0.3])},
                },
            ],
        },
    ]
    manifest["scenes"][0]["objects"] = [
        {
            "name": "good_cube",
            "mesh": "cube",
            "material": "good",
            "position": [-0.6, 0.0, 0.0],
            "rotation": [10.0, 20.0, 0.0],
            "scale": [0.5, 0.5, 0.5],
        },
        {
            "name": "range_cube",
            "mesh": "cube",
            "material": "out_of_range",
            "position": [0.55, 0.0, 0.0],
            "rotation": [-10.0, 30.0, 5.0],
            "scale": [0.5, 0.5, 0.5],
        },
    ]
    return manifest


def _blend_manifest() -> dict[str, Any]:
    manifest = _valid_manifest()
    manifest["comparison"] = {
        "absolute": 1.0e-7,
        "relative": 1.0e-7,
        "channels": ["R", "G", "B"],
    }
    manifest["materials"] = [
        _material(
            "shaders/standard.vert",
            "shaders/standard.frag",
            "opaque",
            [0.42, 0.58, 0.68],
            metallic=0.05,
            roughness=0.75,
        ),
        _material(
            "shaders/standard.vert",
            "shaders/standard.frag",
            "glass",
            [0.55, 0.9, 1.0],
            metallic=0.0,
            roughness=0.1,
            opacity=0.42,
            blend={
                "enabled": True,
                "src_factor": "src_alpha",
                "dst_factor": "one_minus_src_alpha",
                "equation": "add",
            },
            depth={"test": True, "write": False, "func": "lequal"},
        ),
    ]
    manifest["channels"] = [
        {"name": "depth", "kind": "depth_prepass", "materials": ["opaque", "glass"]},
        {"name": "normal", "kind": "normal", "materials": ["opaque", "glass"]},
        {"name": "deferred", "kind": "deferred", "materials": ["opaque", "glass"]},
    ]
    manifest["scenes"][0]["objects"] = [
        {
            "name": "opaque_cube",
            "mesh": "cube",
            "material": "opaque",
            "position": [-0.45, 0.0, 0.0],
            "rotation": [8.0, 25.0, 0.0],
            "scale": [0.55, 0.55, 0.55],
        },
        {
            "name": "glass_cube",
            "mesh": "cube",
            "material": "glass",
            "position": [0.35, 0.05, 0.55],
            "rotation": [-12.0, 35.0, 7.0],
            "scale": [0.5, 0.5, 0.5],
        },
    ]
    return manifest


def sample_manifest(kind: str = "valid") -> dict[str, Any]:
    if kind == "valid":
        return _valid_manifest()
    if kind == "invalid":
        return _invalid_manifest()
    if kind == "blend":
        return _blend_manifest()
    raise ValueError(f"Unknown sample kind: {kind}")


def sample_shader_sources() -> dict[str, str]:
    return {
        "standard.vert": STANDARD_VERTEX,
        "standard.frag": STANDARD_FRAGMENT,
        "broken.vert": BROKEN_VERTEX,
        "broken.frag": BROKEN_FRAGMENT,
        "missing_uniform.frag": MISSING_FRAGMENT,
    }


def generate_sample(directory: str | Path, kind: str = "valid") -> Path:
    target = Path(directory).resolve()
    shader_dir = target / "shaders"
    shader_dir.mkdir(parents=True, exist_ok=True)
    for name, source in sample_shader_sources().items():
        with (shader_dir / name).open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(source)
    manifest = sample_manifest(kind)
    manifest_path = target / f"{kind}_manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest_path
