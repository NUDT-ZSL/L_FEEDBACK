"""GLSL 4.50 shader sources for the deferred pipeline."""

GBUFFER_VERTEX = r"""#version 450 core
layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_normal;
layout(location=2) in vec2 a_uv;
layout(location=3) in vec3 a_tangent;
layout(location=4) in float a_bitangent_sign;

uniform mat4 u_model;
uniform mat4 u_view_proj;
uniform mat3 u_normal_matrix;

out vec3 v_world_pos;
out vec3 v_world_normal;
out vec2 v_uv;
out vec3 v_world_tangent;
out float v_bitangent_sign;

void main() {
    vec4 world = u_model * vec4(a_position, 1.0);
    v_world_pos = world.xyz;
    v_world_normal = normalize(u_normal_matrix * a_normal);
    v_world_tangent = normalize(u_normal_matrix * a_tangent);
    v_bitangent_sign = a_bitangent_sign;
    v_uv = a_uv;
    gl_Position = u_view_proj * world;
}
"""

GBUFFER_FRAGMENT = r"""#version 450 core
in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;
in vec3 v_world_tangent;
in float v_bitangent_sign;

layout(location=0) out vec4 g_position;
layout(location=1) out vec4 g_normal;
layout(location=2) out vec4 g_albedo;
layout(location=3) out vec4 g_material;

uniform vec3 u_camera_position;
uniform vec3 u_albedo;
uniform float u_metallic;
uniform float u_roughness;
uniform float u_normal_strength;
uniform int u_has_normal_map;
uniform sampler2D u_normal_map;

void main() {
    vec3 N = normalize(v_world_normal);
    if (u_has_normal_map != 0) {
        vec3 T = v_world_tangent - N * dot(N, v_world_tangent);
        if (length(T) < 1e-5) {
            vec3 axis = abs(N.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
            T = cross(axis, N);
        }
        T = normalize(T);
        vec3 B = cross(N, T) * v_bitangent_sign;
        vec3 tangent_normal = texture(u_normal_map, v_uv).xyz * 2.0 - 1.0;
        tangent_normal.xy *= u_normal_strength;
        tangent_normal = normalize(tangent_normal);
        N = normalize(T * tangent_normal.x + B * tangent_normal.y + N * tangent_normal.z);
    }
    g_position = vec4(v_world_pos, distance(v_world_pos, u_camera_position));
    g_normal = vec4(N, 1.0);
    g_albedo = vec4(clamp(u_albedo, 0.0, 1.0), 1.0);
    g_material = vec4(clamp(u_metallic, 0.0, 1.0), clamp(u_roughness, 0.03, 1.0), 0.0, 1.0);
}
"""

DEPTH_VERTEX = r"""#version 450 core
layout(location=0) in vec3 a_position;
uniform mat4 u_model;
uniform mat4 u_light_view_proj;
void main() {
    gl_Position = u_light_view_proj * u_model * vec4(a_position, 1.0);
}
"""

DEPTH_FRAGMENT = r"""#version 450 core
void main() {}
"""

FULLSCREEN_VERTEX = r"""#version 450 core
out vec2 v_uv;
void main() {
    const vec2 positions[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    vec2 p = positions[gl_VertexID];
    v_uv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

DIRECT_LIGHT_FRAGMENT = r"""#version 450 core
in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_g_position;
uniform sampler2D u_g_normal;
uniform sampler2D u_g_albedo;
uniform sampler2D u_g_material;

uniform vec3 u_camera_position;
uniform int u_light_type;       // 0 directional, 1 point, 2 spot
uniform vec3 u_light_position;
uniform vec3 u_light_direction; // rays travel toward the scene
uniform vec3 u_light_color;
uniform float u_light_intensity;
uniform float u_light_range;
uniform float u_cos_inner;
uniform float u_cos_outer;

uniform int u_shadow_kind;       // 0 none, 1 2D, 2 cube
uniform sampler2DShadow u_shadow_map;
uniform samplerCubeShadow u_shadow_cube;
uniform mat4 u_light_vp;
uniform vec2 u_shadow_texel;
uniform float u_shadow_bias;
uniform float u_shadow_normal_bias;
uniform int u_filter_radius;
uniform int u_filter_taps;

const float PI = 3.14159265358979323846;

float distribution_ggx(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denom * denom, 1e-7);
}

float geometry_schlick_ggx(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return NdotV / max(NdotV * (1.0 - k) + k, 1e-7);
}

float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometry_schlick_ggx(max(dot(N, V), 0.0), roughness)
         * geometry_schlick_ggx(max(dot(N, L), 0.0), roughness);
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

float sample_shadow_2d(vec2 uv, float reference) {
    if (u_filter_taps == 1) {
        return texture(u_shadow_map, vec3(uv, reference));
    }
    float sum = 0.0;
    float count = 0.0;
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            if (abs(x) > u_filter_radius || abs(y) > u_filter_radius) {
                continue;
            }
            vec2 offset = vec2(float(x), float(y)) * u_shadow_texel;
            sum += texture(u_shadow_map, vec3(uv + offset, reference));
            count += 1.0;
        }
    }
    return count > 0.0 ? sum / count : 1.0;
}

float sample_shadow_cube(vec3 direction, float reference) {
    if (u_filter_taps == 1) {
        return texture(u_shadow_cube, vec4(direction, reference));
    }
    vec3 L = normalize(direction);
    vec3 up = abs(L.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, L));
    vec3 bitangent = cross(L, tangent);
    float sum = 0.0;
    float count = 0.0;
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            if (abs(x) > u_filter_radius || abs(y) > u_filter_radius) {
                continue;
            }
            vec3 offset = tangent * float(x) * u_shadow_texel.x
                        + bitangent * float(y) * u_shadow_texel.y;
            sum += texture(u_shadow_cube, vec4(normalize(L + offset), reference));
            count += 1.0;
        }
    }
    return count > 0.0 ? sum / count : 1.0;
}

float shadow_visibility(vec3 P, vec3 N, vec3 L, float NdotL) {
    if (u_shadow_kind == 0) {
        return 1.0;
    }
    float normal_offset = u_shadow_normal_bias * (1.0 - clamp(NdotL, 0.0, 1.0));
    if (u_shadow_kind == 2) {
        vec3 to_fragment = P - u_light_position;
        float distance_value = length(to_fragment);
        float reference = distance_value / u_light_range - u_shadow_bias - normal_offset;
        if (distance_value >= u_light_range) {
            return 1.0;
        }
        return sample_shadow_cube(to_fragment, reference);
    }
    vec4 projected = u_light_vp * vec4(P + N * normal_offset * 0.02, 1.0);
    if (projected.w <= 0.0) {
        return 1.0;
    }
    vec3 coord = projected.xyz / projected.w;
    coord = coord * 0.5 + 0.5;
    if (coord.x <= 0.0 || coord.x >= 1.0 || coord.y <= 0.0 || coord.y >= 1.0 || coord.z >= 1.0) {
        return 1.0;
    }
    return sample_shadow_2d(coord.xy, coord.z - u_shadow_bias - normal_offset);
}

void main() {
    vec4 normal_sample = texture(u_g_normal, v_uv);
    if (normal_sample.a < 0.5) {
        discard;
    }
    vec3 P = texture(u_g_position, v_uv).xyz;
    vec3 N = normalize(normal_sample.xyz);
    vec3 albedo = texture(u_g_albedo, v_uv).rgb;
    vec4 material = texture(u_g_material, v_uv);
    float metallic = material.r;
    float roughness = clamp(material.g, 0.045, 1.0);
    vec3 V = normalize(u_camera_position - P);

    vec3 L;
    float attenuation = 1.0;
    if (u_light_type == 0) {
        L = normalize(-u_light_direction);
    } else {
        vec3 delta = u_light_position - P;
        float distance_value = max(length(delta), 1e-5);
        L = delta / distance_value;
        float range_window = clamp(1.0 - pow(distance_value / u_light_range, 4.0), 0.0, 1.0);
        attenuation = range_window * range_window / max(distance_value * distance_value, 1e-4);
        if (u_light_type == 2) {
            float cos_angle = dot(normalize(-u_light_direction), -L);
            float spot = clamp((cos_angle - u_cos_outer) / max(u_cos_inner - u_cos_outer, 1e-5), 0.0, 1.0);
            attenuation *= spot;
        }
    }
    float NdotL = max(dot(N, L), 0.0);
    if (NdotL <= 0.0 || attenuation <= 0.0) {
        discard;
    }
    float NdotV = max(dot(N, V), 1e-4);
    vec3 H = normalize(V + L);
    float HdotV = max(dot(H, V), 0.0);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    float D = distribution_ggx(N, H, roughness);
    float G = geometry_smith(N, V, L, roughness);
    vec3 F = fresnel_schlick(HdotV, F0);
    vec3 specular = (D * G * F) / max(4.0 * NdotV * NdotL, 1e-4);
    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
    vec3 radiance = u_light_color * u_light_intensity * attenuation;
    float visibility = shadow_visibility(P, N, L, NdotL);
    vec3 color = (kD * albedo / PI + specular) * radiance * NdotL * visibility;
    out_color = vec4(max(color, vec3(0.0)), 1.0);
}
"""

INDIRECT_FRAGMENT = r"""#version 450 core
in vec2 v_uv;
out vec4 out_color;

uniform sampler2D u_g_position;
uniform sampler2D u_g_normal;
uniform sampler2D u_g_albedo;
uniform sampler2D u_g_material;
uniform vec3 u_camera_position;
uniform vec3 u_sky_color;
uniform vec3 u_ground_color;
uniform float u_probe_intensity;
uniform int u_indirect_enabled;

const float PI = 3.14159265358979323846;

vec3 fresnel_schlick_roughness(float cos_theta, vec3 f0, float roughness) {
    return f0 + (max(vec3(1.0 - roughness), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

void main() {
    vec4 normal_sample = texture(u_g_normal, v_uv);
    if (normal_sample.a < 0.5 || u_indirect_enabled == 0) {
        discard;
    }
    vec3 P = texture(u_g_position, v_uv).xyz;
    vec3 N = normalize(normal_sample.xyz);
    vec3 albedo = texture(u_g_albedo, v_uv).rgb;
    vec4 material = texture(u_g_material, v_uv);
    float metallic = material.r;
    float roughness = clamp(material.g, 0.045, 1.0);
    vec3 V = normalize(u_camera_position - P);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 F = fresnel_schlick_roughness(max(dot(N, V), 0.0), F0, roughness);
    vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

    float hemisphere = N.y * 0.5 + 0.5;
    vec3 irradiance = mix(u_ground_color, u_sky_color, hemisphere) * u_probe_intensity;
    vec3 diffuse = kD * albedo * irradiance;

    vec3 R = reflect(-V, N);
    vec3 reflected = mix(u_ground_color, u_sky_color, R.y * 0.5 + 0.5) * u_probe_intensity;
    vec3 specular = reflected * F * mix(0.10, 0.72, 1.0 - roughness);
    out_color = vec4(max(diffuse + specular, vec3(0.0)), 1.0);
}
"""

COMPOSITE_FRAGMENT = r"""#version 450 core
in vec2 v_uv;
out vec4 out_color;
uniform sampler2D u_direct;
uniform sampler2D u_indirect;
void main() {
    vec4 direct = texture(u_direct, v_uv);
    vec4 indirect = texture(u_indirect, v_uv);
    out_color = vec4(direct.rgb + indirect.rgb, 1.0);
}
"""

DOWNSAMPLE_FRAGMENT = r"""#version 450 core
in vec2 v_uv;
out vec4 out_color;
uniform sampler2D u_source;
uniform int u_ssaa;
void main() {
    ivec2 base = ivec2(gl_FragCoord.xy) * u_ssaa;
    vec4 sum = vec4(0.0);
    float count = 0.0;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            if (x >= u_ssaa || y >= u_ssaa) {
                continue;
            }
            sum += texelFetch(u_source, base + ivec2(x, y), 0);
            count += 1.0;
        }
    }
    out_color = sum / max(count, 1.0);
}
"""

TONEMAP_FRAGMENT = r"""#version 450 core
in vec2 v_uv;
out vec4 out_color;
uniform sampler2D u_source;
uniform int u_ssaa;
uniform float u_exposure;
uniform int u_operator; // 0 ACES, 1 Reinhard

vec3 aces_fitted(vec3 x) {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

void main() {
    ivec2 base = ivec2(gl_FragCoord.xy) * u_ssaa;
    vec3 sum = vec3(0.0);
    float count = 0.0;
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            if (x >= u_ssaa || y >= u_ssaa) {
                continue;
            }
            sum += texelFetch(u_source, base + ivec2(x, y), 0).rgb;
            count += 1.0;
        }
    }
    vec3 hdr = sum / max(count, 1.0);
    vec3 mapped = u_operator == 0 ? aces_fitted(hdr * u_exposure) : (hdr * u_exposure) / (1.0 + hdr * u_exposure);
    mapped = clamp(mapped, 0.0, 1.0);
    out_color = vec4(pow(mapped, vec3(1.0 / 2.2)), 1.0);
}
"""
