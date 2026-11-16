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
uniform float u_missing_tint;
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

layout(location = 0) out vec4 frag_world_position;
layout(location = 1) out vec4 frag_world_normal;
layout(location = 2) out vec4 frag_albedo;
layout(location = 3) out vec4 frag_metallic_roughness;
layout(location = 4) out vec4 frag_color;

void main() {
    float normal_length_squared = dot(v_world_normal, v_world_normal);
    bool normal_valid = normal_length_squared > 1.0e-10;
    vec3 normal = normal_valid
        ? v_world_normal * inversesqrt(normal_length_squared)
        : normalize(u_normal_fallback);
    vec3 tinted = u_enable_tint ? u_tint_matrix * u_base_color : u_base_color;
    vec3 albedo = clamp(tinted + vec3(u_missing_tint), 0.0, 1.0);
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
}
