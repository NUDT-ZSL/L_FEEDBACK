#version 410 core

in vec3 v_world_pos;
in vec3 v_world_normal;
in vec2 v_uv;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec4 out_normal;
layout(location = 2) out vec4 out_depth;
layout(location = 3) out vec4 out_object_id;

uniform vec3 u_base_color;
uniform float u_metallic;
uniform float u_roughness;
uniform float u_emissive_strength;
uniform float u_uv_scale;
uniform vec3 u_camera_pos;
uniform vec3 u_light_pos;
uniform vec3 u_light_color;
uniform float u_object_id;
uniform float u_material_index;
uniform float u_shape_id;
uniform sampler2D u_albedo_tex;
uniform mat4 u_view;

const float PI = 3.14159265358979323846;

float distribution_ggx(vec3 n, vec3 h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float n_dot_h = max(dot(n, h), 0.0);
    float denominator = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denominator * denominator, 1e-6);
}

float geometry_schlick_ggx(float n_dot_x, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return n_dot_x / max(n_dot_x * (1.0 - k) + k, 1e-6);
}

float geometry_smith(vec3 n, vec3 v, vec3 l, float roughness) {
    return geometry_schlick_ggx(max(dot(n, v), 0.0), roughness)
        * geometry_schlick_ggx(max(dot(n, l), 0.0), roughness);
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

void main() {
    vec3 n = normalize(v_world_normal);
    vec3 v = normalize(u_camera_pos - v_world_pos);
    vec3 l = normalize(u_light_pos - v_world_pos);
    vec3 h = normalize(v + l);

    vec4 sampled = texture(u_albedo_tex, v_uv * u_uv_scale);
    vec3 albedo = sampled.rgb * u_base_color;
    float metallic = clamp(u_metallic, 0.0, 1.0);
    float roughness = clamp(u_roughness, 0.04, 1.0);
    vec3 f0 = mix(vec3(0.04), albedo, metallic);

    float ndf = distribution_ggx(n, h, roughness);
    float geometry = geometry_smith(n, v, l, roughness);
    vec3 fresnel = fresnel_schlick(max(dot(h, v), 0.0), f0);
    vec3 specular = (ndf * geometry * fresnel)
        / max(4.0 * max(dot(n, v), 0.0) * max(dot(n, l), 0.0), 0.001);
    vec3 diffuse = (vec3(1.0) - fresnel) * (1.0 - metallic) * albedo / PI;
    vec3 color = (diffuse + specular) * u_light_color * max(dot(n, l), 0.0);

    float rim = pow(1.0 - max(dot(n, v), 0.0), 5.0);
    color += vec3(0.06, 0.08, 0.11) * rim * (0.5 + 0.5 * roughness);
    color += albedo * u_emissive_strength * 0.08;
    color = color / (color + vec3(1.0));

    out_color = vec4(max(color, vec3(0.0)), 1.0);
    out_normal = vec4(n * 0.5 + 0.5, 1.0);
    float view_depth = max(-(u_view * vec4(v_world_pos, 1.0)).z, 0.0);
    out_depth = vec4(view_depth / 100.0, gl_FragCoord.z, view_depth, 1.0);
    out_object_id = vec4(u_object_id, u_material_index, u_shape_id, 1.0);
}

