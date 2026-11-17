#version 330 core

#define MAX_TILE_LIGHTS 32
#define MAX_MATERIALS 32

uniform sampler2D u_gbuffer_albedo_roughness;
uniform sampler2D u_gbuffer_normal_side;
uniform sampler2D u_gbuffer_linear_depth;
uniform usampler2D u_gbuffer_material_id;
uniform usampler2D u_tile_lights;

uniform mat4 u_inv_projection;
uniform mat4 u_inv_view;
uniform vec3 u_camera_position;
uniform vec3 u_background;
uniform float u_ambient_strength;
uniform vec2 u_resolution;
uniform int u_tile_size;
uniform int u_tiles_x;
uniform int u_light_count;

uniform vec4 u_light_position_radius[32];
uniform vec4 u_light_color_intensity[32];
uniform float u_material_back_roughness_scale[MAX_MATERIALS + 1];
uniform float u_material_back_roughness_bias[MAX_MATERIALS + 1];

out vec4 out_hdr;

const float PI = 3.14159265358979323846;

vec3 reconstruct_world(vec2 uv, float linear_depth) {
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 near_h = u_inv_projection * vec4(ndc, -1.0, 1.0);
    vec3 ray = near_h.xyz / near_h.w;
    ray /= -ray.z;
    vec3 view_position = ray * linear_depth;
    return (u_inv_view * vec4(view_position, 1.0)).xyz;
}

uint packed_component(uvec4 value, int component) {
    if (component == 0) return value.x;
    if (component == 1) return value.y;
    if (component == 2) return value.z;
    return value.w;
}

float distribution_ggx(float n_dot_h, float alpha) {
    float alpha_sq = alpha * alpha;
    float denominator = n_dot_h * n_dot_h * (alpha_sq - 1.0) + 1.0;
    return alpha_sq / max(PI * denominator * denominator, 1e-7);
}

float geometry_schlick_ggx(float n_dot_v, float alpha) {
    float k = alpha * 0.5;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 1e-7);
}

void main() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 albedo_roughness = texelFetch(u_gbuffer_albedo_roughness, pixel, 0);
    vec4 normal_side = texelFetch(u_gbuffer_normal_side, pixel, 0);
    float linear_depth = texelFetch(u_gbuffer_linear_depth, pixel, 0).r;
    uint material_id = texelFetch(u_gbuffer_material_id, pixel, 0).r;

    if (linear_depth <= 0.0) {
        out_hdr = vec4(u_background, 1.0);
        return;
    }

    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec3 world_position = reconstruct_world(uv, linear_depth);
    vec3 view_direction = normalize(u_camera_position - world_position);
    vec3 normal = normalize(normal_side.xyz);
    if (dot(normal, view_direction) < 0.0) {
        normal = -normal;
    }

    int material_index = int(min(material_id, uint(MAX_MATERIALS)));
    float roughness = clamp(albedo_roughness.a, 0.04, 1.0);
    if (normal_side.w < 0.0) {
        roughness = clamp(
            roughness * u_material_back_roughness_scale[material_index]
            + u_material_back_roughness_bias[material_index],
            0.04,
            1.0
        );
    }

    int tile_x = pixel.x / u_tile_size;
    int tile_y = pixel.y / u_tile_size;
    int tile_index = tile_y * u_tiles_x + tile_x;
    uint light_count = texelFetch(u_tile_lights, ivec2(0, tile_index), 0).r;

    vec3 color = vec3(0.0);
    float n_dot_v = max(dot(normal, view_direction), 1e-4);
    vec3 f0 = vec3(0.04);
    float alpha = max(roughness * roughness, 0.0025);

    for (int slot = 0; slot < MAX_TILE_LIGHTS; ++slot) {
        if (slot >= int(light_count)) {
            break;
        }
        int word_index = slot + 1;
        uvec4 packed_light = texelFetch(
            u_tile_lights,
            ivec2(word_index / 4, tile_index),
            0
        );
        uint packed_light_index = packed_component(packed_light, word_index & 3);
        int light_index = int(packed_light_index);
        if (light_index >= u_light_count) {
            continue;
        }

        vec4 position_radius = u_light_position_radius[light_index];
        vec3 to_light = position_radius.xyz - world_position;
        float distance_sq = dot(to_light, to_light);
        float radius_sq = position_radius.w * position_radius.w;
        if (distance_sq > radius_sq) {
            continue;
        }
        float distance_to_light = sqrt(max(distance_sq, 1e-8));
        vec3 light_direction = to_light / distance_to_light;
        float n_dot_l = max(dot(normal, light_direction), 0.0);
        if (n_dot_l <= 0.0) {
            continue;
        }

        vec3 halfway = normalize(view_direction + light_direction);
        float n_dot_h = max(dot(normal, halfway), 0.0);
        float v_dot_h = max(dot(view_direction, halfway), 0.0);
        float fresnel = pow(1.0 - v_dot_h, 5.0);
        vec3 fresnel_term = f0 + (vec3(1.0) - f0) * fresnel;
        float distribution = distribution_ggx(n_dot_h, alpha);
        float geometry = geometry_schlick_ggx(n_dot_l, alpha)
            * geometry_schlick_ggx(n_dot_v, alpha);
        vec3 specular = (distribution * geometry * fresnel_term)
            / max(4.0 * n_dot_l * n_dot_v, 1e-5);
        vec3 diffuse = (vec3(1.0) - fresnel_term) * albedo_roughness.rgb / PI;

        float attenuation_fraction = clamp(
            1.0 - distance_to_light / position_radius.w,
            0.0,
            1.0
        );
        float attenuation = attenuation_fraction * attenuation_fraction;
        vec3 radiance = u_light_color_intensity[light_index].rgb
            * u_light_color_intensity[light_index].a
            * attenuation
            / max(distance_sq, 0.02);
        color += (diffuse + specular) * radiance * n_dot_l;
    }

    color += albedo_roughness.rgb * u_background * u_ambient_strength;
    out_hdr = vec4(color, 1.0);
}

