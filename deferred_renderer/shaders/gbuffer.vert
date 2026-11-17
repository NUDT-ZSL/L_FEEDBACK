#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_tangent;
layout(location = 3) in vec2 in_uv;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform mat3 u_normal_matrix;
uniform float u_near;
uniform float u_far;
uniform vec2 u_uv_scale;
uniform vec2 u_uv_offset;

out VS_OUT {
    vec3 world_position;
    vec3 world_normal;
    vec3 world_tangent;
    float tangent_sign;
    vec2 uv;
    float linear_depth;
} vs_out;

void main() {
    vec4 world_position = u_model * vec4(in_position, 1.0);
    vec4 view_position = u_view * world_position;
    gl_Position = u_projection * view_position;

    float ndc_depth = gl_Position.z / gl_Position.w;
    vs_out.linear_depth = (2.0 * u_near * u_far)
        / (u_far + u_near - ndc_depth * (u_far - u_near));
    vs_out.world_position = world_position.xyz;
    vs_out.world_normal = normalize(u_normal_matrix * in_normal);
    vs_out.world_tangent = normalize(mat3(u_model) * in_tangent.xyz);
    vs_out.tangent_sign = in_tangent.w;
    vs_out.uv = in_uv * u_uv_scale + u_uv_offset;
}

