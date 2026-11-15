#version 410 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

out vec3 v_world_pos;
out vec3 v_world_normal;
out vec2 v_uv;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_world_pos = world_pos.xyz;
    v_world_normal = mat3(transpose(inverse(u_model))) * in_normal;
    v_uv = in_uv;
    gl_Position = u_proj * u_view * world_pos;
}

