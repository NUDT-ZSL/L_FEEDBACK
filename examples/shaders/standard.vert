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
