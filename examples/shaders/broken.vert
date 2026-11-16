#version 330 core
in vec3 in_position;
uniform mat4 u_model;
void main() {
    vec4 world = u_model * vec4(in_position, 1.0)
    gl_Position = world
}
