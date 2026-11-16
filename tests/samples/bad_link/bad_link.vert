#version 450 core

out vec3 vValue;

void main() {
    vec2 position = vec2(
        gl_VertexID == 1 ? 3.0 : -1.0,
        gl_VertexID == 2 ? 3.0 : -1.0);
    vValue = vec3(1.0);
    gl_Position = vec4(position, 0.0, 1.0);
}
