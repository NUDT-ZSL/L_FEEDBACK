#version 450 core

void main() {
    vec2 position = vec2(
        gl_VertexID == 1 ? 3.0 : -1.0,
        gl_VertexID == 2 ? 3.0 : -1.0);
    float z = 0.2 + float(gl_InstanceID) * 0.3;
    gl_Position = vec4(position, z, 1.0);
}
