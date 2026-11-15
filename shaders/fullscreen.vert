#version 460

layout(location = 0) out vec2 uv;

void main() {
    const vec2 position = vec2(
        float((gl_VertexIndex << 1) & 2),
        float(gl_VertexIndex & 2));
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
    uv = position;
}
