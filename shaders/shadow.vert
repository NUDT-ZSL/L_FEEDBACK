#version 460

layout(push_constant) uniform ShadowPush {
    mat4 lightViewProj;
} pushData;

layout(location = 0) in vec3 inPosition;

void main() {
    gl_Position = pushData.lightViewProj * vec4(inPosition, 1.0);
}
