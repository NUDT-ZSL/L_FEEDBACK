#version 460

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inAlbedo;

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec3 albedo;

void main() {
    worldPosition = inPosition;
    worldNormal = inNormal;
    albedo = inAlbedo;
    gl_Position = ubo.viewProj * vec4(inPosition, 1.0);
}
