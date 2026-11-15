#version 330 core

in vec3 vViewPosition;
in vec3 vViewNormal;

uniform vec3 uAlbedo;
uniform float uRoughness;
uniform float uMetallic;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormal;
layout(location = 2) out vec4 outMaterial;

void main() {
    outPosition = vec4(vViewPosition, 1.0);
    outNormal = vec4(normalize(vViewNormal), clamp(uRoughness, 0.04, 1.0));
    outMaterial = vec4(uAlbedo, clamp(uMetallic, 0.0, 1.0));
}

