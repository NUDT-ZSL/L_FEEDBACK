#version 450 core

layout(location = 0) out vec4 gPosition;
layout(location = 1) out vec4 gNormal;
layout(location = 2) out vec4 gAlbedo;
layout(location = 3) out vec4 gMaterial;

in vec3 vWorldPosition;
in vec3 vWorldNormal;
in float vViewDepth;

uniform vec3 uAlbedo;
uniform float uRoughness;
uniform float uMetallic;
uniform float uObjectId;

void main() {
    gPosition = vec4(vWorldPosition, vViewDepth);
    gNormal = vec4(normalize(vWorldNormal), uObjectId);
    gAlbedo = vec4(uAlbedo, uRoughness);
    gMaterial = vec4(uMetallic, 0.0, 0.0, 1.0);
}

