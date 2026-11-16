#version 330 core

in vec3 vWorldPosition;
in vec3 vWorldNormal;
flat in float vMaterial;

layout(location = 0) out vec4 outPosition;
layout(location = 1) out vec4 outNormalMaterial;

void main()
{
    outPosition = vec4(vWorldPosition, 1.0);
    outNormalMaterial = vec4(normalize(vWorldNormal), vMaterial);
}

