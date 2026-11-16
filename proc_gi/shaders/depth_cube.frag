#version 330 core

in vec3 vWorldPosition;

uniform vec3 uLightPosition;
uniform float uLightRange;

out float outDistance;

void main()
{
    outDistance = clamp(length(vWorldPosition - uLightPosition) / uLightRange, 0.0, 1.0);
}

