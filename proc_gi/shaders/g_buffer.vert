#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in float aMaterial;

uniform mat4 uModel;
uniform mat4 uViewProjection;

out vec3 vWorldPosition;
out vec3 vWorldNormal;
flat out float vMaterial;

void main()
{
    vec4 world = uModel * vec4(aPosition, 1.0);
    vWorldPosition = world.xyz;
    vWorldNormal = mat3(uModel) * aNormal;
    vMaterial = aMaterial;
    gl_Position = uViewProjection * world;
}

