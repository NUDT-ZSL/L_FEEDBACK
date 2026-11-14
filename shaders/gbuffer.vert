#version 450 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vWorldPosition;
out vec3 vWorldNormal;
out float vViewDepth;

void main() {
    vec4 world = uModel * vec4(aPosition, 1.0);
    vec4 view = uView * world;
    vWorldPosition = world.xyz;
    vWorldNormal = normalize(mat3(uModel) * aNormal);
    vViewDepth = -view.z;
    gl_Position = uProjection * view;
}

