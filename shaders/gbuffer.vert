#version 450 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aTangent;
layout(location = 3) in vec2 aUv;

layout(location = 4) in mat4 aModel;
layout(location = 8) in vec4 aAlbedoRoughness;
layout(location = 9) in vec4 aMaterialLayers;
layout(location = 10) in vec4 aMaterialParams;

uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vWorldPosition;
out vec3 vWorldNormal;
out vec3 vWorldTangent;
out vec2 vUv;
out float vViewDepth;

flat out vec3 vAlbedo;
flat out float vRoughness;
flat out float vMetallic;
flat out vec4 vLayers;
flat out vec4 vMaterialParams;
flat out float vObjectId;

void main() {
    const vec4 world = aModel * vec4(aPosition, 1.0);
    const vec4 view = uView * world;
    gl_Position = uProjection * view;

    vWorldPosition = world.xyz;
    vWorldNormal = normalize(mat3(aModel) * aNormal);
    vWorldTangent = normalize(mat3(aModel) * aTangent);
    vUv = aUv;
    vViewDepth = -view.z;

    vAlbedo = aAlbedoRoughness.rgb;
    vRoughness = aAlbedoRoughness.a;
    vMetallic = aMaterialParams.y;
    vLayers = aMaterialLayers;
    vMaterialParams = aMaterialParams;
    vObjectId = aMaterialLayers.w;
}
