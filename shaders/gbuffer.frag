#version 450 core

#include "gbuffer_layout.glsl"

layout(location = GBUF_WORLD_POSITION) out vec4 outWorldPosition;
layout(location = GBUF_VIEW_NORMAL) out vec4 outViewNormal;
layout(location = GBUF_ALBEDO) out vec4 outAlbedo;
layout(location = GBUF_SURFACE) out vec4 outSurface;

in vec3 vWorldPosition;
in vec3 vWorldNormal;
in vec3 vWorldTangent;
in vec2 vUv;
in float vViewDepth;

flat in vec3 vAlbedo;
flat in float vRoughness;
flat in float vMetallic;
flat in vec4 vLayers;
flat in vec4 vMaterialParams;
flat in float vObjectId;

uniform sampler2DArray uMaterialTextures;
uniform mat4 uView;
uniform int uMaterialPreset;

void main() {
    const vec3 geometricNormal = normalize(vWorldNormal);
    const vec3 tangent = normalize(vWorldTangent -
                                   geometricNormal * dot(vWorldTangent, geometricNormal));
    const vec3 bitangent = normalize(cross(geometricNormal, tangent));

    const float uvScale = vMaterialParams.x;
    const vec2 uv = vUv * uvScale;
    const int baseLayer = int(vLayers.x + 0.5);
    const int normalLayer = int(vLayers.y + 0.5);
    const int roughMetalLayer = int(vLayers.z + 0.5);

    const vec3 base = texture(uMaterialTextures,
                              vec3(uv, float(baseLayer))).rgb;
    const vec3 detail = texture(uMaterialTextures,
                                vec3(uv * 2.37 + vec2(0.17, 0.41),
                                     float(baseLayer))).rgb;

    // The five presets only change how the three procedural layers
    // (base color, normal, roughness/metal) are composed into G-Buffer values.
    float detailBlend = 0.0;
    float normalBlend = 0.0;
    float roughMetalBlend = 0.0;
    if (uMaterialPreset == 1) {
        normalBlend = 0.78;
    } else if (uMaterialPreset == 2) {
        roughMetalBlend = 1.0;
    } else if (uMaterialPreset == 3) {
        detailBlend = 0.32;
        normalBlend = 0.52;
    } else if (uMaterialPreset == 4) {
        detailBlend = 0.48;
        normalBlend = 1.0;
        roughMetalBlend = 1.0;
    }

    const vec3 composedAlbedo = vAlbedo *
        mix(base, base * detail * 1.62, detailBlend);
    const vec3 tangentNormal = normalize(
        texture(uMaterialTextures,
                vec3(uv, float(normalLayer))).xyz * 2.0 - 1.0);
    const vec3 normal = normalize(mix(geometricNormal,
        normalize(tangent * tangentNormal.x +
                  bitangent * tangentNormal.y +
                  geometricNormal * tangentNormal.z),
        normalBlend * vMaterialParams.z));
    const vec4 roughMetal = texture(uMaterialTextures,
                                    vec3(uv * 1.31, float(roughMetalLayer)));
    const float roughness = mix(vRoughness, roughMetal.r, roughMetalBlend);
    const float metallic = mix(vMetallic, roughMetal.g, roughMetalBlend);

    const mat3 viewNormalMatrix = mat3(uView);
    outWorldPosition = vec4(vWorldPosition, vViewDepth);
    outViewNormal = vec4(normalize(viewNormalMatrix * normal), vObjectId);
    outAlbedo = vec4(composedAlbedo, roughMetal.b);
    outSurface = vec4(roughness, metallic,
                      float(uMaterialPreset) / 4.0, vObjectId);
}
