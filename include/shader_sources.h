#pragma once

namespace dr::shaders {

inline constexpr const char* kGBufferVertex = R"GLSL(
#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uViewProjection;

out vec3 vWorldPosition;
out vec3 vWorldNormal;

void main() {
    vec4 world = uModel * vec4(aPosition, 1.0);
    vWorldPosition = world.xyz;
    vWorldNormal = normalize(mat3(uModel) * aNormal);
    gl_Position = uViewProjection * world;
}
)GLSL";

inline constexpr const char* kGBufferFragment = R"GLSL(
#version 330 core
layout(location = 0) out vec4 oWorld;
layout(location = 1) out vec4 oNormal;
layout(location = 2) out vec4 oAlbedo;

in vec3 vWorldPosition;
in vec3 vWorldNormal;

uniform vec3 uAlbedo;
uniform float uSpecular;

void main() {
    oWorld = vec4(vWorldPosition, 1.0);
    oNormal = vec4(normalize(vWorldNormal), 1.0);
    oAlbedo = vec4(uAlbedo, clamp(uSpecular, 0.0, 1.0));
}
)GLSL";

inline constexpr const char* kFullscreenVertex = R"GLSL(
#version 330 core
out vec2 vUv;

void main() {
    vec2 position = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    vUv = position;
    gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

inline constexpr const char* kLightingFragment = R"GLSL(
#version 330 core
layout(std140) uniform LightBlock {
    vec4 uAmbientExposure;
    vec4 uCounts;
    vec4 uPointPositionRadius[96];
    vec4 uPointColorIntensity[96];
    vec4 uDirection[16];
    vec4 uDirectionColorIntensity[16];
};

uniform sampler2D uWorldTexture;
uniform sampler2D uNormalTexture;
uniform sampler2D uAlbedoTexture;
uniform vec3 uCameraPosition;

in vec2 vUv;
out vec4 oColor;

vec3 tonemapAces(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

vec3 shadePoint(vec3 world, vec3 normal, vec3 viewDirection, vec3 albedo, float specular,
                vec3 lightPosition, vec3 lightColor, float intensity, float radius) {
    vec3 toLight = lightPosition - world;
    float distanceSquared = max(dot(toLight, toLight), 1.0e-6);
    float distanceToLight = sqrt(distanceSquared);
    if (distanceToLight >= radius) {
        return vec3(0.0);
    }

    vec3 lightDirection = toLight / distanceToLight;
    float diffuse = max(dot(normal, lightDirection), 0.0);
    vec3 halfway = normalize(lightDirection + viewDirection);
    float specularTerm = pow(max(dot(normal, halfway), 0.0), 48.0) * specular;
    float falloff = 1.0 - distanceToLight / radius;
    float attenuation = falloff * falloff / (1.0 + 0.12 * distanceSquared);
    return lightColor * intensity * attenuation * (albedo * diffuse + vec3(specularTerm));
}

vec3 shadeDirectional(vec3 normal, vec3 viewDirection, vec3 albedo, float specular,
                      vec3 direction, vec3 lightColor, float intensity) {
    vec3 lightDirection = normalize(-direction);
    float diffuse = max(dot(normal, lightDirection), 0.0);
    vec3 halfway = normalize(lightDirection + viewDirection);
    float specularTerm = pow(max(dot(normal, halfway), 0.0), 64.0) * specular;
    return lightColor * intensity * (albedo * diffuse + vec3(specularTerm));
}

void main() {
    vec3 world = texture(uWorldTexture, vUv).xyz;
    vec4 normalData = texture(uNormalTexture, vUv);
    vec4 albedoData = texture(uAlbedoTexture, vUv);

    float coverage = texture(uWorldTexture, vUv).w;
    vec3 background = mix(vec3(0.008, 0.012, 0.022), vec3(0.025, 0.04, 0.065), vUv.y);
    if (coverage < 0.5) {
        oColor = vec4(pow(background, vec3(1.0 / 2.2)), 1.0);
        return;
    }

    vec3 normal = normalize(normalData.xyz);
    vec3 albedo = albedoData.rgb;
    float specular = albedoData.a;
    vec3 viewDirection = normalize(uCameraPosition - world);
    vec3 radiance = uAmbientExposure.rgb * albedo;

    int pointCount = int(uCounts.x);
    for (int index = 0; index < pointCount; ++index) {
        vec4 positionRadius = uPointPositionRadius[index];
        vec4 colorIntensity = uPointColorIntensity[index];
        radiance += shadePoint(world, normal, viewDirection, albedo, specular,
                               positionRadius.xyz, colorIntensity.rgb,
                               colorIntensity.a, positionRadius.w);
    }

    int directionalCount = int(uCounts.y);
    for (int index = 0; index < directionalCount; ++index) {
        vec4 direction = uDirection[index];
        vec4 colorIntensity = uDirectionColorIntensity[index];
        radiance += shadeDirectional(normal, viewDirection, albedo, specular,
                                     direction.xyz, colorIntensity.rgb, colorIntensity.a);
    }

    radiance *= uAmbientExposure.a;
    vec3 mapped = tonemapAces(radiance);
    oColor = vec4(pow(mapped, vec3(1.0 / 2.2)), 1.0);
}
)GLSL";

} // namespace dr::shaders
