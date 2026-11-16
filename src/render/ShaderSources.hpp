#pragma once

#include <string>

namespace lcr {

inline const std::string kCharacterVertexShader = R"GLSL(
#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;
layout(location = 3) in ivec4 aJoints;
layout(location = 4) in vec4 aWeights;

uniform mat4 uBones[24];
uniform mat4 uViewProjection;

out vec3 vWorldPosition;
out vec3 vNormal;
out vec2 vUv;

void main() {
    mat4 skin =
        uBones[aJoints.x] * aWeights.x +
        uBones[aJoints.y] * aWeights.y +
        uBones[aJoints.z] * aWeights.z +
        uBones[aJoints.w] * aWeights.w;
    vec4 world = skin * vec4(aPosition, 1.0);
    vWorldPosition = world.xyz;
    vNormal = normalize(mat3(skin) * aNormal);
    vUv = aUv;
    gl_Position = uViewProjection * world;
}
)GLSL";

inline const std::string kCharacterFragmentShader = R"GLSL(
#version 330 core

in vec3 vWorldPosition;
in vec3 vNormal;
in vec2 vUv;

uniform sampler2D uBaseTexture;
uniform vec3 uCameraPosition;

uniform vec3 uBaseColor;
uniform float uMetallic;
uniform float uRoughness;
uniform vec3 uEmissiveColor;
uniform float uEmissiveStrength;
uniform float uOpacity;

uniform vec3 uMainLightDirection;
uniform vec3 uMainLightColor;
uniform vec3 uFillLightDirection;
uniform vec3 uFillLightColor;
uniform vec3 uAmbientColor;

uniform float uCapeMode;
uniform float uFeatherStart;
uniform float uFeatherWidth;

out vec4 fragColor;

vec3 blinnPhong(vec3 normal, vec3 viewDirection, vec3 lightDirection, vec3 lightColor) {
    float ndotl = max(dot(normal, lightDirection), 0.0);
    vec3 halfDirection = normalize(lightDirection + viewDirection);
    float specularPower = mix(96.0, 10.0, uRoughness);
    float specular = pow(max(dot(normal, halfDirection), 0.0), specularPower);
    specular *= mix(0.10, 0.85, uMetallic);
    return lightColor * (ndotl + specular);
}

void main() {
    vec3 normal = normalize(vNormal);
    if (uCapeMode > 0.5 && !gl_FrontFacing) {
        normal = -normal;
    }
    vec4 texel = texture(uBaseTexture, vUv);
    vec3 albedo = uBaseColor * (0.55 + 0.75 * texel.rgb);
    vec3 viewDirection = normalize(uCameraPosition - vWorldPosition);

    vec3 lighting = vec3(0.0);
    lighting += blinnPhong(normal, viewDirection, normalize(uMainLightDirection), uMainLightColor);
    lighting += blinnPhong(normal, viewDirection, normalize(uFillLightDirection), uFillLightColor);
    vec3 color = albedo * (uAmbientColor + lighting);
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    float fresnel = pow(1.0 - max(dot(normal, viewDirection), 0.0), 3.0);
    color += uEmissiveColor * uEmissiveStrength * texel.a;
    color += uEmissiveColor * uEmissiveStrength * fresnel * 0.12;

    float alpha = uOpacity;
    if (uCapeMode > 0.5) {
        float border = min(min(vUv.x, 1.0 - vUv.x), min(vUv.y, 1.0 - vUv.y));
        alpha *= smoothstep(uFeatherStart, uFeatherStart + uFeatherWidth, border);
    }
    fragColor = vec4(color, alpha);
}
)GLSL";

inline const std::string kForcedShaderFailure = R"GLSL(
#error forced shader compile failure
)GLSL";

inline std::string injectShaderFailure(const std::string& source) {
  return kForcedShaderFailure + source;
}

}  // namespace lcr

