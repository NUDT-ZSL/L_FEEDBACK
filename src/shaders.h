#pragma once

namespace sr::shaders {

inline constexpr const char* kSkinnedSceneVertex = R"GLSL(
#version 430 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in uvec4 aBoneIndices;
layout(location = 3) in vec4 aBoneWeights;

layout(std140, binding = 0) uniform BoneBlock {
    mat4 uBoneMatrices[4];
};

uniform mat4 uModel;
uniform mat4 uViewProjection;

out vec3 vWorldPosition;
out vec3 vWorldNormal;

invariant gl_Position;

void main() {
    vec4 weights = aBoneWeights;
    float weightSum = dot(weights, vec4(1.0));
    if (weightSum < 1.0e-6) {
        weights = vec4(1.0, 0.0, 0.0, 0.0);
        weightSum = 1.0;
    }
    weights /= weightSum;

    mat4 skin =
        uBoneMatrices[aBoneIndices.x] * weights.x +
        uBoneMatrices[aBoneIndices.y] * weights.y +
        uBoneMatrices[aBoneIndices.z] * weights.z +
        uBoneMatrices[aBoneIndices.w] * weights.w;

    vec4 worldPosition = uModel * skin * vec4(aPosition, 1.0);
    vWorldPosition = worldPosition.xyz;

    mat3 boneNormalMatrix =
        mat3(transpose(inverse(uBoneMatrices[aBoneIndices.x]))) * weights.x +
        mat3(transpose(inverse(uBoneMatrices[aBoneIndices.y]))) * weights.y +
        mat3(transpose(inverse(uBoneMatrices[aBoneIndices.z]))) * weights.z +
        mat3(transpose(inverse(uBoneMatrices[aBoneIndices.w]))) * weights.w;
    mat3 modelNormalMatrix = mat3(transpose(inverse(uModel)));
    vWorldNormal = normalize(modelNormalMatrix * boneNormalMatrix * aNormal);

    gl_Position = uViewProjection * worldPosition;
}
)GLSL";

inline constexpr const char* kStaticSceneVertex = R"GLSL(
#version 430 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uViewProjection;

out vec3 vWorldPosition;
out vec3 vWorldNormal;

invariant gl_Position;

void main() {
    vec4 worldPosition = uModel * vec4(aPosition, 1.0);
    vWorldPosition = worldPosition.xyz;
    vWorldNormal = normalize(mat3(transpose(inverse(uModel))) * aNormal);
    gl_Position = uViewProjection * worldPosition;
}
)GLSL";

inline constexpr const char* kSkinnedShadowVertex = R"GLSL(
#version 430 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in uvec4 aBoneIndices;
layout(location = 3) in vec4 aBoneWeights;

layout(std140, binding = 0) uniform BoneBlock {
    mat4 uBoneMatrices[4];
};

layout(std140, binding = 1) uniform ShadowBlock {
    mat4 uShadowMatrices[4];
    vec4 uCascadeSplits;
    vec4 uShadowParams;
};

layout(std140, binding = 2) uniform LightBlock {
    vec4 uDirectionalDirection;
    vec4 uDirectionalColorIntensity;
    vec4 uPoint0PositionRange;
    vec4 uPoint0ColorIntensity;
    vec4 uPoint1PositionRange;
    vec4 uPoint1ColorIntensity;
    vec4 uAmbientColorIntensity;
};

uniform mat4 uModel;
uniform int uCascadeIndex;

invariant gl_Position;

void main() {
    vec4 weights = aBoneWeights;
    float weightSum = dot(weights, vec4(1.0));
    if (weightSum < 1.0e-6) {
        weights = vec4(1.0, 0.0, 0.0, 0.0);
        weightSum = 1.0;
    }
    weights /= weightSum;

    mat4 skin =
        uBoneMatrices[aBoneIndices.x] * weights.x +
        uBoneMatrices[aBoneIndices.y] * weights.y +
        uBoneMatrices[aBoneIndices.z] * weights.z +
        uBoneMatrices[aBoneIndices.w] * weights.w;
    vec4 worldPosition = uModel * skin * vec4(aPosition, 1.0);
    gl_Position = uShadowMatrices[uCascadeIndex] * worldPosition;
}
)GLSL";

inline constexpr const char* kStaticShadowVertex = R"GLSL(
#version 430 core

layout(location = 0) in vec3 aPosition;

layout(std140, binding = 1) uniform ShadowBlock {
    mat4 uShadowMatrices[4];
    vec4 uCascadeSplits;
    vec4 uShadowParams;
};

layout(std140, binding = 2) uniform LightBlock {
    vec4 uDirectionalDirection;
    vec4 uDirectionalColorIntensity;
    vec4 uPoint0PositionRange;
    vec4 uPoint0ColorIntensity;
    vec4 uPoint1PositionRange;
    vec4 uPoint1ColorIntensity;
    vec4 uAmbientColorIntensity;
};

uniform mat4 uModel;
uniform int uCascadeIndex;

invariant gl_Position;

void main() {
    gl_Position = uShadowMatrices[uCascadeIndex] * uModel * vec4(aPosition, 1.0);
}
)GLSL";

inline constexpr const char* kForwardFragment = R"GLSL(
#version 430 core

in vec3 vWorldPosition;
in vec3 vWorldNormal;

layout(location = 0) out vec4 outColor;

layout(std140, binding = 2) uniform LightBlock {
    vec4 uDirectionalDirection;
    vec4 uDirectionalColorIntensity;
    vec4 uPoint0PositionRange;
    vec4 uPoint0ColorIntensity;
    vec4 uPoint1PositionRange;
    vec4 uPoint1ColorIntensity;
    vec4 uAmbientColorIntensity;
};

layout(std140, binding = 1) uniform ShadowBlock {
    mat4 uShadowMatrices[4];
    vec4 uCascadeSplits;
    vec4 uShadowParams;
};

uniform sampler2D uShadowMap0;
uniform sampler2D uShadowMap1;
uniform sampler2D uShadowMap2;
uniform sampler2D uShadowMap3;
uniform mat4 uView;
uniform vec3 uCameraPosition;
uniform vec3 uBaseColor;
uniform vec4 uMaterial;

float sampleShadow(sampler2D shadowMap, mat4 shadowMatrix,
                   vec3 worldPosition, vec3 normal, vec3 lightDirection) {
    vec4 lightPosition = shadowMatrix * vec4(worldPosition, 1.0);
    if (abs(lightPosition.w) < 1.0e-8) {
        return 1.0;
    }
    vec3 projected = lightPosition.xyz / lightPosition.w;
    projected = projected * 0.5 + 0.5;
    if (projected.x < 0.0 || projected.x > 1.0 ||
        projected.y < 0.0 || projected.y > 1.0 ||
        projected.z < 0.0 || projected.z > 1.0) {
        return 1.0;
    }

    float bias = uShadowParams.z +
                 uShadowParams.w * (1.0 - max(dot(normal, lightDirection), 0.0));
    vec2 texel = vec2(1.0 / max(uShadowParams.y, 1.0));
    float visible = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            vec2 uv = projected.xy + vec2(float(x), float(y)) * texel;
            float storedDepth = texture(shadowMap, uv).r;
            visible += (projected.z - bias <= storedDepth) ? 1.0 : 0.0;
        }
    }
    return visible / 9.0;
}

vec3 evaluatePointLight(vec4 positionRange, vec4 colorIntensity,
                        vec3 normal, vec3 viewDirection, vec3 albedo) {
    vec3 toLight = positionRange.xyz - vWorldPosition;
    float distanceToLight = length(toLight);
    vec3 lightDirection = toLight / max(distanceToLight, 1.0e-6);
    float rangeAttenuation =
        clamp(1.0 - distanceToLight / max(positionRange.w, 1.0e-6), 0.0, 1.0);
    float attenuation = rangeAttenuation * rangeAttenuation /
                        (1.0 + 0.11 * distanceToLight * distanceToLight);
    float diffuse = max(dot(normal, lightDirection), 0.0);
    vec3 halfVector = normalize(lightDirection + viewDirection);
    float specular = pow(max(dot(normal, halfVector), 0.0),
                         mix(64.0, 8.0, uMaterial.x));
    return colorIntensity.rgb * colorIntensity.w * attenuation *
           (albedo * diffuse + vec3(uMaterial.y) * specular);
}

void main() {
    vec3 normal = normalize(vWorldNormal);
    vec3 viewDirection = normalize(uCameraPosition - vWorldPosition);
    vec3 albedo = uBaseColor;
    if (uMaterial.z > 0.0) {
        vec2 cell = floor(vWorldPosition.xz * uMaterial.z);
        float checker = mod(cell.x + cell.y, 2.0);
        albedo *= mix(0.88, 1.06, checker);
    }

    vec3 lightDirection = normalize(-uDirectionalDirection.xyz);
    float viewDepth = -(uView * vec4(vWorldPosition, 1.0)).z;
    float shadow = 1.0;
    if (viewDepth <= uCascadeSplits.x) {
        shadow = sampleShadow(uShadowMap0, uShadowMatrices[0],
                              vWorldPosition, normal, lightDirection);
    } else if (int(uShadowParams.x) > 1 && viewDepth <= uCascadeSplits.y) {
        shadow = sampleShadow(uShadowMap1, uShadowMatrices[1],
                              vWorldPosition, normal, lightDirection);
    } else if (int(uShadowParams.x) > 2 && viewDepth <= uCascadeSplits.z) {
        shadow = sampleShadow(uShadowMap2, uShadowMatrices[2],
                              vWorldPosition, normal, lightDirection);
    } else if (int(uShadowParams.x) > 3 && viewDepth <= uCascadeSplits.w) {
        shadow = sampleShadow(uShadowMap3, uShadowMatrices[3],
                              vWorldPosition, normal, lightDirection);
    }

    vec3 color = uAmbientColorIntensity.rgb * uAmbientColorIntensity.w * albedo;
    float directionalDiffuse = max(dot(normal, lightDirection), 0.0);
    vec3 halfVector = normalize(lightDirection + viewDirection);
    float directionalSpecular = pow(
        max(dot(normal, halfVector), 0.0),
        mix(64.0, 8.0, uMaterial.x));
    color += shadow * uDirectionalColorIntensity.rgb *
             uDirectionalColorIntensity.w *
             (albedo * directionalDiffuse +
              vec3(uMaterial.y) * directionalSpecular);
    color += evaluatePointLight(uPoint0PositionRange, uPoint0ColorIntensity,
                                normal, viewDirection, albedo);
    color += evaluatePointLight(uPoint1PositionRange, uPoint1ColorIntensity,
                                normal, viewDirection, albedo);

    color = color / (color + vec3(1.0));
    color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
    outColor = vec4(color, 1.0);
}
)GLSL";

inline constexpr const char* kDepthFragment = R"GLSL(
#version 430 core

layout(std140, binding = 1) uniform ShadowBlock {
    mat4 uShadowMatrices[4];
    vec4 uCascadeSplits;
    vec4 uShadowParams;
};

layout(std140, binding = 2) uniform LightBlock {
    vec4 uDirectionalDirection;
    vec4 uDirectionalColorIntensity;
    vec4 uPoint0PositionRange;
    vec4 uPoint0ColorIntensity;
    vec4 uPoint1PositionRange;
    vec4 uPoint1ColorIntensity;
    vec4 uAmbientColorIntensity;
};

void main() {
}
)GLSL";

} // namespace sr::shaders
