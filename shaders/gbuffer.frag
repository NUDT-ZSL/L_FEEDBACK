#version 460

#extension GL_GOOGLE_include_directive : require

#include "common.glsl"

layout(set = 0, binding = 2) uniform sampler2D directionalShadowMap;
layout(set = 0, binding = 3) uniform sampler2DArray pointShadowMap;

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec3 albedo;

layout(location = 0) out vec4 opaqueColor;

void main() {
    const vec3 normal = normalize(worldNormal);
    const vec3 toLight = -normalize(ubo.dirLightDirection.xyz);
    const float sunCosine = max(dot(normal, toLight), 0.0);
    const float sunVisibility =
        sunCosine > 0.0 ? directionalVisibility(directionalShadowMap, worldPosition) : 0.0;
    const vec3 sunRadiance =
        ubo.dirLightColor.rgb * ubo.dirLightDirection.w * sunCosine * sunVisibility;

    vec3 toPoint = ubo.pointLightPosition.xyz - worldPosition;
    const float pointDistance = length(toPoint);
    vec3 pointRadiance = vec3(0.0);
    if (pointDistance > 1.0e-5 && pointDistance < ubo.pointLightPosition.w) {
        toPoint /= pointDistance;
        const float pointCosine = max(dot(normal, toPoint), 0.0);
        const float pointVisibility =
            pointCosine > 0.0 ? pointVisibility(pointShadowMap, worldPosition) : 0.0;
        const float rangeRatio = pointDistance / ubo.pointLightPosition.w;
        const float window = pow(max(1.0 - pow(rangeRatio, 4.0), 0.0), 2.0);
        const float attenuation = window / (1.0 + pointDistance * pointDistance);
        pointRadiance = ubo.pointLightColor.rgb * ubo.pointLightColor.w *
                        pointCosine * attenuation * pointVisibility;
    }

    const vec3 ambient = albedo * 0.012;
    const vec3 direct = albedo * (sunRadiance + pointRadiance);
    opaqueColor = vec4(ambient + direct, 1.0);
}
