#pragma once

namespace subdiv {

inline constexpr const char* kSurfaceVertexShader = R"GLSL(
#version 410 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aVertexNormal;
layout(location = 2) in vec3 aFaceNormal;
layout(location = 3) in float aFaceId;

uniform mat4 uViewProjection;

out vec3 vWorldPosition;
out vec3 vVertexNormal;
out vec3 vFaceNormal;
flat out float vFaceId;

void main() {
    vWorldPosition = aPosition;
    vVertexNormal = aVertexNormal;
    vFaceNormal = aFaceNormal;
    vFaceId = aFaceId;
    gl_Position = uViewProjection * vec4(aPosition, 1.0);
}
)GLSL";

inline constexpr const char* kSurfaceFragmentShader = R"GLSL(
#version 410 core

in vec3 vWorldPosition;
in vec3 vVertexNormal;
in vec3 vFaceNormal;
flat in float vFaceId;

uniform vec3 uCameraPosition;
uniform vec3 uDiffuseColor;
uniform float uRoughness;
uniform int uNormalMode;
uniform int uUseFaceNormal;

out vec4 fragmentColor;

const float PI = 3.14159265358979323846;

float hash31(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float hash11(float p) {
    p = fract(p * 0.1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

float detailHeight(vec3 p) {
    float a = sin(dot(p, vec3(11.7, 17.3, 13.1)));
    float b = sin(dot(p, vec3(-7.9, 23.1, 5.3)));
    float c = sin(dot(p, vec3(19.3, -4.7, 9.1)));
    return (a + b * 0.6 + c * 0.35) / 1.95;
}

vec3 detailNormal(vec3 p, vec3 n) {
    vec3 tangent = normalize(cross(abs(n.y) < 0.9 ? vec3(0.0, 1.0, 0.0)
                                                  : vec3(1.0, 0.0, 0.0), n));
    vec3 bitangent = cross(n, tangent);
    const float epsilon = 0.018;
    float h0 = detailHeight(p * 24.0);
    float hx = detailHeight((p + tangent * epsilon) * 24.0);
    float hy = detailHeight((p + bitangent * epsilon) * 24.0);
    vec3 slope = tangent * ((hx - h0) / epsilon) +
                 bitangent * ((hy - h0) / epsilon);
    return normalize(n - slope * 0.016);
}

float marchViewRay(vec3 position, vec3 normal, vec3 viewDirection) {
    // The subdivided cage is the macro surface. This short ray march resolves
    // a deterministic micro-surface in tangent space for view-dependent shading.
    vec3 start = position + normal * 0.020;
    float previousRayHeight = dot(start - position, normal);
    float previousSurfaceHeight = detailHeight(start * 31.0) * 0.006;
    const int stepCount = 12;
    for (int i = 0; i < stepCount; ++i) {
        float fraction = float(i + 1) / float(stepCount);
        vec3 samplePosition = start - viewDirection * (0.050 * fraction);
        float sampleHeight = dot(samplePosition - position, normal);
        float surfaceHeight = detailHeight(samplePosition * 31.0) * 0.006;
        if (sampleHeight <= surfaceHeight) {
            float previousDelta = previousRayHeight - previousSurfaceHeight;
            float currentDelta = sampleHeight - surfaceHeight;
            float denominator = max(previousDelta - currentDelta, 1.0e-5);
            return clamp(previousDelta / denominator, 0.0, 1.0);
        }
        previousRayHeight = sampleHeight;
        previousSurfaceHeight = surfaceHeight;
    }
    return 0.0;
}

float distributionGGX(float nDotH, float roughness) {
    float alpha = max(roughness * roughness, 0.002);
    float alpha2 = alpha * alpha;
    float denominator = nDotH * nDotH * (alpha2 - 1.0) + 1.0;
    return alpha2 / max(PI * denominator * denominator, 1.0e-6);
}

float geometrySmith(float nDotV, float nDotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float gv = nDotV / (nDotV * (1.0 - k) + k);
    float gl = nDotL / (nDotL * (1.0 - k) + k);
    return gv * gl;
}

vec3 fresnelSchlick(float cosTheta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 shadeSurface(vec3 position, vec3 normal, vec3 viewDirection) {
    float rayConfidence = marchViewRay(position, normal, viewDirection);
    vec3 detailedNormal = detailNormal(position, normal);
    normal = normalize(mix(normal, detailedNormal, 0.11 + 0.10 * rayConfidence));

    vec3 lightDirection = normalize(vec3(0.45, 0.82, 0.36));
    vec3 halfVector = normalize(viewDirection + lightDirection);
    float nDotL = max(dot(normal, lightDirection), 0.0);
    float nDotV = max(dot(normal, viewDirection), 0.001);
    float nDotH = max(dot(normal, halfVector), 0.0);
    float vDotH = max(dot(viewDirection, halfVector), 0.0);

    vec3 f0 = vec3(0.04);
    vec3 fresnel = fresnelSchlick(vDotH, f0);
    float distribution = distributionGGX(nDotH, uRoughness);
    float geometry = geometrySmith(nDotV, nDotL, uRoughness);
    vec3 specular = (fresnel * distribution * geometry) /
                    max(4.0 * nDotV * nDotL, 1.0e-5);
    vec3 diffuse = (vec3(1.0) - fresnel) * uDiffuseColor / PI;

    vec3 keyRadiance = vec3(1.0, 0.93, 0.82) * 3.1;
    vec3 color = (diffuse + specular) * keyRadiance * nDotL;

    vec3 fillDirection = normalize(vec3(-0.65, 0.25, 0.45));
    float fillAmount = max(dot(normal, fillDirection), 0.0);
    color += uDiffuseColor * vec3(0.16, 0.24, 0.38) * fillAmount;
    color += uDiffuseColor * (0.10 + 0.05 * max(normal.y, 0.0));

    float rim = pow(1.0 - nDotV, 3.0);
    color += vec3(0.20, 0.34, 0.55) * rim * 0.23;
    color += vec3(0.025) * rayConfidence;

    color = color / (color + vec3(1.0));
    color = pow(max(color, vec3(0.0)), vec3(1.0 / 2.2));
    return color;
}

void main() {
    vec3 faceNormal = normalize(vFaceNormal);
    vec3 vertexNormal = normalize(vVertexNormal);
    if (!gl_FrontFacing) {
        faceNormal = -faceNormal;
        vertexNormal = -vertexNormal;
    }

    if (uNormalMode != 0) {
        vec3 normal = uUseFaceNormal != 0 ? faceNormal : vertexNormal;
        vec3 normalColor = normal * 0.5 + 0.5;
        if (uUseFaceNormal != 0) {
            normalColor *= 0.94 + 0.06 * hash11(vFaceId);
        }
        fragmentColor = vec4(clamp(normalColor, 0.0, 1.0), 1.0);
        return;
    }

    vec3 viewDirection = normalize(uCameraPosition - vWorldPosition);
    vec3 color = shadeSurface(vWorldPosition, vertexNormal, viewDirection);
    fragmentColor = vec4(color, 1.0);
}
)GLSL";

}  // namespace subdiv

