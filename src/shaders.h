#pragma once

constexpr const char* kSceneVertexShader = R"GLSL(
#version 450 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uViewProjection;
uniform mat3 uNormalMatrix;

out vec3 vWorldPosition;
out vec3 vNormal;

void main() {
    vec4 world = uModel * vec4(aPosition, 1.0);
    vWorldPosition = world.xyz;
    vNormal = normalize(uNormalMatrix * aNormal);
    gl_Position = uViewProjection * world;
}
)GLSL";

constexpr const char* kSceneFragmentShader = R"GLSL(
#version 450 core

const int MAX_CASCADES = 3;
const int MAX_POINT_LIGHTS = 2;
const int BLOCKER_SAMPLES = 16;
const int FILTER_SAMPLES = 32;

in vec3 vWorldPosition;
in vec3 vNormal;

layout(location = 0) out vec4 fragColor;

uniform vec3 uCameraPosition;
uniform vec3 uBaseColor;
uniform float uRoughness;
uniform float uMetallic;
uniform int uPattern;
uniform int uDebugCascades;

uniform int uCascadeCount;
uniform float uSplitDepths[MAX_CASCADES];
uniform mat4 uDirLightVP[MAX_CASCADES];
uniform mat4 uDirLightView[MAX_CASCADES];
uniform float uDirFar[MAX_CASCADES];
uniform float uDirRadius[MAX_CASCADES];
uniform sampler2DArray uDirShadowMaps;
uniform vec3 uDirDirection;
uniform vec3 uDirColor;
uniform float uDirIntensity;
uniform float uDirSourceDiameter;
uniform int uDirEnabled;

uniform int uPointActiveCount;
uniform vec3 uPointPosition[MAX_POINT_LIGHTS];
uniform vec3 uPointColor[MAX_POINT_LIGHTS];
uniform float uPointIntensity[MAX_POINT_LIGHTS];
uniform float uPointFar[MAX_POINT_LIGHTS];
uniform float uPointSourceDiameter[MAX_POINT_LIGHTS];
uniform int uPointEnabled[MAX_POINT_LIGHTS];
uniform samplerCube uPointShadowMaps[MAX_POINT_LIGHTS];

uniform float uShadowResolution;
uniform float uNormalOffsetTexels;
uniform float uDepthBiasTexels;
uniform float uSlopeBiasTexels;
uniform mat4 uView;

const vec2 kPoissonDisk[32] = vec2[](
    vec2(-0.613392,  0.617481), vec2( 0.170019, -0.040254),
    vec2(-0.299417,  0.791925), vec2( 0.645680,  0.493210),
    vec2(-0.651784,  0.717887), vec2( 0.421003,  0.027070),
    vec2(-0.817194, -0.271096), vec2(-0.705374, -0.668203),
    vec2( 0.977050, -0.108615), vec2( 0.063326,  0.142369),
    vec2( 0.203528,  0.214331), vec2(-0.667531,  0.326090),
    vec2(-0.098422, -0.295589), vec2(-0.756922, -0.454347),
    vec2(-0.017020,  0.885994), vec2( 0.676805,  0.843491),
    vec2( 0.532731,  0.601566), vec2( 0.723216,  0.196249),
    vec2( 0.847652, -0.317906), vec2( 0.636793, -0.632239),
    vec2( 0.251431, -0.853237), vec2(-0.145162, -0.950945),
    vec2(-0.512609, -0.841862), vec2(-0.795495, -0.584470),
    vec2(-0.975476, -0.178415), vec2(-0.918154,  0.201206),
    vec2(-0.719622,  0.434568), vec2( 0.312825,  0.843064),
    vec2( 0.939844,  0.325231), vec2( 0.447720, -0.328150),
    vec2(-0.298274, -0.489857), vec2( 0.138173, -0.627095)
);

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 rotateVec2(vec2 v, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

vec2 poissonSample(int index, float angle) {
    return rotateVec2(kPoissonDisk[index % 32], angle);
}

float directionalCascadeShadow(int cascadeIndex,
                               vec3 worldPosition,
                               vec3 normal,
                               float ndl,
                               float rotationAngle) {
    float texelWorld = (2.0 * uDirRadius[cascadeIndex]) / uShadowResolution;
    float normalOffset = texelWorld * uNormalOffsetTexels *
                         clamp(1.0 - ndl * 0.75, 0.20, 1.0);
    vec3 biasedWorld = worldPosition + normal * normalOffset;

    vec4 lightClip = uDirLightVP[cascadeIndex] * vec4(biasedWorld, 1.0);
    if (lightClip.w <= 0.0) {
        return 1.0;
    }
    vec3 ndc = lightClip.xyz / lightClip.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
        return 1.0;
    }

    float farPlane = uDirFar[cascadeIndex];
    float receiver = -(uDirLightView[cascadeIndex] * vec4(biasedWorld, 1.0)).z /
                     farPlane;
    float depthBias = (uDepthBiasTexels + uSlopeBiasTexels * (1.0 - ndl)) *
                      texelWorld / farPlane;
    float receiverForCompare = receiver - depthBias;
    float worldToUv = 1.0 / (2.0 * uDirRadius[cascadeIndex]);
    float searchWorld = max(uDirSourceDiameter * 2.0, texelWorld * 1.5);
    float searchRadiusUv = min(searchWorld * worldToUv, 0.20);

    float blockerSum = 0.0;
    int blockerCount = 0;
    for (int i = 0; i < BLOCKER_SAMPLES; ++i) {
        vec2 offset = poissonSample(i * 2, rotationAngle) * searchRadiusUv;
        float sampleDepth = texture(uDirShadowMaps,
                                    vec3(uv + offset, float(cascadeIndex))).r;
        if (sampleDepth < receiverForCompare) {
            blockerSum += sampleDepth;
            ++blockerCount;
        }
    }
    if (blockerCount == 0) {
        return 1.0;
    }

    float averageBlocker = blockerSum / float(blockerCount);
    if (averageBlocker >= receiverForCompare) {
        return 1.0;
    }

    float receiverWorld = receiver * farPlane;
    float blockerWorld = max(averageBlocker * farPlane, 0.001);
    float penumbraWorld = uDirSourceDiameter *
                          max(receiverWorld - blockerWorld, 0.0) / blockerWorld;
    float filterRadiusUv = clamp(penumbraWorld * worldToUv,
                                 0.5 / uShadowResolution, 0.20);

    float lit = 0.0;
    for (int i = 0; i < FILTER_SAMPLES; ++i) {
        vec2 offset = poissonSample(i, rotationAngle + 1.399) * filterRadiusUv;
        float sampleDepth = texture(uDirShadowMaps,
                                    vec3(uv + offset, float(cascadeIndex))).r;
        lit += sampleDepth >= receiverForCompare ? 1.0 : 0.0;
    }
    return lit / float(FILTER_SAMPLES);
}

vec3 cubeDirectionOffset(vec3 direction, vec2 diskOffset, float angularScale) {
    vec3 helper = abs(direction.y) > 0.95 ? vec3(0.0, 0.0, 1.0)
                                          : vec3(0.0, 1.0, 0.0);
    vec3 tangent = normalize(cross(helper, direction));
    vec3 bitangent = cross(direction, tangent);
    return normalize(direction + (tangent * diskOffset.x +
                                  bitangent * diskOffset.y) * angularScale);
}

float pointLightShadow(int lightIndex,
                       vec3 worldPosition,
                       vec3 normal,
                       float ndl,
                       float rotationAngle) {
    vec3 lightPosition = uPointPosition[lightIndex];
    vec3 toLightVector = lightPosition - worldPosition;
    float distanceToLight = length(toLightVector);
    vec3 lightDirection = toLightVector / max(distanceToLight, 0.0001);
    float farPlane = uPointFar[lightIndex];
    float texelWorld = (2.0 * distanceToLight) / uShadowResolution;
    float normalOffset = texelWorld * uNormalOffsetTexels *
                         clamp(1.0 - ndl * 0.75, 0.20, 1.0);
    vec3 biasedWorld = worldPosition + normal * normalOffset;
    vec3 shadowVector = biasedWorld - lightPosition;
    float receiverDistance = length(shadowVector);
    vec3 shadowDirection = shadowVector / max(receiverDistance, 0.0001);
    float depthBias = (uDepthBiasTexels + uSlopeBiasTexels * (1.0 - ndl)) *
                      texelWorld / farPlane;
    float receiver = receiverDistance / farPlane - depthBias;

    float searchUv = clamp((max(uPointSourceDiameter[lightIndex] * 2.0,
                                texelWorld * 1.5) * 0.5) /
                               max(distanceToLight, 0.001),
                           0.0005, 0.20);
    float blockerSum = 0.0;
    int blockerCount = 0;
    for (int i = 0; i < BLOCKER_SAMPLES; ++i) {
        vec2 disk = poissonSample(i * 2, rotationAngle) * searchUv;
        vec3 sampleDirection = cubeDirectionOffset(shadowDirection, disk, searchUv * 1.8);
        float sampleDepth = texture(uPointShadowMaps[lightIndex], sampleDirection).r;
        if (sampleDepth < receiver) {
            blockerSum += sampleDepth;
            ++blockerCount;
        }
    }
    if (blockerCount == 0) {
        return 1.0;
    }

    float averageBlocker = blockerSum / float(blockerCount);
    if (averageBlocker >= receiver) {
        return 1.0;
    }
    float receiverWorld = receiverDistance;
    float blockerWorld = max(averageBlocker * farPlane, 0.001);
    float penumbraWorld = uPointSourceDiameter[lightIndex] *
                          max(receiverWorld - blockerWorld, 0.0) / blockerWorld;
    float filterUv = clamp(penumbraWorld /
                               max(2.0 * receiverDistance, 0.001),
                           0.5 / uShadowResolution, 0.20);

    float lit = 0.0;
    for (int i = 0; i < FILTER_SAMPLES; ++i) {
        vec2 disk = poissonSample(i, rotationAngle + 1.399) * filterUv;
        vec3 sampleDirection = cubeDirectionOffset(shadowDirection, disk, filterUv * 1.8);
        float sampleDepth = texture(uPointShadowMaps[lightIndex], sampleDirection).r;
        lit += sampleDepth >= receiver ? 1.0 : 0.0;
    }
    return lit / float(FILTER_SAMPLES);
}

float directionalShadow(vec3 worldPosition,
                        vec3 normal,
                        float viewDepth,
                        float ndl,
                        out int selectedCascade) {
    selectedCascade = 0;
    if (uDirEnabled == 0) {
        return 1.0;
    }
    for (int i = 0; i < MAX_CASCADES; ++i) {
        if (i >= uCascadeCount) {
            break;
        }
        if (viewDepth <= uSplitDepths[i]) {
            selectedCascade = i;
            break;
        }
        selectedCascade = i;
    }

    float angle = hash12(gl_FragCoord.xy) * 6.28318530718;
    float shadow = directionalCascadeShadow(selectedCascade, worldPosition,
                                            normal, ndl, angle);

    if (selectedCascade < uCascadeCount - 1) {
        int current = selectedCascade;
        float split = uSplitDepths[current];
        float previous = current == 0 ? 0.0 : uSplitDepths[current - 1];
        float blendBand = max((split - previous) * 0.16, 0.05);
        float blendStart = split - blendBand;
        if (viewDepth > blendStart) {
            float blend = smoothstep(blendStart, split, viewDepth);
            float nextShadow = directionalCascadeShadow(current + 1, worldPosition,
                                                        normal, ndl, angle);
            shadow = mix(shadow, nextShadow, blend);
            selectedCascade = blend > 0.5 ? current + 1 : current;
        }
    }
    return shadow;
}

vec3 proceduralAlbedo(vec3 baseColor, vec3 worldPosition) {
    if (uPattern == 1) {
        float checker = mod(floor(worldPosition.x) + floor(worldPosition.z), 2.0);
        return baseColor * mix(0.76, 1.04, checker);
    }
    float grain = fract(sin(dot(floor(worldPosition.xz * 4.0),
                                vec2(12.9898, 78.233))) * 43758.5453);
    return baseColor * mix(0.94, 1.04, grain);
}

void main() {
    vec3 normal = normalize(vNormal);
    vec3 viewDirection = normalize(uCameraPosition - vWorldPosition);
    float viewDepth = -(uView * vec4(vWorldPosition, 1.0)).z;
    vec3 albedo = proceduralAlbedo(uBaseColor, vWorldPosition);

    vec3 ambient = albedo * (0.105 + 0.075 * max(normal.y, 0.0));
    vec3 direct = vec3(0.0);

    vec3 dirToLight = normalize(-uDirDirection);
    float dirNdl = max(dot(normal, dirToLight), 0.0);
    int selectedCascade = 0;
    if (uDirEnabled != 0 && dirNdl > 0.0) {
        float shadow = directionalShadow(vWorldPosition, normal, viewDepth,
                                         dirNdl, selectedCascade);
        vec3 halfVector = normalize(dirToLight + viewDirection);
        float specular = pow(max(dot(normal, halfVector), 0.0),
                              mix(48.0, 8.0, uRoughness));
        direct += uDirColor * uDirIntensity * dirNdl * shadow *
                  (albedo + specular * (1.0 - uRoughness) * (1.0 - uMetallic));
    }

    for (int i = 0; i < MAX_POINT_LIGHTS; ++i) {
        if (i >= uPointActiveCount || uPointEnabled[i] == 0) {
            continue;
        }
        vec3 toLight = uPointPosition[i] - vWorldPosition;
        float distanceToLight = length(toLight);
        vec3 lightDirection = toLight / max(distanceToLight, 0.0001);
        float ndl = max(dot(normal, lightDirection), 0.0);
        if (ndl <= 0.0) {
            continue;
        }
        float pixelAngle = hash12(gl_FragCoord.xy + float(i) * 17.13) *
                           6.28318530718;
        float shadow = pointLightShadow(i, vWorldPosition, normal, ndl,
                                        pixelAngle);
        float attenuation = 1.0 / (1.0 + 0.09 * distanceToLight +
                                   0.032 * distanceToLight * distanceToLight);
        vec3 halfVector = normalize(lightDirection + viewDirection);
        float specular = pow(max(dot(normal, halfVector), 0.0),
                              mix(52.0, 10.0, uRoughness));
        direct += uPointColor[i] * uPointIntensity[i] * attenuation * ndl * shadow *
                  (albedo + specular * (1.0 - uRoughness) * (1.0 - uMetallic));
    }

    vec3 color = ambient + direct;
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));

    if (uDebugCascades != 0) {
        vec3 palette[3] = vec3[3](
            vec3(1.0, 0.23, 0.08),
            vec3(0.10, 0.92, 0.28),
            vec3(0.10, 0.35, 1.0)
        );
        color = mix(color, palette[selectedCascade] * 1.15, 0.28);
    }

    fragColor = vec4(color, 1.0);
}
)GLSL";

constexpr const char* kOverlayVertexShader = R"GLSL(
#version 450 core
layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec4 aColor;

uniform vec2 uViewport;
uniform float uScale;

out vec4 vColor;

void main() {
    vec2 pixel = aPosition * uScale;
    vec2 ndc = vec2((pixel.x / uViewport.x) * 2.0 - 1.0,
                    1.0 - (pixel.y / uViewport.y) * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    vColor = aColor;
}
)GLSL";

constexpr const char* kOverlayFragmentShader = R"GLSL(
#version 450 core
in vec4 vColor;
layout(location = 0) out vec4 fragColor;
void main() {
    fragColor = vColor;
}
)GLSL";

