#ifndef VLR_COMMON_GLSL
#define VLR_COMMON_GLSL

layout(std140, set = 0, binding = 0) uniform SceneUBO {
    mat4 viewProj;
    mat4 invViewProj;
    mat4 dirLightViewProj;
    mat4 pointFaceViewProj[6];
    vec4 cameraPos;
    vec4 cameraParams;
    vec4 dirLightDirection;
    vec4 dirLightColor;
    vec4 pointLightPosition;
    vec4 pointLightColor;
    vec4 mediumCoefficients;
    vec4 volumeBoundsMin;
    vec4 volumeBoundsMax;
    vec4 noiseParams;
    vec4 shadowParams;
    vec4 pointShadowParams;
    vec4 toneParams;
    vec4 backgroundColor;
    vec4 renderParams;
} ubo;

int cubeFaceIndex(vec3 direction, out vec2 faceUv) {
    const float ax = abs(direction.x);
    const float ay = abs(direction.y);
    const float az = abs(direction.z);
    float major;
    float sc;
    float tc;
    int face;
    if (ax >= ay && ax >= az) {
        if (direction.x >= 0.0) {
            face = 0;
            major = direction.x;
            sc = -direction.z / major;
            tc = -direction.y / major;
        } else {
            face = 1;
            major = -direction.x;
            sc = direction.z / major;
            tc = -direction.y / major;
        }
    } else if (ay >= az) {
        if (direction.y >= 0.0) {
            face = 2;
            major = direction.y;
            sc = direction.x / major;
            tc = direction.z / major;
        } else {
            face = 3;
            major = -direction.y;
            sc = direction.x / major;
            tc = -direction.z / major;
        }
    } else {
        if (direction.z >= 0.0) {
            face = 4;
            major = direction.z;
            sc = direction.x / major;
            tc = -direction.y / major;
        } else {
            face = 5;
            major = -direction.z;
            sc = -direction.x / major;
            tc = -direction.y / major;
        }
    }
    faceUv = vec2(sc, tc) * 0.5 + 0.5;
    return face;
}

vec3 cubeFaceDirection(int face, vec2 uv) {
    const vec2 st = uv * 2.0 - 1.0;
    if (face == 0) {
        return normalize(vec3(1.0, -st.y, -st.x));
    }
    if (face == 1) {
        return normalize(vec3(-1.0, -st.y, st.x));
    }
    if (face == 2) {
        return normalize(vec3(st.x, 1.0, st.y));
    }
    if (face == 3) {
        return normalize(vec3(st.x, -1.0, -st.y));
    }
    if (face == 4) {
        return normalize(vec3(st.x, -st.y, 1.0));
    }
    return normalize(vec3(-st.x, -st.y, -1.0));
}

float directionalVisibility(sampler2D shadowMap, vec3 worldPosition) {
    const vec4 clip = ubo.dirLightViewProj * vec4(worldPosition, 1.0);
    if (clip.w <= 0.0) {
        return 0.0;
    }
    const vec3 ndc = clip.xyz / clip.w;
    if (ndc.z < 0.0 || ndc.z > 1.0) {
        return 0.0;
    }
    const vec2 uv = ndc.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
        return 0.0;
    }

    const float resolution = max(1.0, ubo.shadowParams.x);
    const float halfTexel = 0.5 / resolution;
    const vec2 texel = vec2(1.0 / resolution);
    const int kernel = max(1, int(ubo.pointShadowParams.z + 0.5));
    const int radius = kernel / 2;
    const float reference = ndc.z - ubo.shadowParams.z;
    float visible = 0.0;
    float count = 0.0;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            vec2 tapUv = uv + vec2(float(x), float(y)) * texel;
            tapUv = clamp(tapUv, vec2(halfTexel), vec2(1.0 - halfTexel));
            const float nearestDepth = textureLod(shadowMap, tapUv, 0.0).r;
            visible += reference <= nearestDepth + 0.000001 ? 1.0 : 0.0;
            count += 1.0;
        }
    }
    return count > 0.0 ? visible / count : 0.0;
}

float pointVisibility(sampler2DArray shadowMap, vec3 worldPosition) {
    const vec3 lightToPoint = worldPosition - ubo.pointLightPosition.xyz;
    const float distanceFromLight = length(lightToPoint);
    if (distanceFromLight <= 1.0e-6) {
        return 0.0;
    }
    const vec3 direction = lightToPoint / distanceFromLight;
    vec2 uv;
    const int originalFace = cubeFaceIndex(direction, uv);
    const vec4 clip = ubo.pointFaceViewProj[originalFace] * vec4(worldPosition, 1.0);
    if (clip.w <= 0.0) {
        return 0.0;
    }
    const float reference = clip.z / clip.w - ubo.shadowParams.w;
    if (reference < 0.0 || reference > 1.0) {
        return 0.0;
    }

    const float resolution = max(1.0, ubo.shadowParams.y);
    const float halfTexel = 0.5 / resolution;
    const vec2 texel = vec2(1.0 / resolution);
    const int kernel = max(1, int(ubo.pointShadowParams.z + 0.5));
    const int radius = kernel / 2;
    float visible = 0.0;
    float count = 0.0;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            vec2 tapUv = uv + vec2(float(x), float(y)) * texel;
            int tapFace = originalFace;
            if (any(lessThan(tapUv, vec2(0.0))) ||
                any(greaterThan(tapUv, vec2(1.0)))) {
                const vec3 tapDirection = cubeFaceDirection(originalFace, tapUv);
                tapFace = cubeFaceIndex(tapDirection, tapUv);
            }
            tapUv = clamp(tapUv, vec2(halfTexel), vec2(1.0 - halfTexel));
            const float nearestDepth =
                textureLod(shadowMap, vec3(tapUv, float(tapFace)), 0.0).r;
            visible += reference <= nearestDepth + 0.000001 ? 1.0 : 0.0;
            count += 1.0;
        }
    }
    return count > 0.0 ? visible / count : 0.0;
}

float henyeyGreenstein(float cosTheta, float anisotropy) {
    const float g = clamp(anisotropy, -0.95, 0.95);
    const float denominator = max(1.0e-6, 1.0 + g * g - 2.0 * g * cosTheta);
    return (1.0 - g * g) /
           (12.566370614359172 * pow(denominator, 1.5));
}

vec3 reinhard(vec3 linearColor) {
    return linearColor / (vec3(1.0) + max(linearColor, vec3(0.0)));
}

vec3 linearToSrgb(vec3 value) {
    const vec3 low = value * 12.92;
    const vec3 high = 1.055 * pow(max(value, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(low, high, greaterThan(value, vec3(0.0031308)));
}

#endif
