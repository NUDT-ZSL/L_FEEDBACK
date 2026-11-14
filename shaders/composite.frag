#version 450 core

in vec2 vUv;
out vec4 outColor;

struct DebugLight {
    vec4 positionRange;
    vec4 colorIntensity;
    ivec4 meta;
};

layout(std430, binding = 0) readonly buffer AllLights {
    DebugLight uDebugLights[];
};

uniform sampler2D uWorldPosition;
uniform sampler2D uViewNormal;
uniform sampler2D uAlbedo;
uniform sampler2D uSurface;
uniform sampler2D uHdr;
uniform int uDebugView;
uniform int uDebugLightCount;
uniform float uCacheOccupancy;
uniform float uExposure;

vec3 acesFilm(vec3 value) {
    return clamp((value * (2.51 * value + 0.03)) /
                 (value * (2.43 * value + 0.59) + 0.14), 0.0, 1.0);
}

vec3 hueColor(float hue) {
    const vec3 p = abs(fract(vec3(hue) + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 -
                       3.0);
    return clamp(p - 1.0, 0.0, 1.0);
}

void main() {
    const vec4 world = texture(uWorldPosition, vUv);
    const vec4 normal = texture(uViewNormal, vUv);
    const vec4 albedo = texture(uAlbedo, vUv);
    const vec4 surface = texture(uSurface, vUv);
    const vec3 hdr = texture(uHdr, vUv).rgb;

    if (uDebugView == 5) {
        float bestWeight = -1.0;
        float bestVersion = 0.0;
        for (int i = 0; i < uDebugLightCount; ++i) {
            const vec3 delta = uDebugLights[i].positionRange.xyz - world.xyz;
            const float range = uDebugLights[i].positionRange.w;
            const float weight = 1.0 - clamp(length(delta) / max(range, 0.001),
                                              0.0, 1.0);
            if (weight > bestWeight) {
                bestWeight = weight;
                bestVersion = float(uDebugLights[i].meta.x);
            }
        }
        if (world.a <= 0.0 || bestWeight <= 0.0) {
            outColor = vec4(vec3(0.015, 0.018, 0.024), 1.0);
        } else {
            const vec3 versionColor = hueColor(fract(bestVersion * 0.173 + 0.07));
            outColor = vec4(versionColor * (0.28 + 0.72 * bestWeight), 1.0);
        }
        return;
    }

    if (uDebugView == 6) {
        const vec3 background = vec3(0.012, 0.016, 0.022);
        if (vUv.y > 0.88 && vUv.y < 0.98) {
            const float inner = step(vUv.y, 0.965) * step(0.895, vUv.y);
            const float filled = step(vUv.x, clamp(uCacheOccupancy, 0.0, 1.0));
            vec3 occupancyColor = mix(vec3(0.16, 0.95, 0.32),
                                     vec3(1.0, 0.16, 0.08),
                                     smoothstep(0.72, 0.98, uCacheOccupancy));
            outColor = vec4(mix(background, occupancyColor,
                                filled * inner), 1.0);
        } else {
            outColor = vec4(background, 1.0);
        }
        return;
    }

    vec3 color = vec3(0.0);
    if (uDebugView == 1) {
        color = world.a > 0.0 ? world.xyz / 48.0 + 0.5
                              : vec3(0.015, 0.018, 0.024);
    } else if (uDebugView == 2) {
        color = world.a > 0.0 ? normal.xyz * 0.5 + 0.5
                              : vec3(0.015, 0.018, 0.024);
    } else if (uDebugView == 3) {
        color = world.a > 0.0 ? albedo.rgb : vec3(0.015, 0.018, 0.024);
    } else if (uDebugView == 4) {
        color = world.a > 0.0
            ? vec3(surface.r, surface.g, 1.0 - surface.r)
            : vec3(0.015, 0.018, 0.024);
    } else {
        if (world.a > 0.0) {
            const vec3 ambient = albedo.rgb * 0.095;
            color = acesFilm((hdr + ambient) * uExposure);
            color = pow(color, vec3(1.0 / 2.2));
        } else {
            const float horizon = smoothstep(0.0, 0.65, vUv.y);
            color = mix(vec3(0.012, 0.016, 0.022),
                        vec3(0.035, 0.044, 0.060), horizon);
        }
    }
    outColor = vec4(color, 1.0);
}
