#version 450 core

const int MAX_LIGHTS = 32;

struct PackedLight {
    vec4 position;
    vec4 colorIntensity;
    vec4 directionType;
};

layout(std140, binding = 0) uniform LightBlock {
    int uLightCount;
    int uMaxLights;
    ivec2 uLightPadding;
    PackedLight uLights[MAX_LIGHTS];
};

layout(binding = 0) uniform sampler2D uDepth;

uniform mat4 uInvProjection;
uniform mat4 uInvView;
uniform vec3 uCameraPosition;
uniform int uSteps;
uniform float uDensityScale;
uniform float uMaxDistance;
uniform float uNoiseScale;
uniform float uAbsorption;
uniform float uScattering;

in vec2 vUv;
out vec4 fragColor;

uint hash3D(ivec3 p, uint seed) {
    uint h = seed;
    h ^= uint(p.x) * 0x8da6b343u;
    h ^= uint(p.y) * 0xd8163841u;
    h ^= uint(p.z) * 0xcb1ab31fu;
    h ^= h >> 16;
    h *= 0x7feb352du;
    h ^= h >> 15;
    h *= 0x846ca68bu;
    h ^= h >> 16;
    return h;
}

float valueNoise(vec3 p, uint seed) {
    ivec3 i = ivec3(floor(p));
    vec3 f = fract(p);
    vec3 u = f * f * (3.0 - 2.0 * f);
    float l000 = float(hash3D(i + ivec3(0, 0, 0), seed) & 0x00ffffffu) / 16777215.0;
    float l100 = float(hash3D(i + ivec3(1, 0, 0), seed) & 0x00ffffffu) / 16777215.0;
    float l010 = float(hash3D(i + ivec3(0, 1, 0), seed) & 0x00ffffffu) / 16777215.0;
    float l110 = float(hash3D(i + ivec3(1, 1, 0), seed) & 0x00ffffffu) / 16777215.0;
    float l001 = float(hash3D(i + ivec3(0, 0, 1), seed) & 0x00ffffffu) / 16777215.0;
    float l101 = float(hash3D(i + ivec3(1, 0, 1), seed) & 0x00ffffffu) / 16777215.0;
    float l011 = float(hash3D(i + ivec3(0, 1, 1), seed) & 0x00ffffffu) / 16777215.0;
    float l111 = float(hash3D(i + ivec3(1, 1, 1), seed) & 0x00ffffffu) / 16777215.0;
    float x00 = mix(l000, l100, u.x);
    float x10 = mix(l010, l110, u.x);
    float x01 = mix(l001, l101, u.x);
    float x11 = mix(l011, l111, u.x);
    float y0 = mix(x00, x10, u.y);
    float y1 = mix(x01, x11, u.y);
    return mix(y0, y1, u.z);
}

float fbm(vec3 p, uint seed) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int i = 0; i < 4; ++i) {
        value += amplitude * valueNoise(p, seed + uint(i) * 1013u);
        p = p * 2.03 + vec3(17.1, 9.2, 13.7);
        amplitude *= 0.5;
    }
    return value;
}

float densityAt(vec3 world) {
    vec3 p = world * uNoiseScale + vec3(3.7, 5.1, 1.9);
    float base = fbm(p, 0x51ed270bu);
    float wisps = max(0.0, base - 0.43) * 2.7;
    float height = clamp(1.0 - world.y / 10.0, 0.08, 1.0);
    float floorFade = smoothstep(0.02, 0.45, world.y);
    return uDensityScale * wisps * height * floorFade;
}

vec3 fogInscatter(vec3 position, vec3 rayDirection) {
    vec3 result = vec3(0.0);
    int count = min(uLightCount, min(uMaxLights, MAX_LIGHTS));
    for (int i = 0; i < count; ++i) {
        PackedLight light = uLights[i];
        if (light.directionType.w > 0.5) {
            float phase = 0.45 + 0.25 * max(dot(rayDirection, -light.directionType.xyz), 0.0);
            result += light.colorIntensity.rgb * (light.colorIntensity.w * 0.085 * phase);
        } else {
            vec3 delta = light.position.xyz - position;
            float distanceSq = max(dot(delta, delta), 0.05);
            vec3 dir = normalize(delta);
            float phase = 0.65 + 0.35 * max(dot(rayDirection, dir), 0.0);
            result += light.colorIntensity.rgb *
                      (light.colorIntensity.w * 0.42 * phase / (1.0 + distanceSq));
        }
    }
    return result;
}

void main() {
    vec2 ndc = vUv * 2.0 - 1.0;
    float depth = texture(uDepth, vUv).r;
    vec4 clip = vec4(ndc, depth * 2.0 - 1.0, 1.0);
    vec4 view = uInvProjection * clip;
    view /= view.w;
    vec3 world = (uInvView * vec4(view.xyz, 1.0)).xyz;
    vec3 ray = world - uCameraPosition;
    float rayLength = min(length(ray), uMaxDistance);
    vec3 rayDirection = normalize(ray);

    float steps = float(max(uSteps, 1));
    float dt = rayLength / steps;
    float transmittance = 1.0;
    vec3 scattered = vec3(0.0);
    for (int i = 0; i < uSteps; ++i) {
        float t = (float(i) + 0.5) * dt;
        vec3 samplePosition = uCameraPosition + rayDirection * t;
        float density = densityAt(samplePosition);
        if (density <= 0.0) continue;
        float sigma = density * (uAbsorption + uScattering);
        float sampleTransmittance = exp(-sigma * dt);
        float integrated = (1.0 - sampleTransmittance) / max(sigma, 1.0e-6);
        scattered += fogInscatter(samplePosition, rayDirection) *
                     (transmittance * density * uScattering * integrated);
        transmittance *= sampleTransmittance;
        if (transmittance < 0.002) break;
    }
    fragColor = vec4(scattered, 1.0 - transmittance);
}

