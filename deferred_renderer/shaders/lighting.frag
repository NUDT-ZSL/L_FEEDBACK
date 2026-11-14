#version 330 core

in vec2 vUv;
out vec4 fragColor;

const int MAX_MATERIALS = 32;
const int MAX_POINT_LIGHTS = 16;
const float PI = 3.14159265359;

uniform sampler2D gPosition;   // slot 0, RGBA32F
uniform sampler2D gNormal;     // slot 1, RGB16F
uniform sampler2D gAlbedoMat;  // slot 2, RGBA8

uniform vec3 uCamPos;

uniform int uMaterialCount;
uniform float uRoughness[MAX_MATERIALS];
uniform float uMetallic[MAX_MATERIALS];
uniform vec3 uEmissive[MAX_MATERIALS];

struct PointLight {
    vec3 position;
    float range;
    vec3 color;
    float intensity;
};

uniform PointLight uPointLights[MAX_POINT_LIGHTS];
uniform int uPointLightCount;

uniform vec3 uDirDirection;
uniform vec3 uDirColor;
uniform float uDirIntensity;

// ---------------------------------------------------------------------------
// Half-float emulation. GLSL 3.30 has no native fp16, so round-tripping through
// these integer bit helpers reproduces IEEE-754 half precision deterministically
// on every driver (including software rasterizers). Used only with --fast-approx.
// ---------------------------------------------------------------------------
uint f32ToF16Bits(float value) {
    uint x = floatBitsToUint(value);
    uint sign = (x >> 16u) & 0x8000u;
    uint mantissa = x & 0x007FFFFFu;
    uint exponent = (x >> 23u) & 0x000000FFu;
    uint result;
    if (exponent <= 112u) {
        if (exponent < 103u) {
            result = sign;
        } else {
            mantissa |= 0x00800000u;
            uint shift = 113u - exponent;
            result = sign | (mantissa >> shift);
        }
    } else if (exponent >= 143u) {
        if (exponent == 255u && mantissa != 0u) {
            result = sign | 0x00007FFFu;
        } else {
            result = sign | 0x00007C00u;
        }
    } else {
        result = sign | ((exponent - 112u) << 10u) | (mantissa >> 13u);
    }
    return result;
}

float f16BitsToF32(uint h) {
    uint sign = (h & 0x8000u) << 16u;
    uint exponent = (h >> 10u) & 0x0000001Fu;
    uint mantissa = h & 0x000003FFu;
    uint result;
    if (exponent == 0u) {
        if (mantissa == 0u) {
            result = sign;
        } else {
            uint m = mantissa;
            uint e = 113u;
            while ((m & 0x00000400u) == 0u) {
                m <<= 1u;
                e -= 1u;
            }
            m &= 0x000003FFu;
            result = sign | (e << 23u) | (m << 13u);
        }
    } else if (exponent == 31u) {
        result = sign | 0x7F800000u | (mantissa << 13u);
    } else {
        result = sign | ((exponent + 112u) << 23u) | (mantissa << 13u);
    }
    return uintBitsToFloat(result);
}

float fp16(float v) {
    return f16BitsToF32(f32ToF16Bits(v));
}

vec3 fp16(vec3 v) {
    return vec3(fp16(v.x), fp16(v.y), fp16(v.z));
}

#ifdef FAST_APPROX
#define H1(x) fp16(x)
#define H3(x) fp16(x)
#else
#define H1(x) (x)
#define H3(x) (x)
#endif

float distributionGGX(float nDotH, float roughness) {
    float a = H1(roughness * roughness);
    float a2 = H1(a * a);
    float d = H1(nDotH * nDotH) * H1(a2 - 1.0) + 1.0;
    return H1(a2 / max(H1(PI * d * d), 0.0001));
}

float geometrySchlickGGX(float nDotX, float roughness) {
    float r = H1(roughness + 1.0);
    float k = H1(H1(r * r) / 8.0);
    return H1(nDotX / max(H1(nDotX * (1.0 - k)) + k, 0.0001));
}

vec3 fresnelSchlick(float cosTheta, vec3 f0) {
    return H3(f0 + (1.0 - f0) * H1(pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0)));
}

vec3 cookTorrance(vec3 n, vec3 v, vec3 l, vec3 albedo,
                  float roughness, float metallic) {
    vec3 h = normalize(v + l);
    float nDotL = H1(max(dot(n, l), 0.0));
    float nDotV = H1(max(dot(n, v), 0.0001));
    float nDotH = H1(max(dot(n, h), 0.0));
    float vDotH = H1(max(dot(v, h), 0.0));

    vec3 f0 = H3(mix(vec3(0.04), albedo, metallic));
    float d = distributionGGX(nDotH, roughness);
    float g = H1(geometrySchlickGGX(nDotV, roughness) *
                 geometrySchlickGGX(nDotL, roughness));
    vec3 f = fresnelSchlick(vDotH, f0);

    vec3 specular = H3((d * g * f) /
                       max(H1(4.0 * nDotV * nDotL), 0.0001));
    vec3 kd = H3((1.0 - f) * (1.0 - metallic));
    return H3((kd * H3(albedo / PI) + specular) * nDotL);
}

void main() {
    vec4 positionSample = texture(gPosition, vUv);
    vec3 worldPos = H3(positionSample.xyz);
    vec3 n = H3(normalize(texture(gNormal, vUv).xyz));
    vec4 albedoSample = texture(gAlbedoMat, vUv);
    vec3 albedo = H3(albedoSample.rgb);
    int materialId = clamp(int(albedoSample.a * 255.0 + 0.5), 0, uMaterialCount - 1);

    float roughness = H1(clamp(uRoughness[materialId], 0.04, 1.0));
    float metallic = H1(clamp(uMetallic[materialId], 0.0, 1.0));
    vec3 emissive = H3(uEmissive[materialId]);

    vec3 v = H3(normalize(uCamPos - worldPos));
    vec3 color = H3(albedo * 0.03) + emissive;

    // Directional light: no distance attenuation.
    vec3 dirL = H3(normalize(-uDirDirection));
    vec3 dirRadiance = H3(uDirColor * uDirIntensity);
    color += H3(cookTorrance(n, v, dirL, albedo, roughness, metallic) * dirRadiance);

    // Point lights with inverse-square falloff and a smooth range cutoff.
    for (int i = 0; i < uPointLightCount; ++i) {
        vec3 toLight = uPointLights[i].position - worldPos;
        float dist = H1(length(toLight));
        vec3 l = H3(toLight / max(dist, 0.0001));
        float cutoff = clamp(1.0 - pow(dist / max(uPointLights[i].range, 0.0001), 4.0),
                             0.0, 1.0);
        float attenuation = H1(cutoff * cutoff / max(dist * dist, 1.0));
        vec3 radiance = H3(uPointLights[i].color *
                           H1(uPointLights[i].intensity * attenuation));
        color += H3(cookTorrance(n, v, l, albedo, roughness, metallic) * radiance);
    }

    // Reinhard tone mapping + gamma correction before the 8-bit default FBO.
    color = H3(color / (color + vec3(1.0)));
    color = H3(pow(color, vec3(1.0 / 2.2)));
    fragColor = vec4(color, 1.0);
}
