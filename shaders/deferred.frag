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

layout(binding = 0) uniform sampler2D uGPosition;
layout(binding = 1) uniform sampler2D uGNormal;
layout(binding = 2) uniform sampler2D uGAlbedo;
layout(binding = 3) uniform sampler2D uGMaterial;

uniform vec3 uCameraPosition;

in vec2 vUv;
out vec4 fragColor;

vec3 lightContribution(PackedLight light, vec3 position, vec3 normal, vec3 viewDir,
                       vec3 albedo, float roughness, float metallic) {
    vec3 lightDirection;
    float attenuation = 1.0;
    if (light.directionType.w > 0.5) {
        lightDirection = normalize(-light.directionType.xyz);
    } else {
        vec3 delta = light.position.xyz - position;
        float distance = length(delta);
        if (distance < 1.0e-5) return vec3(0.0);
        lightDirection = delta / distance;
        attenuation = 1.0 / (1.0 + 0.08 * distance + 0.055 * distance * distance);
    }
    float ndotl = max(dot(normal, lightDirection), 0.0);
    if (ndotl <= 0.0) return vec3(0.0);
    vec3 halfDir = normalize(lightDirection + viewDir);
    float ndoth = max(dot(normal, halfDir), 0.0);
    float shininess = 2.0 + 126.0 * (1.0 - roughness) * (1.0 - roughness);
    float specPower = pow(ndoth, shininess);
    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    float specStrength = (0.15 + 0.85 * (1.0 - roughness)) * (1.0 - roughness);
    vec3 diffuse = albedo * ((1.0 - metallic) / 3.14159265);
    vec3 specular = f0 * (specPower * specStrength * (shininess + 2.0) / (25.1327412));
    return (diffuse * ndotl + specular * ndotl) *
           light.colorIntensity.rgb * light.colorIntensity.w * attenuation;
}

void main() {
    vec4 position = texture(uGPosition, vUv);
    vec4 normalData = texture(uGNormal, vUv);
    vec4 albedoData = texture(uGAlbedo, vUv);
    vec4 materialData = texture(uGMaterial, vUv);

    if (materialData.w < 0.5 || position.w <= 0.0) {
        vec3 sky = mix(vec3(0.018, 0.025, 0.045), vec3(0.07, 0.12, 0.21),
                      clamp(vUv.y, 0.0, 1.0));
        fragColor = vec4(sky, 1.0);
        return;
    }

    vec3 worldPosition = position.xyz;
    vec3 normal = normalize(normalData.xyz);
    vec3 viewDir = normalize(uCameraPosition - worldPosition);
    vec3 color = vec3(0.028, 0.035, 0.052) + albedoData.rgb * 0.045;
    int count = min(uLightCount, min(uMaxLights, MAX_LIGHTS));
    for (int i = 0; i < count; ++i) {
        color += lightContribution(uLights[i], worldPosition, normal, viewDir,
                                   albedoData.rgb, albedoData.w, materialData.x);
    }
    fragColor = vec4(color, 1.0);
}

