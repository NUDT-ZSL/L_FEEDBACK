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

in vec3 vWorldPosition;
in vec3 vWorldNormal;
in float vViewDepth;

uniform vec3 uCameraPosition;
uniform vec3 uAlbedo;
uniform vec3 uTint;
uniform float uOpacity;
uniform float uRoughness;
uniform float uMetallic;

out vec4 fragColor;

void main() {
    vec3 normal = normalize(vWorldNormal);
    vec3 viewDir = normalize(uCameraPosition - vWorldPosition);
    vec3 color = vec3(0.028, 0.035, 0.052) + uAlbedo * 0.045;
    int count = min(uLightCount, min(uMaxLights, MAX_LIGHTS));
    for (int i = 0; i < count; ++i) {
        PackedLight light = uLights[i];
        vec3 lightDirection;
        float attenuation = 1.0;
        if (light.directionType.w > 0.5) {
            lightDirection = normalize(-light.directionType.xyz);
        } else {
            vec3 delta = light.position.xyz - vWorldPosition;
            float distance = length(delta);
            if (distance < 1.0e-5) continue;
            lightDirection = delta / distance;
            attenuation = 1.0 / (1.0 + 0.08 * distance + 0.055 * distance * distance);
        }
        float ndotl = max(dot(normal, lightDirection), 0.0);
        if (ndotl <= 0.0) continue;
        vec3 halfDir = normalize(lightDirection + viewDir);
        float ndoth = max(dot(normal, halfDir), 0.0);
        float shininess = 2.0 + 126.0 * (1.0 - uRoughness) * (1.0 - uRoughness);
        float specPower = pow(ndoth, shininess);
        vec3 f0 = mix(vec3(0.04), uAlbedo, uMetallic);
        vec3 diffuse = uAlbedo * ((1.0 - uMetallic) / 3.14159265) * ndotl;
        vec3 specular = f0 * (specPower * (0.15 + 0.85 * (1.0 - uRoughness)) *
                              (1.0 - uRoughness) * (shininess + 2.0) / 25.1327412) * ndotl;
        color += (diffuse + specular) * light.colorIntensity.rgb *
                 light.colorIntensity.w * attenuation;
    }
    color *= uTint;
    fragColor = vec4(color * uOpacity, uOpacity);
}

