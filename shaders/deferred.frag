#version 330 core

in vec2 vUV;

uniform sampler2D uPosition;
uniform sampler2D uNormal;
uniform sampler2D uMaterial;
uniform sampler2D uShadowMap;

uniform vec2 uResolution;
uniform vec3 uClearColor;
uniform vec3 uAmbient;
uniform mat4 uInvView;
uniform mat4 uLightViewProjection;
uniform int uShadowEnabled;
uniform float uShadowBias;
uniform float uShadowStrength;
uniform vec2 uShadowTexel;

uniform int uLightCount;
uniform int uLightType[16];
uniform vec3 uLightPosition[16];
uniform vec3 uLightDirection[16];
uniform vec3 uLightColor[16];
uniform float uLightIntensity[16];
uniform vec2 uLightCone[16];

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979323846;

float distributionGGX(vec3 n, vec3 h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float nDotH = max(dot(n, h), 0.0);
    float denominator = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denominator * denominator, 1.0e-5);
}

float geometrySchlickGGX(float nDotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return nDotV / max(nDotV * (1.0 - k) + k, 1.0e-5);
}

float geometrySmith(float nDotV, float nDotL, float roughness) {
    return geometrySchlickGGX(nDotV, roughness) *
           geometrySchlickGGX(nDotL, roughness);
}

vec3 fresnelSchlick(float cosTheta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float directionalShadow(vec3 viewPosition) {
    if (uShadowEnabled == 0) {
        return 1.0;
    }
    vec4 worldPosition = uInvView * vec4(viewPosition, 1.0);
    vec4 lightClip = uLightViewProjection * worldPosition;
    vec3 projected = lightClip.xyz / lightClip.w;
    vec2 uv = projected.xy * 0.5 + 0.5;
    if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0 ||
        projected.z < -1.0 || projected.z > 1.0) {
        return 1.0;
    }

    float depth = projected.z * 0.5 + 0.5;
    float visible = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float stored = texture(uShadowMap,
                                   uv + vec2(float(x), float(y)) * uShadowTexel).r;
            visible += depth - uShadowBias <= stored ? 1.0 : 0.0;
        }
    }
    visible /= 9.0;
    return mix(1.0, visible, uShadowStrength);
}

void main() {
    vec4 position = texture(uPosition, vUV);
    if (position.w < 0.5) {
        float height = gl_FragCoord.y / max(uResolution.y, 1.0);
        vec3 background = mix(uClearColor * 0.55, uClearColor * 1.35, height);
        outColor = vec4(background, 1.0);
        return;
    }

    vec3 viewPosition = position.xyz;
    vec3 normal = normalize(texture(uNormal, vUV).xyz);
    float roughness = clamp(texture(uNormal, vUV).w, 0.04, 1.0);
    vec4 material = texture(uMaterial, vUV);
    vec3 albedo = max(material.rgb, vec3(0.0));
    float metallic = clamp(material.a, 0.0, 1.0);
    float nDotV = max(dot(normal, normalize(-viewPosition)), 1.0e-4);
    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    vec3 diffuseColor = albedo * (1.0 - metallic);
    vec3 outgoing = vec3(0.0);

    for (int i = 0; i < uLightCount; ++i) {
        vec3 toLight = vec3(0.0);
        vec3 radiance = vec3(0.0);
        float shadow = 1.0;
        int lightType = uLightType[i];

        if (lightType == 0) {
            toLight = normalize(-uLightDirection[i]);
            radiance = uLightColor[i] * uLightIntensity[i];
            shadow = directionalShadow(viewPosition);
        } else {
            vec3 offset = uLightPosition[i] - viewPosition;
            float distanceToLight = length(offset);
            toLight = offset / max(distanceToLight, 1.0e-5);
            float attenuation = 1.0 /
                (1.0 + 0.14 * distanceToLight + 0.12 * distanceToLight * distanceToLight);
            radiance = uLightColor[i] * uLightIntensity[i] * attenuation;
            if (lightType == 2) {
                float coneCosine = dot(normalize(uLightDirection[i]), -toLight);
                float cone = smoothstep(uLightCone[i].y, uLightCone[i].x,
                                        coneCosine);
                radiance *= cone;
            }
        }

        float nDotL = max(dot(normal, toLight), 0.0);
        if (nDotL <= 0.0 || max(radiance.r, max(radiance.g, radiance.b)) <= 0.0) {
            continue;
        }
        vec3 halfVector = normalize(toLight + normalize(-viewPosition));
        float nDotH = max(dot(normal, halfVector), 0.0);
        float vDotH = max(dot(normalize(-viewPosition), halfVector), 0.0);
        float ndf = distributionGGX(normal, halfVector, roughness);
        float g = geometrySmith(nDotV, nDotL, roughness);
        vec3 fresnel = fresnelSchlick(vDotH, f0);
        vec3 specular = (ndf * g * fresnel) /
                        max(4.0 * nDotV * nDotL, 1.0e-5);
        vec3 diffuse = diffuseColor / PI;
        outgoing += (diffuse + specular) * radiance * nDotL * shadow;
    }

    float hemisphere = 0.58 + 0.42 * max(normal.y, 0.0);
    vec3 ambientDiffuse = diffuseColor * uAmbient * hemisphere;
    vec3 ambientSpecular = f0 * uAmbient * (0.16 + 0.12 * (1.0 - roughness));
    outColor = vec4(outgoing + ambientDiffuse + ambientSpecular, 1.0);
}

