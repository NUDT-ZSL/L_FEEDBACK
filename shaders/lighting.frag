#version 450 core

in vec2 vUv;
out vec4 outHdr;

layout(std140, binding = 0) uniform LightBatch {
    vec4 uPositionRange[11];
    vec4 uColorIntensity[11];
    ivec4 uMeta[11];
    ivec4 uCounts;
};

uniform sampler2D uWorldPosition;
uniform sampler2D uViewNormal;
uniform sampler2D uAlbedo;
uniform sampler2D uSurface;
uniform samplerCube uShadowMaps[11];
uniform mat4 uInverseView;
uniform vec3 uCameraPosition;
uniform float uShadowNear;

const float PI = 3.14159265359;

float distributionGgx(vec3 normal, vec3 halfVector, float roughness) {
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float nDotH = max(dot(normal, halfVector), 0.0);
    const float denominator = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denominator * denominator + 1.0e-6);
}

float geometrySchlick(float nDotV, float roughness) {
    const float k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    return nDotV / (nDotV * (1.0 - k) + k);
}

float pointShadow(int index, vec3 worldPosition, float farPlane) {
    const ivec4 meta = uMeta[index];
    if (meta.z == 0 || meta.x != meta.y) {
        return 1.0;
    }
    const vec3 offset = worldPosition - uPositionRange[index].xyz;
    const float distanceToLight = length(offset);
    if (distanceToLight >= farPlane) {
        return 1.0;
    }
    const vec3 direction = offset / max(distanceToLight, 1.0e-5);
    const float nearPlane = uShadowNear;
    const float zNdc = ((farPlane + nearPlane) / (farPlane - nearPlane)) -
        ((2.0 * farPlane * nearPlane) / (farPlane - nearPlane)) /
        max(distanceToLight, nearPlane + 1.0e-4);
    const float fragmentDepth = zNdc * 0.5 + 0.5;
    const vec3 axis = abs(direction.y) < 0.98
        ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    const vec3 tangent = normalize(cross(axis, direction));
    const vec3 bitangent = cross(direction, tangent);
    const float bias = 0.008;
    float visibility = 0.0;
    const vec2 offsets[5] = vec2[](
        vec2(0.0, 0.0),
        vec2(0.0032, 0.0),
        vec2(-0.0032, 0.0),
        vec2(0.0, 0.0032),
        vec2(0.0, -0.0032));
    for (int tap = 0; tap < 5; ++tap) {
        const vec3 sampleDirection = normalize(direction +
            tangent * offsets[tap].x + bitangent * offsets[tap].y);
        const float stored = texture(uShadowMaps[index], sampleDirection).r;
        visibility += fragmentDepth - bias <= stored ? 1.0 : 0.0;
    }
    return visibility / 5.0;
}

void main() {
    const vec4 world = texture(uWorldPosition, vUv);
    if (world.a <= 0.0) {
        outHdr = vec4(0.0);
        return;
    }
    const vec4 normalSample = texture(uViewNormal, vUv);
    const vec3 normal = normalize(mat3(uInverseView) * normalSample.xyz);
    const vec3 albedo = texture(uAlbedo, vUv).rgb;
    const vec4 surface = texture(uSurface, vUv);
    const float roughness = clamp(surface.r, 0.035, 1.0);
    const float metallic = clamp(surface.g, 0.0, 1.0);
    const vec3 viewDirection = normalize(uCameraPosition - world.xyz);

    vec3 radiance = vec3(0.0);
    for (int i = 0; i < uCounts.x; ++i) {
        const vec3 toLight = uPositionRange[i].xyz - world.xyz;
        const float distanceToLight = length(toLight);
        const float range = uPositionRange[i].w;
        if (distanceToLight >= range) {
            continue;
        }
        const vec3 lightDirection = toLight / max(distanceToLight, 1.0e-5);
        const float nDotL = max(dot(normal, lightDirection), 0.0);
        if (nDotL <= 0.0) {
            continue;
        }
        const vec3 halfVector = normalize(viewDirection + lightDirection);
        const float nDotV = max(dot(normal, viewDirection), 1.0e-4);
        const float nDotH = max(dot(normal, halfVector), 0.0);
        const float vDotH = max(dot(viewDirection, halfVector), 0.0);
        const float distribution = distributionGgx(normal, halfVector, roughness);
        const float geometry = geometrySchlick(nDotV, roughness) *
                               geometrySchlick(nDotL, roughness);
        const vec3 f0 = mix(vec3(0.04), albedo, metallic);
        const vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - vDotH, 5.0);
        const vec3 specular = (distribution * geometry * fresnel) /
            (4.0 * nDotV * nDotL + 1.0e-4);
        const vec3 diffuse = (1.0 - fresnel) * (1.0 - metallic) *
                             albedo / PI;
        const float window = clamp(1.0 - distanceToLight / range, 0.0, 1.0);
        const float attenuation = (window * window) /
                                  (distanceToLight * distanceToLight + 0.45);
        const float visibility = pointShadow(i, world.xyz, range);
        radiance += (diffuse + specular) *
                    uColorIntensity[i].rgb * uColorIntensity[i].a *
                    nDotL * attenuation * visibility;
    }
    outHdr = vec4(radiance, 1.0);
}
