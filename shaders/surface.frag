#version 430 core

in vec3 vWorldPosition;
in vec3 vWorldNormal;
in vec2 vTexCoord;

layout(location = 0) out vec4 outColor;

uniform sampler2D uSssLut;
uniform vec2 uLutCurvatureRange;
uniform vec3 uCameraPosition;

uniform vec3 uAlbedo;
uniform float uRoughness;
uniform float uAnisotropy;
uniform float uAnisotropyRotation;
uniform vec3 uSssColor;
uniform float uSssStrength;
uniform float uSssCurvature;
uniform float uSpecularF0;

uniform vec3 uLightCenter;
uniform vec3 uLightRight;
uniform vec3 uLightUp;
uniform vec3 uLightNormal;
uniform vec3 uLightColor;
uniform vec2 uLightSize;
uniform ivec2 uLightSampleCount;

uniform vec3 uAmbientSky;
uniform vec3 uAmbientGround;
uniform float uAmbientIntensity;

uniform float uExposure;
uniform float uGamma;
uniform int uToneMapMode;
uniform float uSeed;

const float PI = 3.14159265358979323846;

vec3 sampleSss(float nDotL, float curvature) {
    float u = clamp(nDotL * 0.5 + 0.5, 0.0, 1.0);
    float denominator = max(
        uLutCurvatureRange.y - uLutCurvatureRange.x,
        0.00001
    );
    float v = clamp(
        (curvature - uLutCurvatureRange.x) / denominator,
        0.0,
        1.0
    );
    return texture(uSssLut, vec2(u, v)).rgb;
}

vec3 fresnelSchlick(float cosTheta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float distributionGgxAniso(
    float nDotH,
    float tDotH,
    float bDotH,
    float alphaT,
    float alphaB
) {
    float alpha2 = alphaT * alphaB;
    vec3 d = vec3(alphaB * tDotH, alphaT * bDotH, alpha2 * nDotH);
    float dLength2 = max(dot(d, d), 0.0000001);
    float weight2 = alpha2 / dLength2;
    return alpha2 * weight2 * weight2 / PI;
}

float smithG1Aniso(
    float nDotX,
    vec3 x,
    vec3 tangent,
    vec3 bitangent,
    float alphaT,
    float alphaB
) {
    float xDotT = dot(x, tangent);
    float xDotB = dot(x, bitangent);
    float a2 = alphaT * alphaT * xDotT * xDotT
        + alphaB * alphaB * xDotB * xDotB;
    return 2.0 * nDotX / (nDotX + sqrt(a2 + nDotX * nDotX));
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec3 acesToneMap(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp(
        (color * (a * color + b)) / (color * (c * color + d) + e),
        0.0,
        1.0
    );
}

void main() {
    vec3 normal = normalize(vWorldNormal);
    vec3 view = normalize(uCameraPosition - vWorldPosition);
    float nDotV = max(dot(normal, view), 0.0001);

    vec3 dp1 = dFdx(vWorldPosition);
    vec3 dp2 = dFdy(vWorldPosition);
    vec2 duv1 = dFdx(vTexCoord);
    vec2 duv2 = dFdy(vTexCoord);
    float determinant = duv1.x * duv2.y - duv1.y * duv2.x;

    vec3 tangent;
    if (abs(determinant) < 0.000001) {
        vec3 helper = abs(normal.y) < 0.99
            ? vec3(0.0, 1.0, 0.0)
            : vec3(1.0, 0.0, 0.0);
        tangent = normalize(cross(helper, normal));
    } else {
        tangent = normalize(
            (dp1 * duv2.y - dp2 * duv1.y) / determinant
        );
    }
    tangent = normalize(tangent - normal * dot(normal, tangent));

    float rotationCos = cos(uAnisotropyRotation);
    float rotationSin = sin(uAnisotropyRotation);
    vec3 baseBitangent = normalize(cross(normal, tangent));
    tangent = normalize(rotationCos * tangent + rotationSin * baseBitangent);
    vec3 bitangent = normalize(cross(normal, tangent));

    float clampedAnisotropy = clamp(uAnisotropy, 0.0, 0.999);
    float aspect = sqrt(1.0 - 0.9 * clampedAnisotropy);
    float roughnessSquared = max(uRoughness * uRoughness, 0.0001);
    float alphaT = max(roughnessSquared / aspect, 0.0001);
    float alphaB = max(roughnessSquared * aspect, 0.0001);

    vec3 f0 = vec3(uSpecularF0);
    float curvature = clamp(
        uSssCurvature,
        uLutCurvatureRange.x,
        uLutCurvatureRange.y
    );

    int samplesX = max(uLightSampleCount.x, 1);
    int samplesY = max(uLightSampleCount.y, 1);
    int sampleCount = samplesX * samplesY;
    float area = max(uLightSize.x, 0.0) * max(uLightSize.y, 0.0);
    float deltaArea = area / float(sampleCount);

    vec3 directDiffuse = vec3(0.0);
    vec3 directSpecular = vec3(0.0);

    for (int y = 0; y < samplesY; ++y) {
        for (int x = 0; x < samplesX; ++x) {
            vec2 sampleIndex = vec2(float(x), float(y));
            vec2 jitter = vec2(
                hash12(sampleIndex + vec2(uSeed, 0.0)),
                hash12(sampleIndex.yx + vec2(17.0, uSeed))
            );
            vec2 lightUv = (sampleIndex + jitter)
                / vec2(float(samplesX), float(samplesY));
            vec2 centered = lightUv * 2.0 - 1.0;

            vec3 lightPosition = uLightCenter
                + uLightRight * (centered.x * uLightSize.x * 0.5)
                + uLightUp * (centered.y * uLightSize.y * 0.5);

            vec3 lightVector = lightPosition - vWorldPosition;
            float distanceSquared = max(dot(lightVector, lightVector), 0.0001);
            vec3 lightDirection = lightVector * inversesqrt(distanceSquared);
            float emitterCos = max(
                dot(uLightNormal, -lightDirection),
                0.0
            );
            if (emitterCos <= 0.0) {
                continue;
            }

            float nDotL = dot(normal, lightDirection);
            vec3 diffuseProfile = sampleSss(nDotL, curvature);
            vec3 lightRadiance = uLightColor
                * emitterCos
                * deltaArea
                / distanceSquared;

            directDiffuse += uAlbedo
                * uSssColor
                * uSssStrength
                * diffuseProfile
                * lightRadiance;

            float positiveNdl = max(nDotL, 0.0);
            if (positiveNdl > 0.0) {
                vec3 halfVector = normalize(lightDirection + view);
                float nDotH = max(dot(normal, halfVector), 0.0);
                float tDotH = dot(tangent, halfVector);
                float bDotH = dot(bitangent, halfVector);
                float vDotH = max(dot(view, halfVector), 0.0);

                float distribution = distributionGgxAniso(
                    nDotH,
                    tDotH,
                    bDotH,
                    alphaT,
                    alphaB
                );
                float g1L = smithG1Aniso(
                    positiveNdl,
                    lightDirection,
                    tangent,
                    bitangent,
                    alphaT,
                    alphaB
                );
                float g1V = smithG1Aniso(
                    nDotV,
                    view,
                    tangent,
                    bitangent,
                    alphaT,
                    alphaB
                );
                vec3 fresnel = fresnelSchlick(vDotH, f0);
                float denominator = max(4.0 * positiveNdl * nDotV, 0.0001);
                vec3 specularBrdf = distribution
                    * g1L
                    * g1V
                    * fresnel
                    / denominator;
                directSpecular += specularBrdf
                    * positiveNdl
                    * lightRadiance;
            }
        }
    }

    float hemisphere = 0.5 + 0.5 * normal.y;
    vec3 ambientRadiance = mix(
        uAmbientGround,
        uAmbientSky,
        hemisphere
    ) * uAmbientIntensity;
    vec3 ambientProfile = sampleSss(
        dot(normal, normalize(vec3(0.25, 0.92, 0.30))),
        curvature
    );
    vec3 ambientDiffuse = uAlbedo
        * uSssColor
        * uSssStrength
        * ambientProfile
        * ambientRadiance;
    vec3 ambientFresnel = fresnelSchlick(nDotV, f0);
    vec3 ambientSpecular = ambientFresnel
        * ambientRadiance
        * (0.10 + 0.18 * (1.0 - clamp(uRoughness, 0.0, 1.0)));

    vec3 color = directDiffuse
        + directSpecular
        + ambientDiffuse
        + ambientSpecular;
    color *= max(uExposure, 0.0);

    if (uToneMapMode == 0) {
        color = acesToneMap(color);
    } else if (uToneMapMode == 1) {
        color = color / (vec3(1.0) + color);
    }
    color = clamp(color, 0.0, 1.0);
    color = pow(color, vec3(1.0 / max(uGamma, 0.0001)));
    outColor = vec4(color, 1.0);
}

