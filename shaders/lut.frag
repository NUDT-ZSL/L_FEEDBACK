#version 430 core

in vec2 vUv;
layout(location = 0) out vec4 outLut;

uniform int uProfileSamples;
uniform int uCurvatureSamples;
uniform int uScatteringSamples;
uniform vec2 uCurvatureRange;
uniform vec3 uScatterColor;
uniform vec3 uScatterRadius;
uniform float uWrap;
uniform float uProfilePower;

void main() {
    float profileUv = gl_FragCoord.x / float(uProfileSamples);
    float curvatureUv = gl_FragCoord.y / float(uCurvatureSamples);

    float nDotL = profileUv * 2.0 - 1.0;
    float curvature = mix(
        uCurvatureRange.x,
        uCurvatureRange.y,
        curvatureUv
    );
    int integrationSamples = clamp(uScatteringSamples, 1, 256);
    vec3 integratedProfile = vec3(0.0);
    float weightSum = 0.0;
    for (int i = 0; i < integrationSamples; ++i) {
        float scatteringUv = (float(i) + 0.5) / float(integrationSamples);
        float spread = scatteringUv * 2.0 - 1.0;
        float localNdotL = clamp(
            nDotL + spread * curvature * 2.0,
            -1.0,
            1.0
        );
        float localWrapped = clamp(
            (localNdotL + uWrap) / (1.0 + uWrap),
            0.00001,
            1.0
        );
        float phase = pow(localWrapped, max(uProfilePower, 0.0001));
        vec3 transmittance = exp(
            -uScatterRadius * (1.0 / localWrapped - 1.0) * 8.0
        );
        float weight = exp(-abs(spread) * 1.5);
        integratedProfile += phase
            * (0.25 + 0.75 * transmittance)
            * weight;
        weightSum += weight;
    }
    integratedProfile /= max(weightSum, 0.00001);

    float curvatureAttenuation = mix(0.70, 1.0, curvatureUv);
    vec3 profile = uScatterColor
        * integratedProfile
        * curvatureAttenuation;

    outLut = vec4(max(profile, vec3(0.0)), 1.0);
}

