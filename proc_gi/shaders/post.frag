#version 330 core

in vec2 vUv;

uniform sampler2D uHdrTexture;
uniform float uExposure;
uniform bool uTonemap;

out vec4 outColor;

vec3 aces_tonemap(vec3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

void main()
{
    vec3 hdr = max(texture(uHdrTexture, vUv).rgb * uExposure, vec3(0.0));
    vec3 displayed;
    if (uTonemap) {
        displayed = pow(aces_tonemap(hdr), vec3(1.0 / 2.2));
    } else {
        // Debug view intentionally bypasses both ACES and gamma.
        displayed = clamp(hdr, 0.0, 1.0);
    }
    outColor = vec4(displayed, 1.0);
}

