#version 330 core

in vec2 vUV;

uniform sampler2D uInput;
uniform float uExposure;
uniform int uMode;

out vec4 outColor;

vec3 acesFilm(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 color = max(texture(uInput, vUV).rgb, vec3(0.0)) * uExposure;
    if (uMode == 1) {
        color = color / (vec3(1.0) + color);
    } else {
        color = acesFilm(color);
    }
    outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}

